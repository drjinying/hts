import os
import sys
import copy
import socket
import argparse
import hydra
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel, DataParallel
import torch.nn as nn
from torch.optim import lr_scheduler
import wandb
from omegaconf import DictConfig, OmegaConf
from rich import pretty, print
from tqdm import tqdm

from checkpointer import load_checkpoint, save_checkpoint
from dataloader import build_dataloader
from train import wandb_init, build_scheduler, build_optimizer, build_criterion
from evaluate import test_rec as run_test_ddp
from utils import Metric, mean_ap, build_model


class DistributedTrainer:
    def __init__(self, cfg, local_rank) -> None:
        self.cfg = cfg
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = local_rank
        self.local_gpus = torch.cuda.device_count()

        self.global_master = self.global_rank == 0
        self.local_master = self.local_rank == 0

        self.id_str = '[G %d/%d, L %d/%d]' % (self.global_rank, self.world_size, self.local_rank, self.local_gpus)

        if self.global_master:
            self.log('world size: %d' % self.world_size)
            os.makedirs(cfg.paths.ckpt_out_dir, exist_ok=True)
            os.makedirs(cfg.paths.log_dir, exist_ok=True)
            if not cfg.wandb:
                os.environ['WANDB_MODE'] = 'dryrun'
            wandb.login(key='use your own please!')
 
    def log(self, message, master_only=True):
        if (master_only and self.global_master) or (not master_only):
            print(self.id_str + ': ' + str(message))

    def train(self):
        if self.global_master:
            wandb_init(self.cfg)

        model = build_model(self.cfg).to(self.local_rank)
        model = DistributedDataParallel(model, device_ids=[self.local_rank], find_unused_parameters=False)

        if self.cfg.data.dataset == 'hico':
            train_loader, dist_sampler = build_dataloader(
                self.cfg, 'train' if self.cfg.data.validation else 'train_all', self.global_rank, self.world_size)
        else:
            train_loader, dist_sampler = build_dataloader(
                self.cfg, 'train', self.global_rank, self.world_size)

        self.cfg.data.n_iters = len(train_loader)

        criterion = build_criterion(self.cfg, train_loader)
        optimizer = build_optimizer(self.cfg, model)
        scheduler = build_scheduler(self.cfg, optimizer)
        
        for epoch in range(self.cfg.epochs):
            dist_sampler.set_epoch(epoch)
            metric = Metric()
            model.train()
            if self.global_master:
                wandb.log({'lr': optimizer.param_groups[0]['lr']}, epoch)

            for it, batch in enumerate(train_loader):
                optimizer.zero_grad()

                imgs, targets_onehot, known_obj, certain, idx = batch
                logits = model(imgs.cuda())
                loss = criterion(logits, targets_onehot.cuda())
                loss = loss.mean()  
                loss.backward()

                metric.add_val(loss.detach().item())
                optimizer.step()

                if self.cfg.scheduler.name in ['CosineAnnealingWarmRestarts']:
                    scheduler.step(epoch + it / self.cfg.data.n_iters)
                if self.cfg.scheduler.name in ['CosineAnnealingLR', 'CosineAnnealingWarmup']:
                    scheduler.step()

                if it % self.cfg.print_freq == 0:
                    self.log('Epoch: %3d/%3d, iter: %4d/%4d, loss: %.3f (%.3f)' 
                        % (epoch, self.cfg.epochs, it, len(train_loader), metric.val, metric.mean))

            if self.global_master:
                wandb.log(step=epoch, data={'loss': metric.mean})
            
            if (epoch + 1) % self.cfg.val_freq == 0 or epoch + 1 == self.cfg.epochs:
                self.log('run evaluation...')
                #mAPv = self.run_test(model, 'val', epoch)
                mAPt = self.run_test(model, 'test', epoch)

            if self.cfg.scheduler.name == 'MultiStepLR':
                scheduler.step()
            if self.cfg.scheduler.name == 'ReduceLROnPlateau':
                scheduler.step(mAPt)

            if (epoch + 1) % self.cfg.save_freq == 0 and self.global_master:
                save_checkpoint(epoch, model, optimizer, self.cfg.paths.ckpt_out_dir)

            dist.barrier()

        if self.global_master:
            save_checkpoint(epoch, model, optimizer, self.cfg.paths.ckpt_out_dir)
            wandb.finish()

    def run_test(self, model=None, split='test', epoch=0):
        if not self.global_master:
            mAP = torch.tensor(0, dtype=float).cuda()
        else:
            cfg2 = copy.deepcopy(self.cfg)
            cfg2.data.batch_size = int(cfg2.data.batch_size * torch.cuda.device_count())
            dp_model = DataParallel(model.module)
            mAP = run_test_ddp(cfg2, dp_model, split, epoch)
            del dp_model
            mAP = torch.tensor(mAP, dtype=float).cuda()

        dist.broadcast(mAP, 0)
        mAP = mAP.item()
        return mAP
    
    def run_test_fast_inaccurate(self, model=None, split='test', epoch=0):
        if model == None:
            print('using initial weights')
            model = build_model(self.cfg).to(self.local_rank)
            model = DistributedDataParallel(model, device_ids=[self.local_rank])

        dataloader, dist_sampler = build_dataloader(self.cfg, split, self.global_rank, self.world_size)

        logits = np.zeros([0, 600])
        labels = np.zeros([0, 600])
        known_objs = np.zeros([0, 600]).astype(bool)
        certains = np.zeros([0, 600]).astype(bool)

        with torch.no_grad():
            model.eval()
            for batch in tqdm(dataloader):
                imgs, targets_onehot, known_obj, certain, idx = batch
                s = model(imgs.cuda())
                s = s.detach().cpu().numpy()
                known_obj = known_obj.numpy()
                certain = certain.numpy()

                logits = np.vstack((logits, s))
                labels = np.vstack((labels, targets_onehot.cpu().numpy()))
                known_objs = np.vstack((known_objs, known_obj))
                certains = np.vstack((certains, certain))

        logits_gather = [torch.zeros_like(logits) for _ in range(self.world_size)]
        labels_gather = [torch.zeros_like(labels) for _ in range(self.world_size)]
        known_objs_gather = [torch.zeros_like(known_objs) for _ in range(self.world_size)]
        certains_gather = [torch.zeros_like(certains) for _ in range(self.world_size)]

        dist.all_gather(logits_gather, logits)
        dist.all_gather(labels_gather, labels)
        dist.all_gather(known_objs_gather, known_objs)
        dist.all_gather(certains_gather, certains)

        mAP = torch.tensor(0)
        if self.global_master:
            logits = np.vstack(logits_gather)
            labels = np.vstack(labels_gather)
            known_objs = np.vstack(known_objs_gather)
            certains = np.vstack(certains_gather)

            self.log('len data gathered: %d' % len(logits))

            if self.cfg.eval.save_result and split == 'test':
                label_out_fp = os.path.join(self.cfg.paths.log_dir, 'labels')
                if not os.path.isfile(label_out_fp):
                    np.save(label_out_fp, labels)

                np.save(os.path.join(self.cfg.paths.log_dir, '%s_out_ep%04d' % (split, epoch)), logits)

            mAP = mean_ap(logits, labels, certains)

            print('>> %s set-repeat, \tDefault \tmAP: %.3f' % (split, 100*mAP))
            wandb.log(step=epoch, data={
                'mAP-%s' % split: mAP
            })

            if self.cfg.eval.eval_known_obj:
                ko_certain = np.logical_and(known_objs, certains)
                mAP3 = mean_ap(logits, labels, ko_certain)
                print('           \tKO:     \tmAP: %.3f' % (100*mAP3))
                wandb.log(step=epoch, data={
                    'mAP-%s-ko' % split: mAP3,
                })
            
            mAP = torch.tensor(mAP)
            
        dist.broadcast(mAP, src=0)
        mAP = mAP.item()

        return mAP

@hydra.main(config_path='configs', config_name='defaults')
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    pretty.install()
    #print(OmegaConf.to_yaml(cfg))

    os.environ['NCCL_DEBUG'] = 'VERSION'

    if cfg.debug:
        torch.manual_seed(666)
        torch.set_deterministic(True)
        np.random.seed(0)
        cfg.wandb=False
        cfg.data.batch_size=4

    if cfg.local_rank is None:
        if cfg.local_rank == 0:
            print('OpenMPI mode.')
        
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        master_ip = os.environ['MASTER_IP'] if 'MASTER_IP' in os.environ else os.environ['AZ_BATCHAI_JOB_MASTER_NODE_IP']
        master_port = '6688'

        cfg.local_rank = local_rank

        master_uri = "tcp://%s:%s" % (master_ip, master_port)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=master_uri,
            rank=world_rank,
            world_size=world_size
        )
    else:
        if cfg.local_rank == 0:
            print('torch.distributed.launch mode.')
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    torch.cuda.set_device(cfg.local_rank)

    distributed_trainer = DistributedTrainer(cfg, cfg.local_rank)
    distributed_trainer.train()

if __name__ == '__main__':
    sys.argv = [x.replace('--local_rank=', 'local_rank=') for x in sys.argv]
    sys.argv = list(filter(lambda t: 'dummy' not in t, sys.argv))
    main()
