import math
import os
import socket

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from omegaconf import DictConfig, OmegaConf
from rich import pretty, print
from tqdm import tqdm

from checkpointer import load_checkpoint, save_checkpoint
from criterion import FocalLossWithLogits, SignLoss
from dataloader import build_dataloader
from utils import Metric, mean_ap, build_model
from evaluate import test_rec as run_test


def wandb_init(cfg: DictConfig):
    wandb.init(
        project='defr',
        group=cfg.exp_group,
        name=cfg.exp_name,
        notes=cfg.exp_desc,
        save_code=True,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    OmegaConf.save(config=cfg, f=os.path.join(wandb.run.dir, 'conf.yaml'))

def build_criterion(cfg, train_loader):
    if cfg.loss.name == 'WBCE':
        pos_weights = train_loader.dataset.get_pos_weights()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights).cuda())
    elif cfg.loss.name == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    elif cfg.loss.name == 'Focal':
        criterion = FocalLossWithLogits(alpha=cfg.loss.alpha, gamma=cfg.loss.gamma)
    elif cfg.loss.name == 'Sign' or cfg.loss.name == 'SignLoss':
        criterion = SignLoss()
    else:
        raise ValueError
    
    return criterion

def build_optimizer(cfg, model):
    if cfg.optim.slow_lr_backbone:
        raise NotImplemented()
        # if cfg.model.backbone == 'ImageNet1k-ViT-B':
        #     params = []
        #     for k, v in model.module.named_parameters():
        #         if k.startswith('vit.'):
        #             params.append({'params': v, 'lr': cfg.optim.base_lr / 10})
        #         else:
        #             params.append({'params': v, 'lr': cfg.optim.base_lr})
        #     optimizer = torch.optim.AdamW(params, lr=cfg.optim.base_lr, weight_decay=cfg.optim.weight_decay)
        # elif cfg.model.backbone == 'CLIP-ViT-B':
        #     params = []
        #     for k, v in model.module.named_parameters():
        #         if k.startswith('visual_encoder.'):
        #             params.append({'params': v, 'lr': cfg.optim.base_lr / 10})
        #         else:
        #             params.append({'params': v, 'lr': cfg.optim.base_lr})
        #     optimizer = torch.optim.AdamW(params, lr=cfg.optim.base_lr, weight_decay=cfg.optim.weight_decay)
        # else:
        #     raise NotImplemented()
    else:
        # params = []
        # no_weight_decay = model.module.no_weight_decay()
        # for k, v in model.module.named_parameters():
        #     if k in no_weight_decay or 'bias' in k:
        #         params.append({'params': v, 'weight_decay': 0})
        #     else:
        #         params.append({'params': v})
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.base_lr, weight_decay=cfg.optim.weight_decay)
    
    return optimizer

class CosineAnnealingWarmupScheduler(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(CosineAnnealingWarmupScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
        
def build_scheduler(cfg, optimizer):
    if cfg.scheduler.name == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.scheduler.milestones, gamma=cfg.scheduler.gamma, verbose=False)
    elif cfg.scheduler.name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=cfg.scheduler.gamma,
            patience=cfg.scheduler.patience,
            threshold=cfg.scheduler.threshold,
            threshold_mode='abs',
            verbose=False
        )
    elif cfg.scheduler.name == 'CosineAnnealingWarmRestarts':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=cfg.scheduler.T_0,
                T_mult=cfg.scheduler.T_mult
            )
    elif cfg.scheduler.name == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.data.n_iters * cfg.epochs
        )
    elif cfg.scheduler.name == 'CosineAnnealingWarmup':
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer,
            cfg.data.n_iters * cfg.scheduler.warmup_t,
            cfg.data.n_iters * cfg.epochs
        )
    else:
        raise NotImplemented()

    return scheduler

def train(cfg):
    wandb_init(cfg)

    n_gpu = torch.cuda.device_count()
    print('CUDA Devices:', n_gpu)
    device = torch.device("cuda")

    model = build_model(cfg).to(device)

    if n_gpu > 1:
        cfg.data.batch_size *= n_gpu
        model = torch.nn.DataParallel(model)

    train_loader = build_dataloader(cfg, 'train' if cfg.data.validation else 'train_all')
    cfg.data.n_iters = len(train_loader)

    criterion = build_criterion(cfg, train_loader)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    for epoch in range(cfg.epochs):
        metric = Metric()
        model.train()

        wandb.log({'lr': optimizer.param_groups[0]['lr']}, epoch)

        for it, batch in enumerate(train_loader):
            optimizer.zero_grad()

            imgs, targets_onehot, known_obj, certain, idx = batch
            imgs = imgs.to(device)
            logits = model(imgs)

            loss = criterion(logits, targets_onehot.to(device))
            loss = loss.mean()
            loss.backward()

            metric.add_val(loss.detach().item())
            optimizer.step()

            if cfg.scheduler.name in ['CosineAnnealingWarmRestarts']:
                scheduler.step(epoch + it / cfg.data.n_iters)
            if cfg.scheduler.name in ['CosineAnnealingLR', 'CosineAnnealingWarmup']:
                scheduler.step()
            if it % cfg.print_freq == 0:
                print('Epoch: %3d/%3d, iter: %4d/%4d, loss: %.3f (%.3f)' % (epoch, cfg.epochs, it, len(train_loader), metric.val, metric.mean))

        wandb.log(step=epoch, data={'loss': metric.mean})

        if (epoch + 1) % cfg.val_freq == 0 or epoch + 1 == cfg.epochs:
            print('run evaluation...')
            #mAPv = run_test(cfg, model, 'val', epoch)
            mAPt = run_test(cfg, model, 'test', epoch)

        if cfg.scheduler.name == 'MultiStepLR':
            scheduler.step()
        if cfg.scheduler.name == 'ReduceLROnPlateau':
            scheduler.step(mAPt)
        if epoch % cfg.save_freq == 0:
            save_checkpoint(epoch, (model.module if n_gpu > 1 else model), optimizer, cfg.paths.ckpt_out_dir)

def run_test(cfg, model=None, split='test', epoch=0):
    if model == None:
        print('using initial weights')
        model = build_model(cfg).to('cuda')
        model = torch.nn.DataParallel(model)

    dataloader = build_dataloader(cfg, split)

    ncls = 600 if cfg.data.dataset == 'hico' else 393

    S = np.zeros([0, ncls])
    labels = np.zeros([0, ncls])
    known_objs = np.zeros([0, ncls]).astype(bool)
    certains = np.zeros([0, ncls]).astype(bool)

    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader):
            imgs, targets_onehot, known_obj, certain, idx = batch
            s = model(imgs.cuda())
            s = s.detach().cpu().numpy()
            known_obj = known_obj.numpy()
            certain = certain.numpy()

            S = np.vstack((S, s))
            labels = np.vstack((labels, targets_onehot.cpu().numpy()))
            known_objs = np.vstack((known_objs, known_obj))
            certains = np.vstack((certains, certain))

    if cfg.eval.save_result and split == 'test':
        label_out_fp = os.path.join(cfg.paths.log_dir, 'labels')
        if not os.path.isfile(label_out_fp):
            np.save(label_out_fp, labels)
            np.save(os.path.join(cfg.paths.log_dir, 'certains'), certains)

        np.save(os.path.join(cfg.paths.log_dir, '%s_out_ep%04d' % (split, epoch)), S)

    mAP = mean_ap(S, labels, certains)

    print('>> %s set, \tDefault \tmAP: %.3f' % (split, 100*mAP))
    wandb.log(step=epoch, data={
        'mAP-%s' % split: mAP
    })

    if cfg.eval.eval_known_obj:
        ko_certain = np.logical_and(known_objs, certains)
        mAP3 = mean_ap(S, labels, ko_certain)
        print('           \tKO:     \tmAP: %.3f' % (100*mAP3))
        wandb.log(step=epoch, data={
            'mAP-%s-ko' % split: mAP3,
        })
    
    return mAP

@hydra.main(config_path='configs', config_name='defaults')
def main(cfg):
    pretty.install()
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    
    os.environ['NCCL_DEBUG'] = 'VERSION'

    if not cfg.wandb:
        os.environ['WANDB_MODE'] = 'dryrun'

    wandb.login(key='use your own please!')

    os.makedirs(cfg.paths.ckpt_out_dir, exist_ok=True)
    os.makedirs(cfg.paths.log_dir, exist_ok=True)

    train(cfg)
    wandb.finish()

if __name__ == '__main__':
    main()
