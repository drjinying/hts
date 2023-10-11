import os
import torch
from pocket.utils import DetectionAPMeter, BoxPairAssociation, all_gather
from tqdm import tqdm
import numpy as np
from dataloader import build_dataloader as build_dataloader_rec
from hoi_detection.dataloader import build_dataloader as build_dataloader_det
from utils import mean_ap, few_idx
import hydra
from omegaconf import DictConfig, OmegaConf
from train import build_model
import wandb

def test_rec(cfg, model, split='test', epoch=0):
    dataloader = build_dataloader_rec(cfg, split)

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
            os.makedirs(cfg.paths.log_dir, exist_ok=True)
            np.save(label_out_fp, labels)
            np.save(os.path.join(cfg.paths.log_dir, 'certains'), certains)

        np.save(os.path.join(cfg.paths.log_dir, '%s_out_ep%04d' % (split, epoch)), S)

    AP, _, _ = mean_ap(S, labels, certains, mean=False)
    mAP = AP.mean()

    print('>> %s set, \tDefault \tmAP: %.3f' % (split, 100*mAP))
    wandb.log(step=epoch, data={
        'mAP-%s' % split: mAP
    })  

    for few in [1, 5, 10]:
        mAP_few = AP[few_idx['few_%d' % few]].mean()
        
        print('>> Few@%d \tmAP: %.3f' % (few, 100*mAP_few))
        wandb.log(step=epoch, data={
            'mAP-Few@%d' % few: mAP_few
        })

    if cfg.eval.eval_known_obj:
        ko_certain = np.logical_and(known_objs, certains)
        mAP3 = mean_ap(S, labels, ko_certain)
        print('           \tKO:     \tmAP: %.3f' % (100*mAP3))
        wandb.log(step=epoch, data={
            'mAP-%s-ko' % split: mAP3,
        })
    
    return mAP

def test_det(cfg, model, split='test', use_gt_det=False, epoch=0):
    dataloader = build_dataloader_det(cfg, split)
    associate = BoxPairAssociation(min_iou=0.5)

    meter = DetectionAPMeter(
        num_cls=600 if cfg.dataset == 'hico' else -1,
        nproc=1,
        num_gt=dataloader.dataset.num_gt,
        algorithm='11P'
    )

    with torch.no_grad():
        model.eval()

        for idx, batch in tqdm(enumerate(dataloader)):
            target = batch[-1][0]

            detections = {
                'boxes_h': torch.zeros([0, 4]),
                'boxes_o': torch.zeros([0, 4]),
                'object': torch.zeros([0]),
                'scores': torch.zeros([0])
            }

            if use_gt_det:
                for box_h, box_o, obj in zip(target['boxes_h'], target['boxes_o'], target['object']):
                    if len(detections['boxes_h']) == 0 or associate(
                            (detections['boxes_h'].view(-1, 4),
                             detections['boxes_o'].view(-1, 4)),
                            (box_h.view(-1, 4),
                             box_o.view(-1, 4)),
                            torch.ones(1))[0] == 0:

                        detections['boxes_h'] = torch.vstack((detections['boxes_h'], box_h))
                        detections['boxes_o'] = torch.vstack((detections['boxes_o'], box_o))
                        detections['object'] = torch.hstack((detections['object'], obj))
                        detections['scores'] = torch.hstack((detections['scores'], torch.ones(len(obj))))
            else:
                od_result = batch[1][0]
                is_person = od_result['labels'] == 49
                idx_person = np.argwhere(is_person)[0]
                idx_obj = np.argwhere(np.logical_not(is_person))[0]

                for i_per in idx_person:
                    for i_obj in idx_obj:
                        detections['boxes_h'] = torch.vstack((detections['boxes_h'], od_result['boxes'][i_per]))
                        detections['boxes_o'] = torch.vstack((detections['boxes_o'], od_result['boxes'][i_obj]))
                        detections['object'] = torch.hstack((detections['object'], od_result['labels'][i_obj]))
                        detections['scores'] = torch.hstack((detections['scores'], od_result['scores'][i_obj] * od_result['scores'][i_per]))

            output = model(idx, batch[0][0].size(), detections)
            if output is None:
                continue

            # Format detections
            box_idx = torch.tensor(output['index'])
            boxes_h = output['boxes_h'][box_idx]
            boxes_o = output['boxes_o'][box_idx]
            hoi = torch.tensor(output['hoi'])
            scores = torch.tensor(output['scores'])
            interactions = torch.tensor(hoi)

            # Associate detected pairs with ground truth pairs
            labels = torch.zeros_like(scores)
            unique_hoi = interactions.unique()
            for hoi_idx in unique_hoi:
                gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                if len(gt_idx):
                    labels[det_idx] = associate(
                        (target['boxes_h'][gt_idx].view(-1, 4),
                        target['boxes_o'][gt_idx].view(-1, 4)),
                        (boxes_h[det_idx].view(-1, 4),
                        boxes_o[det_idx].view(-1, 4)),
                        scores[det_idx].view(-1)
                    )

            meter.append(scores, interactions, labels)

    rare = torch.nonzero(dataloader.dataset.num_gt < 10).squeeze(1)
    non_rare = torch.nonzero(dataloader.dataset.num_gt >= 10).squeeze(1)
    AP = meter.eval()
    mAP = AP.mean()
    mAP_rare = AP[rare].mean()
    mAP_non_rare = AP[non_rare].mean()

    return mAP


@hydra.main(config_path='configs', config_name='defaults')
def main(cfg):
    OmegaConf.resolve(cfg)
    os.environ['NCCL_DEBUG'] = 'VERSION'
    os.environ['WANDB_MODE'] = 'dryrun'
    # wandb may require you to login. Please refer to their website to register for a credential.
    wandb.init()

    model = build_model(cfg)
    ckpt = torch.load(cfg.ckpt_fp)['model_state_dict']
    ckpt = {k.replace('module.', ''): v for (k, v) in ckpt.items()}
    model.load_state_dict(ckpt)

    # larger batch size in evaluation mode
    cfg.data.batch_size *= 2
    
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        cfg.data.batch_size *= n_gpu
        model = torch.nn.DataParallel(model.cuda())

    test_rec(cfg, model)
    wandb.finish()

if __name__ == '__main__':
    main()
