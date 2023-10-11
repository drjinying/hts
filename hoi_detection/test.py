import sys
sys.path.insert(0,'./src')
sys.path.insert(0, './pocket')
from pocket.ops import relocate_to_cuda, relocate_to_cpu, to_tensor
from pocket.utils import DetectionAPMeter, HandyTimer, BoxPairAssociation, all_gather

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
from hoi_detection.dataloader import build_dataloader
from hicodet.hicodet import HICODet
from model import DefrDet
from torch.nn import DataParallel


def test(net, test_loader):
    testset = test_loader.dataset.dataset
    associate = BoxPairAssociation(min_iou=0.5)

    meter = DetectionAPMeter(
        600, nproc=1,
        num_gt=testset.anno_interaction,
        algorithm='11P'
    )
    net.eval()

    result_dict = {}
    for batch in tqdm(test_loader):
        image, image_info, detection, target = batch
        
        # image = relocate_to_cuda(image)
        # detection = relocate_to_cuda(detection)
        # target = relocate_to_cuda(target)
        image = torch.stack(image)
        image_idx = torch.tensor([x['img_idx'] for x in image_info])
        image_ori_size = torch.tensor([x['original_size'] for x in image_info])
        box_h = torch.vstack([x['box_h'] for x in detection])
        box_o = torch.vstack([x['box_o'] for x in detection])
        object_cls = torch.vstack([x['object_cls'] for x in detection])
        prior = torch.stack([x['prior'] for x in detection])
        target = torch.vstack(target)

        with torch.no_grad():
            # output = net(image, image_info, detection, target)
            # img_idx, boxes_h, boxes_o, scores, predictions = net(image, image_info, detection, target)
            img_idx, boxes_h, boxes_o, scores, predictions = net(image, image_idx, image_ori_size, box_h, box_o, object_cls, prior, target)

        for (_img_idx, box_h, box_o, _scores, _prior, interaction) in zip(img_idx, boxes_h, boxes_o, scores, prior, predictions):
            _img_idx = _img_idx.item()
            box_h = relocate_to_cpu(box_h)
            box_o = relocate_to_cpu(box_o)
            _scores = relocate_to_cpu(_scores)
            interaction = relocate_to_cpu(interaction)
            _prior = relocate_to_cpu(_prior)
            pred = _scores.nonzero().flatten()
            box_idx = torch.zeros(len(pred)).long()

            out_dict = dict(
                boxes_h = box_h[None][box_idx],
                boxes_o = box_o[None][box_idx],
                prior = _prior[:, pred].permute(1, 0),
                scores = _scores[pred],
                interactions = interaction[pred],
            )

        # for _out in output:
        #     img_idx = _out.pop('img_idx')
        #     img_fp = _out.pop('fp')
        #     _out = relocate_to_cpu(_out)

        #     box_idx = _out['index'].long()
        #     out_dict = dict(
        #         boxes_h = _out['boxes_h'][None][box_idx],
        #         boxes_o = _out['boxes_o'][None][box_idx],
        #         scores = _out['scores'],
        #         interactions = _out['prediction'],
        #     )
            
            if _img_idx not in result_dict:
                result_dict[_img_idx] = out_dict
                #result_dict[img_idx]['image_fp'] = img_fp
            else:
                for k, v in out_dict.items():
                    dest = result_dict[_img_idx][k]
                    if k.startswith('box'):
                        result_dict[_img_idx][k] = torch.vstack([dest, v])
                    else:
                        result_dict[_img_idx][k] = torch.cat([dest, v])
 
    import pickle
    with open('./result.pkl', 'wb') as f:
        pickle.dump(result_dict, f)

    for idx in range(len(testset)):
        if idx not in result_dict:
            continue
        
        output = result_dict[idx]
        
        intra_idx = testset._idx[idx]
        anno = to_tensor(testset._anno[intra_idx], input_format='dict')
        #assert testset._filenames[intra_idx] == output['image_fp']

        scores = output['scores']
        interactions = output['interactions']
        boxes_h = output['boxes_h']
        boxes_o = output['boxes_o']

        # Associate detected pairs with ground truth pairs
        labels = torch.zeros_like(scores)
        unique_hoi = interactions.unique()
        for hoi_idx in unique_hoi:
            gt_idx = torch.nonzero(anno['hoi'] == hoi_idx).squeeze(1)
            det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
            if len(gt_idx):
                labels[det_idx] = associate(
                    (anno['boxes_h'][gt_idx].view(-1, 4),
                    anno['boxes_o'][gt_idx].view(-1, 4)),
                    (boxes_h[det_idx].view(-1, 4),
                    boxes_o[det_idx].view(-1, 4)),
                    scores[det_idx].view(-1)
                )

        meter.append(scores, interactions, labels)

    return meter.eval()

def evaluate(model, cfg):
    # torch.cuda.set_device(0)
    # torch.backends.cudnn.benchmark = False

    dataloader = build_dataloader(cfg, 'test')

    num_anno = torch.tensor(HICODet(None, anno_file=os.path.join(
        cfg.paths.data_dir, 'detections/instances_train2015.json')).anno_interaction)
    rare = torch.nonzero(num_anno < 10).squeeze(1)
    non_rare = torch.nonzero(num_anno >= 10).squeeze(1)

    test_ap = test(model, dataloader)
    print("Full: {:.4f}, rare: {:.4f}, non-rare: {:.4f}".format(
        test_ap.mean(),
        test_ap[rare].mean(), test_ap[non_rare].mean()
    ))


if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('/home/eccv/defr/src/det/configs/defaults.yaml')
    cfg['paths'] = OmegaConf.load('/home/eccv/defr/src/det/configs/paths/aml.yaml')
    cfg['scheduler'] = OmegaConf.load('/home/eccv/defr/src/det/configs/scheduler/CosineAnnealingWarmup.yaml')

    cfg.data.dataset = 'hico'
    cfg.model.mask_layers = 1

    cfg.data.detection = '_finetuned_drg'
    cfg.data.batch_size = 4 * 8
    cfg.data.prefetch = 4
    cfg.model.cls_ckpt = 'clip-clip-sign-vit-b-p16.pt'

    model = DefrDet(cfg, postprocess=True).cuda()
    model = DataParallel(model)

    evaluate(model, cfg)