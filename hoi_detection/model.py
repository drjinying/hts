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

from pocket.ops import relocate_to_cuda

from transforms import DefrDetTransform

from hoi_head import HoiHead
from torch import nn, Tensor
from torchvision.models.detection import transform
from torchvision.models.detection.transform import resize_boxes
from clip.clip_hico import build_model as build_model_clip
from criterion import FocalLossWithLogits, SignLoss
from dataloader import build_dataloader
from typing import Optional, List, Tuple
from clip.clip_hico import ClipViT

class DefrDet(nn.Module):
    """reference: Fred Zhang <frederic.zhang@anu.edu.au>"""
    def __init__(self, cfg, postprocess=False) -> None:
        super().__init__()

        self.cfg = cfg
        defr = self.load_cls_model()
        resblocks = defr.visual_encoder.visual.transformer.resblocks
        backbone_layers = nn.Sequential(*resblocks[:-cfg.model.mask_layers])
        masked_layers = nn.Sequential(*resblocks[-cfg.model.mask_layers:])
        
        self.backbone = defr.visual_encoder
        self.backbone.visual.transformer.resblocks = backbone_layers
        self.interaction_head = HoiHead(cfg, masked_layers, self.backbone.visual.ln_post, defr.classifier, defr.logit_scale)

        # values from CLIP
        if 'CLIP' in cfg.model.backbone.upper():
            image_mean = (0.48145466, 0.4578275, 0.40821073)
            image_std = (0.26862954, 0.26130258, 0.27577711)
        else:
            image_mean = (0.5, 0.5, 0.5)
            image_std = (0.5, 0.5, 0.5)

        self.transform = DefrDetTransform(cfg.model.image_size, image_mean, image_std)

        # If True, rescale bounding boxes to original image size
        self.postprocess = postprocess

    def load_cls_model(self):
        ckpt_fp = os.path.join(self.cfg.paths.data_dir, self.cfg.model.cls_ckpt)
        print('Loading hoi-cls checkpoint from ' + ckpt_fp)
        ckpt_module = torch.load(ckpt_fp, map_location='cpu')['model_state_dict']
        ckpt = {k.replace('module.', ''): v for k, v in ckpt_module.items()}
        
        model = ClipViT(self.cfg, return_attn=self.cfg.model.return_attn, return_all_patches=True)
        model.load_state_dict(ckpt)
        return model

    def preprocess(self,
        images: List[Tensor],
        detections: List[dict],
        targets: Optional[Tensor] = None
    ) -> Tuple[
        List[Tensor], List[dict],
        List[dict], List[Tuple[int, int]]
    ]:
        original_image_sizes = [img.shape[-2:] for img in images]
        images, detections = self.transform(images, detections)

        return images, detections, targets, original_image_sizes

    # def forward(self,
    #     images: List[Tensor],
    #     image_info: List[dict],
    #     detections: List[dict],
    #     targets: Optional[List[dict]] = None
    # ) -> List[dict]:
    def forward(self,
        images,
        image_idx,
        original_image_sizes,
        box_h,
        box_o,
        object_cls,
        prior,
        targets = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
            images: List[Tensor]
            detections: List[dict]
            targets: List[dict]

        Returns:
        --------
            results: List[dict]
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        detections = []
        for _box_h, _box_o, _object_cls, _prior in zip(box_h, box_o, object_cls, prior):
            detections.append({
                'box_h': _box_h,
                'box_o': _box_o,
                'object_cls': _object_cls,
                'prior': _prior
            })

        # images, detections, targets, original_image_sizes = self.preprocess(
        #         images, detections, targets)

        #features = self.backbone(images.tensors)
        features = self.backbone(images)
        results = self.interaction_head(features, detections, targets)

        # if self.training:
        #     loss = results.pop()
        # for res, info in zip(results, image_info):
        #     res.update(info)
        # if self.training:
        #     results.append(loss)

        if self.postprocess and results is not None:
            results = self.transform.postprocess(
                results,
                torch.ones_like(original_image_sizes) * self.cfg.model.image_size,
                original_image_sizes
            )
        
        boxes_h = torch.stack([x['boxes_h'] for x in results]).to(image_idx.device)
        boxes_o = torch.stack([x['boxes_o'] for x in results]).to(image_idx.device)
        scores = torch.stack([x['scores'] for x in results]).to(image_idx.device)
        prediction = torch.stack([x['prediction'] for x in results]).to(image_idx.device)

        return image_idx, boxes_h, boxes_o, scores, prediction

        return results


if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('/home/eccv/defr/src/det/configs/defaults.yaml')
    cfg.data.batch_size = 2
    cfg.data.dataset = 'hico'
    cfg.data.detection = '_finetuned_drg_noise_100'
    cfg['paths'] = OmegaConf.load('/home/eccv/defr/src/det/configs/paths/aml.yaml')
    cfg['scheduler'] = OmegaConf.load('/home/eccv/defr/src/det/configs/scheduler/CosineAnnealingWarmup.yaml')

    from hoi_detection.dataloader import build_dataloader
    dataloader = build_dataloader(cfg, 'test')

    if cfg.data.dataset == 'hico':
        human_idx = 49
        num_classes = 117
    else:
        human_idx = 1
        num_classes = 24

    model = DefrDet(cfg)
    model = model.cuda()
    model.train()

    for batch in dataloader:
        image, image_info, detection, target = batch
        
        image = relocate_to_cuda(image)
        detection = relocate_to_cuda(detection)
        target = relocate_to_cuda(target)

        out = model(image, image_info, detection, target)
        break

    print(out)