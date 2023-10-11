import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.ops.boxes as box_ops
import os
from torch.nn import Module
from torch import nn, Tensor
from pocket.ops import Flatten
from typing import Optional, List, Tuple
from collections import OrderedDict
from torchvision.models.detection.transform import resize_boxes
from pair_generator import PairGenerator
import sys
sys.path.append('./src')

from clip.layers import LayerNorm, QuickGELU
from clip.clip_hico import ClipViT

class HoiHead(Module):
    """Interaction head that constructs and classifies box pairs

    Parameters:
    -----------
    human_idx: int
        The index of human/person class in all objects
    num_classes: int
        Number of target classes
    """
    def __init__(self,
    cfg, transformer_layers, ln_post, classifier, logit_scale
    ) -> None:
        super().__init__()
        self.human_idx = 49 if cfg.data.dataset == 'hico' else 1
        self.box_nms_thresh = cfg.model.box_nms_thresh
        self.box_score_thresh = cfg.model.box_score_thresh
        self.max_human = cfg.model.max_human
        self.max_object = cfg.model.max_object
        self.cfg = cfg

        self.resblocks = transformer_layers
        self.n_head = self.resblocks[0].attn.num_heads
        self.ln_post = ln_post
        self.classifier = classifier
        self.logit_scale = logit_scale
        self.grid_w = cfg.model.image_size // cfg.model.patch_size
    
    def build_attention_masks(self, detections):
        L = self.grid_w**2 + 1
        N = len(detections)
        n_head = self.n_head

        device = detections[0]['box_h'].device
        masks = torch.zeros([N*n_head, L, L], dtype=torch.bool, device='cpu', requires_grad=False)

        for i in range(N):
            mask = masks[i*n_head : (i+1)*n_head, :, :]
            cls_img_attn_mask = mask[:, 0, 1:].view([n_head, self.grid_w, self.grid_w])
            cls_img_attn_mask.fill_(1)
        
            box_h = detections[i]['box_h']
            box_o = detections[i]['box_o']
            boxes = resize_boxes(torch.vstack([box_h, box_o]), (self.cfg.model.image_size, self.cfg.model.image_size), (self.grid_w, self.grid_w))
            for box in boxes:
                xmin, ymin, xmax, ymax = box.unbind(0)
                xmin = max(0, xmin.floor().int())
                ymin = max(0, ymin.floor().int())
                xmax = min(self.grid_w, xmax.ceil().int())
                ymax = min(self.grid_w, ymax.ceil().int())
                cls_img_attn_mask[:, ymin:ymax, xmin:xmax] = 0

        return masks

    def postprocess(self, detections, logits):
        logits = torch.sigmoid(logits)
        results = []
        for _det, _logits in zip(detections, logits):
            scores = _logits * _det['prior'].prod(dim=0)
            #pred = scores.nonzero().flatten()
            pred = torch.arange(len(scores)).long()
            result_dict = dict(
                boxes_h = _det['box_h'],
                boxes_o = _det['box_o'],
                index = torch.zeros(len(pred)),
                prediction = pred,
                scores = scores[pred]
            )
            results.append(result_dict)

        return results

    def hoi_loss(self, results):
        pred = []; labels = []
        for res in results:
            pred.append(res['scores'])


    def forward(self,
        features: Tensor,
        detections: List[dict],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
        features: OrderedDict
            Feature maps returned by FPN
        detections: List[dict]
            Object detections with the following keys
            `boxes`: Tensor[N, 4]
            `labels`: Tensor[N]
            `scores`: Tensor[N]
        image_shapes: List[Tuple[int, int]]
            Image shapes, heights followed by widths
        targets: List[dict], optional
            Interaction targets with the following keys
            `boxes_h`: Tensor[G, 4]
            `boxes_o`: Tensor[G, 4]
            `object`: Tensor[G]
                Object class indices for each pair
            `labels`: Tensor[G]
                Target class indices for each pair

        Returns:
        --------
        results: List[dict]
            Results organised by images. During training the loss dict is appended to the
            end of the list, resulting in the length being larger than the number of images
            by one. For the result dict of each image, refer to `postprocess` for documentation.
            The loss dict has two keys
            `hoi_loss`: Tensor
                Loss for HOI classification
            `interactiveness_loss`: Tensor
                Loss incurred on learned unary weights
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"
        
        if self.cfg.model.return_attn:
            features, attns = features

        attn_masks = self.build_attention_masks(detections)

        features = features.permute([1, 0, 2])
        if self.cfg.model.return_attn:
            x, attn_w = self.resblocks((features, attn_masks))
        else:
            x = self.resblocks((features, attn_masks))

        image_features = x[0]
        image_features = image_features[0, :, :]

        image_features = self.ln_post(image_features)
        
        if self.logit_scale != 0:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        if '-linear' in self.cfg.model.classifier:
            logits = self.classifier(self.logit_scale.exp() * image_features)
        else:
            text_features = self.classifier / self.classifier.norm(dim=-1, keepdim=True)
            logits = self.logit_scale.exp() * image_features @ text_features.t()
        
        results = self.postprocess(detections, logits)

        if self.training:
            loss_dict = dict(
                hoi_loss=self.hoi_loss(results),
            )
            results.append(loss_dict)

        return results