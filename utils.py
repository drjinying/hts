import json

import matplotlib.pylab as plt
import numpy as np
import torch

from clip.prompt_engr import OBJ, OBJ_TO_IND
from meters import AveragePrecisionMeter
from omegaconf import DictConfig, OmegaConf
from vit import build_model as build_model_vit
from clip.clip_hico import build_model as build_model_clip

def build_model(cfg: DictConfig):
    model = {
        'CLIP-ViT-B': build_model_clip,
        'CLIP-ViT-L': build_model_clip,
        'CLIP-RN50': build_model_clip,
        'CLIP-RN101': build_model_clip,
        'CLIP-RN50x4': build_model_clip,
        'MLM-ViT-B': build_model_vit,
        'ImageNet1k-ViT-B': build_model_vit,
        'ImageNet21k-ViT-B': build_model_vit,
    }[cfg.model.backbone](cfg)
    return model

few_idx = dict(
    few_1 = np.array([ 22,  63,  66,  76,  84,  90, 135, 166, 181, 184, 227, 234, 257, \
                    260, 262, 279, 280, 286, 289, 303, 311, 325, 351, 358, 379, 389, \
                    397, 401, 402, 416, 418, 427, 439, 485, 498, 499, 504, 509, 514, \
                    522, 526, 535, 539, 548, 551, 560, 586, 592, 593]),
    few_5 = np.array([  8,  22,  35,  44,  55,  62,  63,  66,  70,  76,  77,  83,  84, \
                    90,  99, 100, 107, 112, 127, 135, 149, 158, 166, 179, 181, 184, \
                    188, 189, 192, 195, 198, 205, 206, 216, 222, 227, 229, 231, 234, \
                    238, 239, 254, 255, 257, 260, 261, 262, 274, 279, 280, 281, 286, \
                    289, 292, 303, 311, 315, 317, 324, 325, 333, 334, 345, 350, 351, \
                    354, 358, 364, 379, 381, 389, 390, 395, 397, 398, 399, 401, 402, \
                    403, 405, 407, 410, 416, 418, 426, 427, 429, 431, 436, 439, 440, \
                    449, 463, 469, 474, 482, 485, 498, 499, 504, 509, 514, 517, 520, \
                    522, 526, 531, 535, 539, 546, 547, 548, 549, 551, 552, 555, 556, \
                    560, 578, 586, 592, 593, 596, 597, 599]),
    few_10 = np.array([  8,  22,  24,  27,  28,  35,  44,  50,  55,  62,  63,  66,  70, \
                    76,  77,  80,  83,  84,  90,  96,  99, 100, 104, 107, 112, 113, \
                    114, 127, 135, 136, 149, 151, 158, 165, 166, 168, 170, 171, 172, \
                    179, 181, 184, 188, 189, 192, 195, 198, 205, 206, 214, 216, 220, \
                    222, 227, 229, 231, 234, 238, 239, 254, 255, 257, 260, 261, 262, \
                    270, 274, 279, 280, 281, 286, 289, 292, 293, 303, 305, 311, 315, \
                    317, 324, 325, 328, 333, 334, 345, 350, 351, 354, 358, 364, 379, \
                    381, 389, 390, 391, 395, 397, 398, 399, 401, 402, 403, 404, 405, \
                    406, 407, 408, 410, 416, 418, 426, 427, 429, 431, 436, 439, 440, \
                    449, 450, 451, 463, 469, 473, 474, 482, 485, 498, 499, 501, 503, \
                    504, 509, 510, 511, 514, 517, 520, 522, 526, 531, 535, 539, 546, \
                    547, 548, 549, 550, 551, 552, 555, 556, 560, 578, 580, 581, 586, \
                    592, 593, 595, 596, 597, 599])
    )
    
class Metric:
    def __init__(self):
        self.cnt = 0
        self.val = 0
        self.total_val = 0
        self.mean = 0

    def add_val(self, val):
        self.cnt += 1
        self.total_val += val
        self.val = val
        self.mean = self.total_val / self.cnt

def mean_ap_vis(preds, labels, img_fp=None):
    pred = torch.tensor(preds)

    labels[labels == -1] = 0
    labels = torch.tensor(labels)

    meter = AveragePrecisionMeter(labels.sum(dim=0), algorithm='11P', output=pred, labels=labels)
    ap = meter.eval().cpu().numpy()
    
    if img_fp is not None:
        with open('/mnt/4t/hico/hoi_list.json') as f:
            hoi = np.array(json.load(f))

        hoi_names_by_obj = {}
        hoi_scores_by_obj = {}

        for _i, _ap in enumerate(ap):
            hoi_name = hoi[_i]
            obj_name = hoi_name.split(': ')[0]
            if obj_name not in hoi_names_by_obj:
                hoi_names_by_obj[obj_name] = []
                hoi_scores_by_obj[obj_name] = []
            hoi_names_by_obj[obj_name].append(hoi_name)
            hoi_scores_by_obj[obj_name].append(_ap)

        for obj_name in hoi_scores_by_obj.keys():
            obj_id = OBJ_TO_IND[obj_name]
            
            hoi_names = np.array(hoi_names_by_obj[obj_name])
            hoi_aps = np.array(hoi_scores_by_obj[obj_name])
            
            sort = np.argsort(hoi_aps)
            hoi_names = hoi_names[sort]
            hoi_aps = hoi_aps[sort]

            plt.figure(figsize=[6, 6])
            plt.barh(list(range(len(hoi_aps))), hoi_aps[::-1], color='green')
            plt.xlabel("HOI")
            plt.ylabel("AP")
            plt.title("HOI AP for " + obj_name)

            plt.yticks(list(range(len(hoi_aps))), hoi_names[::-1])
            plt.tight_layout()
            plt.savefig(img_fp.replace('.jpg', '_%d-%s.jpg' % (obj_id, obj_name)))
            plt.close()

    return ap.mean()

def mean_ap(preds, labels, select=None, mean=True):
    pred = torch.tensor(preds)
    labels = torch.tensor(labels)
    ignore = torch.tensor(np.invert(select))

    meter = AveragePrecisionMeter(labels.sum(dim=0), algorithm='11P', output=pred, labels=labels, ignore=ignore)
    ap, prec, recall = meter.eval()
    ap = ap.cpu().numpy()
    prec = prec.transpose(0,1).cpu().numpy()
    recall = recall.transpose(0,1).cpu().numpy()

    if mean:
        return ap.mean()
    else:
        return ap, prec, recall

    
if __name__ == '__main__':
    preds = np.random.rand(8, 3)
    labels = (np.random.rand(8, 3) > 0.5) * 1
    select = np.ones([8, 3]).astype(bool)

    mean_ap(preds, labels, select)
