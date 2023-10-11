"""
Run inference and cache detections
Fred Zhang <frederic.zhang@anu.edu.au>
The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import torch
import pickle
import argparse
import numpy as np
import scipy.io as sio

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader

import pocket

from utils import DataFactory, custom_collate
import pickle
import shutil

def main(result_fp, out_dir, h_p, o_p, i_p):
    with open(result_fp, 'rb') as f:
        results = pickle.load(f)

    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)

    data_root = '/mnt/4t/hico'
    dataloader = DataLoader(
        dataset=DataFactory(
            name='hicodet', partition='test2015',
            data_root=data_root,
            detection_root='/mnt/4t/hico/detections/test2015_gt',
        ), collate_fn=custom_collate, batch_size=1,
        num_workers=2, pin_memory=True
    )

    dataset = dataloader.dataset.dataset
    
    # Include empty images when counting
    nimages = len(dataset.annotations)
    all_results = np.empty((600, nimages), dtype=object)
    
    for image_idx, result in tqdm(results.items()):
        image_idx = dataset._idx[image_idx]

        boxes_h = result['boxes_h']
        boxes_o = result['boxes_o']

        # TODO: this is to fix the error of ViT-L models using 588 as image size!!!
        # boxes_h = boxes_h * 672 / 588
        # boxes_o = boxes_o * 672 / 588

        priors = result['prior']
        scores = result['scores']
        interactions = result['interactions']
        
        # adjust the power of the object probabilities
        # this follows existing work (SCG) to suppress 
        # overconfident objects
        scores_hoi = scores / priors.prod(dim=1)
        priors[:, 0] = priors[:, 0].pow(1/2.8).pow(h_p)
        priors[:, 1] = priors[:, 1].pow(1/2.8).pow(o_p)
        scores = scores_hoi.pow(i_p) * priors.prod(dim=1)

        # remove loss score predictions
        # thres = 0.00001
        # select = scores > thres
        # boxes_h = boxes_h[select]
        # boxes_o = boxes_o[select]
        # scores = scores[select]
        # interactions = interactions[select]

        # Convert box representation to pixel indices
        boxes_h[:, 2:] -= 1
        boxes_o[:, 2:] -= 1

        permutation = interactions.argsort()
        boxes_h = boxes_h[permutation]
        boxes_o = boxes_o[permutation]
        interactions = interactions[permutation]
        scores = scores[permutation]

        # Store results
        unique_class, counts = interactions.unique(return_counts=True)
        n = 0
        for cls_id, cls_num in zip(unique_class, counts):
            all_results[cls_id.long(), image_idx] = torch.cat([
                boxes_h[n: n + cls_num],
                boxes_o[n: n + cls_num],
                scores[n: n + cls_num, None]
            ], dim=1).numpy()
            n += cls_num

    # Replace None with size (0,0) arrays
    for i in range(600):
        for j in range(nimages):
            if all_results[i, j] is None:
                all_results[i, j] = np.zeros((0, 0))

    with open(os.path.join(data_root, 'coco80tohico80.json'), 'r') as f:
        coco2hico = json.load(f)
    object2int = dataset.object_to_interaction

    # Cache results
    for object_idx in tqdm(coco2hico, desc='writing files'):
        interaction_idx = object2int[coco2hico[object_idx]]
        sio.savemat(
            os.path.join(out_dir, 'detections_{}.mat'.format(object_idx.zfill(2))),
            dict(all_boxes=all_results[interaction_idx])
        )

    shutil.rmtree('/home/ying/Desktop/verify_hico_eval/ho-rcnn-master/evaluation/result', ignore_errors=True)
    # matlab -nodesktop -nodisplay

if __name__ == "__main__":
    result_fp = 'result_finetuned_drg_vit_l14.pkl'

    h_p = 6
    o_p = 3
    i_p = 1
    main(result_fp, '/Desktop/verify_hico_eval/ho-rcnn-master/output/defr/hico_det_test2015/prefix_iter_1', h_p, o_p, i_p)

    """
    h_p = 7
    o_p = 3
    
    mAP / mRec (full):      0.3237 / 0.7899
    mAP / mRec (rare):      0.3360 / 0.7669
    mAP / mRec (non-rare):  0.3201 / 0.7968
    """