import os
import json
import torch
from hoi_detection.transforms import DefrDetTransform
import pocket

from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.ops.boxes import box_iou
import torchvision.ops.boxes as box_ops
from torchvision.transforms.functional import hflip
from torch.utils.data.distributed import DistributedSampler
from hoi_detection.pair_generator import PairGenerator
from hoi_detection.hicodet.hicodet import HICODet
from typing import Optional, List, Tuple
from tqdm import trange

def custom_collate(batch):
    images = []
    images_info = []
    detections = []
    targets = []
    for im, im_info, det, tar in batch:
        images.append(im)
        images_info.append(im_info)
        detections.append(det)
        targets.append(tar)
    return images, images_info, detections, targets

class DataFactory(Dataset):
    def __init__(self,
            cfg, 
            partition,
            flip=False,
            box_score_thresh_h=0.2,
            box_score_thresh_o=0.2
            ):

        name = cfg.data.dataset
        data_root=cfg.paths.data_dir
        detection_root=os.path.join(cfg.paths.data_dir, 'detections', '%s%s' % (partition, cfg.data.detection))

        if name not in ['hico', 'vcoco']:
            raise ValueError("Unknown dataset ", name)

        if name == 'hico':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, 'images', partition),
                anno_file=os.path.join(data_root, 'detections/instances_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            self.human_idx = 49
        else:
            raise NotImplemented()
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, 'instances_vcoco_{}.json'.format(partition)
                ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            self.human_idx = 1

        self.cfg = cfg
        self.name = name
        self.detection_root = detection_root

        self.box_score_thresh_h = box_score_thresh_h
        self.box_score_thresh_o = box_score_thresh_o
        self._flip = torch.randint(0, 2, (len(self.dataset),)) if flip \
            else torch.zeros(len(self.dataset))

        self.image_fp, self.image_idx, self.boxes_h, self.boxes_o, self.object_cls, self.labels, self.priors = self.convert_by_instance()
        
        image_mean = (0.48145466, 0.4578275, 0.40821073)
        image_std = (0.26862954, 0.26130258, 0.27577711)
        self.transform = DefrDetTransform(cfg.model.image_size, image_mean, image_std)

    def convert_by_instance(self):
        pair_generator = PairGenerator(self.cfg, self.dataset.object_to_interaction)
        pair_generator.eval()

        image_fp = []
        image_idx = []
        boxes_h = []
        boxes_o = []
        object_cls = []
        labels = []
        priors = []
        
        for i in trange(len(self.dataset._idx)):
            intra_idx = self.dataset._idx[i]
            image_fp.append(self.dataset._filenames[intra_idx])
            target = pocket.ops.to_tensor(self.dataset._anno[intra_idx], input_format='dict')

            if self.name == 'hico':
                target['labels'] = target['hoi']
                # Convert ground truth boxes to zero-based index and the
                # representation from pixel indices to coordinates
                target['boxes_h'][:, :2] -= 1
                target['boxes_o'][:, :2] -= 1
            else:
                target['labels'] = target['actions'] # TODO: change to hoi
                target['object'] = target.pop('objects')

            detection_path = os.path.join(
                self.detection_root,
                self.dataset.filename(i).replace('jpg', 'json')
            )
            with open(detection_path, 'r') as f:
                detection = pocket.ops.to_tensor(json.load(f),
                    input_format='dict')
            
            all_boxes_h, all_boxes_o, all_object_class, all_labels, all_prior \
                = pair_generator([detection], [target])

            for i_pair in range(len(all_boxes_h[0])):
                _box_h = all_boxes_h[0][i_pair]
                _box_o = all_boxes_o[0][i_pair]
                _obj_cls = all_object_class[0][i_pair]
                _label = all_labels[0][i_pair]
                _prior = all_prior[0][:, i_pair, :]

                image_idx.append(i)
                boxes_h.append(_box_h)
                boxes_o.append(_box_o)
                object_cls.append(_obj_cls)
                labels.append(_label)
                priors.append(_prior)

        print('Dataset converted to by-instance mode')
        print('length: ', len(image_idx))
        return image_fp, image_idx, boxes_h, boxes_o, object_cls, labels, priors

    def __len__(self):
        return len(self.image_idx)

    def flip_boxes(self, detection, target, w):
        detection['boxes'] = pocket.ops.horizontal_flip_boxes(w, detection['boxes'])
        target['boxes_h'] = pocket.ops.horizontal_flip_boxes(w, target['boxes_h'])
        target['boxes_o'] = pocket.ops.horizontal_flip_boxes(w, target['boxes_o'])

    def __getitem__(self, i):
        image, _ = self.dataset[self.image_idx[i]]
        box_h = self.boxes_h[i]
        box_o = self.boxes_o[i]
        object_cls = self.object_cls[i]
        label = self.labels[i]
        prior = self.priors[i]

        if self._flip[self.image_idx[i]]:
            image = hflip(image)
            w, _ = image.size
            box_h = pocket.ops.horizontal_flip_boxes(w, box_h)
            box_o = pocket.ops.horizontal_flip_boxes(w, box_o)

        image = pocket.ops.to_tensor(image, 'pil')
        detection = {
            'box_h': box_h,
            'box_o': box_o,
            'object_cls': object_cls,
            'prior': prior,
        }

        original_image_size = image.shape[-2:]
        image, detection = self.transform([image], [detection])
        image = image.tensors[0]
        detection = detection[0]

        # for evaluating box noise robustness, the coordinates could exceed image area
        # detection['box_h'] = box_ops.clip_boxes_to_image(detection['box_h'], original_image_size)
        # detection['box_o'] = box_ops.clip_boxes_to_image(detection['box_o'], original_image_size)

        image_info = {
            'img_idx': self.image_idx[i],
            'fp': self.image_fp[self.image_idx[i]],
            'original_size': original_image_size
        }
        target = label

        return image, image_info, detection, target


def build_dataloader(cfg, split, rank=None, num_replicas=None):
    partition = {'train': 'train2015', 'test': 'test2015'}[split]
    dataset = DataFactory(
        cfg,
        partition=partition
    )

    if (rank, num_replicas) == (None, None):
        dataloader = DataLoader(
            dataset, 
            batch_size=cfg.data.batch_size, 
            shuffle = (split in ['train', 'train_all']), 
            num_workers=cfg.data.num_workers,
            prefetch_factor=cfg.data.prefetch,
            pin_memory=False,
            drop_last=False,
            collate_fn=custom_collate
        )
        return dataloader
    else:
        sampler = DistributedSampler(
            dataset, 
            num_replicas, 
            rank, 
            shuffle=split in ['train', 'train_all'], 
            drop_last=False
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers // 8,
            prefetch_factor=cfg.data.prefetch,
            pin_memory=True,
            collate_fn=custom_collate,
            sampler=sampler)
    
        return dataloader, sampler