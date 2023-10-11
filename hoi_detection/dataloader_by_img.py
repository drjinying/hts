import os
import json
import torch
import pocket

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.ops.boxes import box_iou
from torchvision.transforms.functional import hflip
from torch.utils.data.distributed import DistributedSampler
from hicodet.hicodet import HICODet

def custom_collate(batch):
    images = []
    detections = []
    targets = []
    for im, det, tar in batch:
        images.append(im)
        detections.append(det)
        targets.append(tar)
    return images, detections, targets

class DataFactory(Dataset):
    def __init__(self,
            name, partition,
            data_root, detection_root,
            flip=False,
            box_score_thresh_h=0.2,
            box_score_thresh_o=0.2
            ):
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

        self.name = name
        self.detection_root = detection_root

        self.box_score_thresh_h = box_score_thresh_h
        self.box_score_thresh_o = box_score_thresh_o
        self._flip = torch.randint(0, 2, (len(self.dataset),)) if flip \
            else torch.zeros(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def filter_detections(self, detection):  
        """Perform NMS and remove low scoring examples"""

        boxes = torch.as_tensor(detection['boxes'])
        labels = torch.as_tensor(detection['labels'])
        scores = torch.as_tensor(detection['scores'])

        # Filter out low scoring human boxes
        idx = torch.nonzero(labels == self.human_idx).squeeze(1)
        keep_idx = idx[torch.nonzero(scores[idx] >= self.box_score_thresh_h).squeeze(1)]

        # Filter out low scoring object boxes
        idx = torch.nonzero(labels != self.human_idx).squeeze(1)
        keep_idx = torch.cat([
            keep_idx,
            idx[torch.nonzero(scores[idx] >= self.box_score_thresh_o).squeeze(1)]
        ])

        boxes = boxes[keep_idx].view(-1, 4)
        scores = scores[keep_idx].view(-1)
        labels = labels[keep_idx].view(-1)

        return dict(boxes=boxes, labels=labels, scores=scores)

    def flip_boxes(self, detection, target, w):
        detection['boxes'] = pocket.ops.horizontal_flip_boxes(w, detection['boxes'])
        target['boxes_h'] = pocket.ops.horizontal_flip_boxes(w, target['boxes_h'])
        target['boxes_o'] = pocket.ops.horizontal_flip_boxes(w, target['boxes_o'])

    def __getitem__(self, i):
        image, target = self.dataset[i]
        if self.name == 'hico':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')

        detection_path = os.path.join(
            self.detection_root,
            self.dataset.filename(i).replace('jpg', 'json')
        )
        with open(detection_path, 'r') as f:
            detection = pocket.ops.to_tensor(json.load(f),
                input_format='dict')

        if self._flip[i]:
            image = hflip(image)
            w, _ = image.size
            self.flip_boxes(detection, target, w)
        image = pocket.ops.to_tensor(image, 'pil')

        return image, detection, target


def build_dataloader(cfg, split, rank=None, num_replicas=None):
    partition = {'train': 'train2015', 'test': 'test2015'}[split]
    dataset = DataFactory(
        name=cfg.data.dataset,
        partition=partition,
        data_root=cfg.paths.data_dir,
        detection_root=os.path.join(cfg.paths.data_dir, 'detections', '%s_%s' % (partition, cfg.data.detection))
    )

    if (rank, num_replicas) == (None, None):
        dataloader = DataLoader(
            dataset, 
            batch_size=cfg.data.batch_size, 
            shuffle = (split in ['train', 'train_all']), 
            num_workers=cfg.data.num_workers,
            prefetch_factor=cfg.data.prefetch,
            pin_memory=True,
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