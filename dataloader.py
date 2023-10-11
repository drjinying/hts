import os
import pickle
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.transforms import (CenterCrop, ColorJitter, RandomAffine, RandomApply,
                                               RandomHorizontalFlip, RandomResizedCrop, RandomRotation)

class HICODataset(torch.utils.data.Dataset):
    idx_train = None
    """
    The HICO dataset has some "ambiguity" labels, see their paper for details
    label values:
        1:  obj & hoi positive
        0:  obj positive, hoi ambiguity
        -1: obj positive, hoi negative
        -2: obj negative (not labeled, can be treated as negative)

    splits:
        - 'train_all':  the whole HICO training set
        - 'train':      90% of 'train_all' for actual training
        - 'val':        the rest 10% of 'train_all' for validation
        - 'test':       HICO test set

        Note:
            Use torch.nn.DataParallel (train.py) when you use the validation set. 
            If you use DistributedDataParallel (train_dist.py), write additional code 
            to make sure the validation split is consistent cross all processes
    """
    def __init__(self, cfg, split):
        super(HICODataset, self).__init__()

        assert split in ['train', 'val', 'train_all', 'test']
        train_test_label = 'test' if split == 'test' else 'train'
        self.split = split

        self.image_dir = os.path.join(cfg.paths.data_dir, 'images/%s2015' % train_test_label)
        assert os.path.isdir(self.image_dir), "%s not found" % self.image_dir

        with open(os.path.join(cfg.paths.data_dir, 'anno.pkl'), 'rb') as f:
            d = pickle.load(f)

        self.hoi_info = d['hoi_list']
        self.img_list = d['files_%s' % train_test_label]
        targets = d['anno_%s' % train_test_label]

        # self.targets_onehot: 1: positive HOI, 0: all others, ambiguous
        temp = np.zeros_like(targets)
        temp[targets == 1] = 1
        self.targets_onehot = temp.transpose()

        # self.known_obj: 1: positive object, 0: all others, ambiguous
        temp = np.zeros_like(targets)
        temp[np.abs(targets) == 1] = 1
        self.known_obj = temp.transpose()

        # self.certain: 1: certain, 0: ambiguous/uncertain
        temp = np.zeros_like(targets)
        temp[targets != 0] = 1
        self.certain = temp.transpose()

        if HICODataset.idx_train is None:
            print('Splitting traing set to train and validation')
            HICODataset.idx_train = np.zeros(len(self.img_list)).astype(bool)
            ratio = 0.9
            is_train = np.random.choice(len(HICODataset.idx_train), int(ratio * len(self.img_list)), replace=False)
            HICODataset.idx_train[is_train] = True

        if split == 'train':
            select = HICODataset.idx_train
            # print('Ambiguity images removed from training')
            # select = np.logical_and(select, np.any(self.certain, axis=1))
            self.img_list = self.img_list[select]
            self.targets_onehot = self.targets_onehot[select]
            self.known_obj = self.known_obj[select]
            self.certain = self.certain[select]
        elif split == 'val':
            select = np.logical_not(HICODataset.idx_train)
            self.img_list = self.img_list[select]
            self.targets_onehot = self.targets_onehot[select]
            self.known_obj = self.known_obj[select]
            self.certain = self.certain[select]
            
        print('dataset len: ', len(self.img_list))
        resolution = cfg.model.image_size

        if cfg.data.min_samples > 1:
            if split not in ['val', 'test']:
                self.balance_dataset(cfg.data.min_samples)
                print('Rebalanced dataset with min_samples=%d to len: %d' % (cfg.data.min_samples, len(self.img_list)))

        if 'CLIP' in cfg.model.backbone:
            # values from CLIP
            norm_mean = (0.48145466, 0.4578275, 0.40821073)
            norm_std = (0.26862954, 0.26130258, 0.27577711)
        else:
            norm_mean = (0.5, 0.5, 0.5)
            norm_std = (0.5, 0.5, 0.5)

        if split == 'train' or split == 'train_all':
            self.transforms = transforms.Compose([
                RandomHorizontalFlip(),
                transforms.Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                ColorJitter(brightness=cfg.data.aug_bright),
                ColorJitter(contrast=cfg.data.aug_contra),
                RandomResizedCrop(size=resolution, scale=cfg.data.aug_crop_scale, ratio=cfg.data.aug_crop_ratio),
                lambda image: image.convert("RGB"),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                lambda image: image.convert("RGB"),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ])

    def __getitem__(self, index):
        img_fp = os.path.join(self.image_dir, self.img_list[index])
        img = self.transforms(Image.open(img_fp))

        targets_onehot = self.targets_onehot[index]
        known_obj = self.known_obj[index]
        certain = self.certain[index]

        targets_onehot = torch.tensor(targets_onehot, dtype=torch.float32)
        known_obj = torch.tensor(known_obj, dtype=torch.bool)
        certain = torch.tensor(certain, dtype=torch.bool)

        uncertain = torch.logical_not(certain)
        targets_onehot[uncertain] = 0

        return img, targets_onehot, known_obj, certain, index

    def __len__(self):
        return len(self.img_list)

    def get_pos_weights(self):
        cnt_positive = np.sum(self.targets_onehot, axis=0)
        cnt_negative = self.targets_onehot.shape[0] - cnt_positive
        return cnt_negative / (cnt_positive + np.finfo(float).eps)

    def balance_dataset(self, min_samples=10):
        """
        Guarentee a minimum number of samples for each class
        by repeating images in the train set
        """
        cnt_positive = np.sum(self.targets_onehot, axis=0)
        n_repeat = np.ceil(min_samples / cnt_positive).astype(int) - 1
        n_repeat[n_repeat > min_samples] = min_samples

        repeat_dict = {}

        for i in range(len(self.targets_onehot)):
            target = self.targets_onehot[i]
            max_repeat = (target * n_repeat).max()
            if max_repeat > 1:
                if i in repeat_dict:
                    repeat_dict[i] = max(max_repeat, repeat_dict[i])
                else:
                    repeat_dict[i] = max_repeat

        idx_append = []
        for k, v in repeat_dict.items():
            idx_append.extend([k] * v)            
        idx_append = np.array(idx_append)

        self.img_list = np.append(self.img_list, self.img_list[idx_append])
        self.targets_onehot = np.vstack((self.targets_onehot, self.targets_onehot[idx_append]))
        self.known_obj = np.vstack((self.known_obj, self.known_obj[idx_append]))
        self.certain = np.vstack((self.certain, self.certain[idx_append]))


class MPIIDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split) -> None:
        super().__init__()
        assert split in ['train', 'val', 'train_all', 'test']
        self.split = split
        self.image_dir = os.path.join(cfg.paths.data_dir, 'images')
        assert os.path.isdir(self.image_dir), "%s not found" % self.image_dir

        with open(os.path.join(cfg.paths.data_dir, 'anno.pkl'), 'rb') as f:
            d = pickle.load(f)
        
        if self.split == 'train':
            select = d['split'] == 0
        elif self.split == 'val':
            select = d['split'] == 1
        elif self.split == 'train_all':
            select = d['split'] <= 1
        else:
            select = d['split'] == 2

        self.img_list = d['img_names'][select]
        self.hoi = d['act_ids'][select]
        self.targets_onehot = np.zeros([len(self.img_list), 393], dtype=int)
        self.targets_onehot[np.arange(len(self.targets_onehot)), self.hoi] = 1

        print('MPII dataset len: ', len(self.img_list))
        resolution = cfg.model.image_size

        if cfg.data.min_samples > 1:
            if split not in ['val', 'test']:
                self.balance_dataset(cfg.data.min_samples)
                print('Rebalanced dataset with min_samples=%d to len: %d' % (cfg.data.min_samples, len(self.img_list)))

        if 'CLIP' in cfg.model.backbone:
            # values from CLIP
            norm_mean = (0.48145466, 0.4578275, 0.40821073)
            norm_std = (0.26862954, 0.26130258, 0.27577711)
        else:
            norm_mean = (0.5, 0.5, 0.5)
            norm_std = (0.5, 0.5, 0.5)

        if split == 'train' or split == 'train_all':
            self.transforms = transforms.Compose([
                RandomHorizontalFlip(),
                transforms.Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                ColorJitter(brightness=cfg.data.aug_bright),
                ColorJitter(contrast=cfg.data.aug_contra),
                RandomResizedCrop(size=resolution, scale=cfg.data.aug_crop_scale, ratio=cfg.data.aug_crop_ratio),
                lambda image: image.convert("RGB"),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                lambda image: image.convert("RGB"),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ])
    
    def __getitem__(self, index):
        img_fp = os.path.join(self.image_dir, self.img_list[index])
        img = self.transforms(Image.open(img_fp))

        targets_onehot = self.targets_onehot[index]

        targets_onehot = torch.tensor(targets_onehot, dtype=torch.float32)
        known_obj = torch.tensor(np.zeros(393), dtype=torch.bool)
        certain = torch.tensor(np.ones(393), dtype=torch.bool)

        return img, targets_onehot, known_obj, certain, index

    def __len__(self):
        return len(self.img_list)

    def get_pos_weights(self):
        cnt_positive = np.sum(self.targets_onehot, axis=0)
        cnt_negative = self.targets_onehot.shape[0] - cnt_positive
        return cnt_negative / (cnt_positive + np.finfo(float).eps)

    def balance_dataset(self, min_samples=10):
        """
        Guarentee a minimum number of samples for each class
        by repeating images in the train set
        """
        cnt_positive = np.sum(self.targets_onehot, axis=0)
        n_repeat = np.ceil(min_samples / cnt_positive).astype(int) - 1
        n_repeat[n_repeat > min_samples] = min_samples

        repeat_dict = {}

        for i in range(len(self.targets_onehot)):
            target = self.targets_onehot[i]
            max_repeat = (target * n_repeat).max()
            if max_repeat > 1:
                if i in repeat_dict:
                    repeat_dict[i] = max(max_repeat, repeat_dict[i])
                else:
                    repeat_dict[i] = max_repeat

        idx_append = []
        for k, v in repeat_dict.items():
            idx_append.extend([k] * v)            
        idx_append = np.array(idx_append)

        self.img_list = np.append(self.img_list, self.img_list[idx_append])
        self.targets_onehot = np.vstack((self.targets_onehot, self.targets_onehot[idx_append]))


def build_dataloader(cfg, split, rank=None, num_replicas=None):
    if cfg.data.dataset == 'hico':
        dataset = HICODataset(cfg, split)
    else:
        if split == 'test':
            print('Overrite MPII test split to val split')
            split = 'val'
        dataset = MPIIDataset(cfg, split)

    if (rank, num_replicas) == (None, None):
        dataloader = DataLoader(
            dataset, 
            batch_size=cfg.data.batch_size, 
            shuffle = (split in ['train', 'train_all']), 
            num_workers=cfg.data.num_workers,
            prefetch_factor=cfg.data.prefetch,
            pin_memory=True,
            drop_last=False
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
            sampler=sampler)
    
        return dataloader, sampler