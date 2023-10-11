import torch
import torch.nn as nn
import os
from typing import Union, List

import sys

from torchvision.transforms import functional

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.prompt_engr import *
# from simple_tokenizer import SimpleTokenizer as _Tokenizer
# from layers import CLIP, CLIP_Visual_Encoder
# from prompt_engr import *
import numpy as np
from clip.utils import convert_weights, gen_text_features, tokenize, build_visual_encoder, build_clip_model

class ClipViT(nn.Module):
    def __init__(self, cfg, return_attn=False, return_all_patches=False):
        super(ClipViT, self).__init__()
        self.cfg = cfg
        self.logit_scale = nn.Parameter(torch.tensor(cfg.model.log_scale), requires_grad=cfg.model.log_scale_trainable)
        self.return_attn = return_attn
        self.return_all_patches = return_all_patches

        if cfg.model.backbone == 'CLIP-ViT-B':
            model_path = os.path.join(cfg.paths.data_dir, 'clip/ViT-B-%d.pt'%cfg.model.patch_size)
        elif cfg.model.backbone == 'CLIP-ViT-L':
            model_path = os.path.join(cfg.paths.data_dir, 'clip/ViT-L-%d.pt'%cfg.model.patch_size)
        else:
            model_path = os.path.join(cfg.paths.data_dir, 'clip/%s.pt'%cfg.model.backbone.split('-')[1])
        state_dict = torch.jit.load(model_path, map_location='cpu').eval().state_dict()
        self.is_vit = "visual.proj" in state_dict

        self.visual_encoder = build_visual_encoder(
            cfg, state_dict, self.return_attn, self.return_all_patches)
        if self.is_vit:
            if cfg.model.classifier not in ['rand-512', 'rand-512-linear']:
                self.visual_encoder.visual.proj = None

        self.init_classifier(cfg)

    def no_weight_decay(self):
        return {'visual_encoder.visual.token_embedding', 'visual_encoder.visual.positional_embedding', 'logit_scale'}

    def init_classifier(self, cfg):
        w = torch.load(os.path.join(cfg.paths.data_dir, self.cfg.model.classifier_fp))
        ncls = 600 if cfg.data.dataset == 'hico' else 393

        """Classifier

        initialize the classifier in different ways.
        
        "rand" is the conventional random initialization
        "bert, clip, ..." are language embedding initialization

        "rand-linear", "bert-linear"... means using a conventional linear classifier,
        which includes a weight and a bias (zero-initialized)

        those without "-linear" is cosine-similarity based. this only works better when
        both the backbone and the language model is CLIP pretrained

        there some other experiments for ablation study, like randomly shuffle the embeddings
        so that they mis-match the classes on purposely (to verify the language embedding works)
        """

        self.classifier_type = cfg.model.classifier
        if cfg.model.classifier == 'rand':
            self.classifier = nn.Parameter(768**-0.5*torch.randn([ncls, 768]), requires_grad=True)
        elif cfg.model.classifier == 'rand-linear':
            self.classifier = nn.Linear(768, ncls)
        elif cfg.model.classifier == 'rand-512':
            self.classifier = nn.Parameter(512**-0.5*torch.randn([ncls, 512]), requires_grad=True)
        elif cfg.model.classifier == 'rand-512-linear':
            self.classifier = nn.Linear(512, ncls)
        elif cfg.model.classifier == 'clip':
            w = (torch.tensor(w['clip_proj']) @ torch.tensor(w['clip_w']).t()).t()
            self.classifier = nn.Parameter(w, requires_grad=True)
        elif cfg.model.classifier == 'clip-linear':
            w = (torch.tensor(w['clip_proj']) @ torch.tensor(w['clip_w']).t()).t()
            w = w / w.norm(dim=-1, keepdim=True)
            self.classifier = nn.Linear(768, ncls)
            self.classifier.weight.data.copy_(w)
            self.classifier.bias.data.zero_()
        elif cfg.model.classifier == 'clip-shuffle':
            w = (torch.tensor(w['clip_proj']) @ torch.tensor(w['clip_w']).t()).t()
            w = w[torch.randperm(ncls)]
            self.classifier = nn.Parameter(w, requires_grad=True)
        elif cfg.model.classifier == 'bert':
            w = torch.tensor(w['bert_w'])
            self.classifier = nn.Parameter(w, requires_grad=True)
        elif cfg.model.classifier == 'bert-linear':
            w = torch.tensor(w['bert_w'])
            w = w / w.norm(dim=-1, keepdim=True)
            self.classifier = nn.Linear(768, ncls)
            self.classifier.weight.data.copy_(w)
            self.classifier.bias.data.zero_()
        elif cfg.model.classifier == 'bert-shuffle':
            w = torch.tensor(w['bert_w'])
            w = w[torch.randperm(ncls)]
            self.classifier = nn.Parameter(w, requires_grad=True)
        elif cfg.model.classifier == 'i-clip':
            w = (torch.tensor(w['clip_proj']) @ torch.tensor(w['clip_w']).t()).t()
            w = w / w.norm(dim=-1, keepdim=True)
            self.m_classifier = nn.Parameter(torch.eye(ncls), requires_grad=True)
            self.classifier = nn.Parameter(w, requires_grad=False)
        elif cfg.model.classifier == 'i-bert':
            w = torch.tensor(w['bert_w'])
            w = w / w.norm(dim=-1, keepdim=True)
            self.m_classifier = nn.Parameter(torch.eye(ncls), requires_grad=True)
            self.classifier = nn.Parameter(w, requires_grad=False)
        else:
            raise NotImplemented

    def forward(self, images):
        if self.return_attn:
            image_features, image_attnw = self.visual_encoder(images)
            image_features = image_features[:, 1:, :]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = self.classifier / self.classifier.norm(dim=-1, keepdim=True)
            logits_per_image = image_features @ text_features.t()

            return logits_per_image, image_attnw[:, 0, 1:]

        if self.logit_scale.requires_grad:
            self.logit_scale.data = self.logit_scale.clip(max=self.cfg.model.log_scale_max)

        image_features = self.visual_encoder(images)

        if self.return_all_patches:
            image_features = image_features[:, 0, :]

        if self.logit_scale != 0:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if '-linear' in self.cfg.model.classifier:
            logits_per_image = self.classifier(self.logit_scale.exp() * image_features)
        else:
            if 'i-' in self.cfg.model.classifier:
                text_features = self.m_classifier @ self.classifier
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            else:
                text_features = self.classifier / self.classifier.norm(dim=-1, keepdim=True)
            logits_per_image = self.logit_scale.exp() * image_features @ text_features.t()

        if self.cfg.data.dataset == 'mpii':
            logits_per_image = logits_per_image.exp() / logits_per_image.exp().sum(dim=-1, keepdim=True)

        return logits_per_image


def build_model(cfg, return_attn=False):
    return ClipViT(cfg, return_attn)
