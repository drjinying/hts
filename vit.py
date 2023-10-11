import torch
from torch._C import Value
from torch.jit import Error
import torch.nn as nn
from torch.nn.parameter import Parameter
import timm
import os

class VisionTransformer(nn.Module):    
    def __init__(self, cfg, return_attn=False):
        super(VisionTransformer, self).__init__()
        self.return_attn = return_attn
        self.cfg = cfg

        """Backbone
        """
        if cfg.model.backbone == 'MLM-ViT-B':
            if cfg.model.patch_size == 16:
                model_fp = os.path.join(cfg.paths.data_dir, 'msvlp/vit_b_16_224.pt')
                assert os.path.isfile(model_fp), "MLM pretrained model not found"
                param = torch.load(model_fp, map_location='cpu')
                if cfg.model.image_size != 224:
                    param = resize_model(param, cfg.model.patch_size, 224, cfg.model.image_size)

            elif cfg.model.patch_size == 32:
                model_fp = os.path.join(cfg.paths.data_dir, 'msvlp/vit_b_32_384.pt')
                assert os.path.isfile(model_fp), "MLM pretrained model not found"
                param = torch.load(model_fp, map_location='cpu')
                if cfg.model.image_size != 384:
                    param = resize_model(param, cfg.model.patch_size, 384, cfg.model.image_size)
            else:
                raise Error('patch size must be 16 or 32.')

            #model = timm.create_model('vit_base_patch%d_384'%cfg.model.patch_size, output_grid=False, num_classes=0, img_size=cfg.model.image_size, return_attn=self.return_attn)
            model = timm.create_model('vit_base_patch%d_384'% cfg.model.patch_size, num_classes=0, img_size=cfg.model.image_size)
            model.load_state_dict(param, strict=False)
            del param
        else:
            if cfg.model.backbone == 'ImageNet1k-ViT-B':
                model_key = 'vit_base_patch%d_224'%cfg.model.patch_size
            elif cfg.model.backbone == 'ImageNet21k-ViT-B':
                model_key = 'vit_base_patch%d_224_in21k'%cfg.model.patch_size
            else:
                raise ValueError(cfg.model.backbone + ' not supported by ImageNetViT')

            print('loading model ', model_key)
            #model = timm.create_model(model_key, num_classes=0, img_size=cfg.model.image_size, pretrained=True, return_attn=self.return_attn)
            model = timm.create_model(model_key, num_classes=0, img_size=cfg.model.image_size, pretrained=True)

        self.vit = model
        self.logit_scale = nn.Parameter(torch.tensor(cfg.model.log_scale), requires_grad=cfg.model.log_scale_trainable)

        """Classifier

        Initialize the classifier in different ways.
        
        "rand" is the conventional random initialization
        "bert, clip, ..." are language embedding initialization

        "rand-linear", "bert-linear"... means using a conventional linear classifier,
        which includes a weight and a bias (zero-initialized)

        those without "-linear" is cosine-similarity based. this only works better when
        both the backbone and the language model is CLIP pretrained

        there some other experiments for ablation study, like randomly shuffle the embeddings
        so that they mis-match the classes on purposely (to verify the language embedding works)
        """

        w = torch.load(os.path.join(cfg.paths.data_dir, self.cfg.model.classifier_fp))
        ncls = 600 if cfg.data.dataset == 'hico' else 393

        self.classifier_type = cfg.model.classifier
        if cfg.model.classifier == 'rand':
            self.classifier = nn.Parameter(768**-0.5*torch.randn([ncls, 768]), requires_grad=True)
        elif cfg.model.classifier == 'rand-linear':
            self.classifier = nn.Linear(768, ncls)
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
        elif cfg.model.classifier == 'simcse-linear':
            w = torch.tensor(w['simcse_w'])
            w = w / w.norm(dim=-1, keepdim=True)
            self.classifier = nn.Linear(768, ncls)
            self.classifier.weight.data.copy_(w)
            self.classifier.bias.data.zero_()
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

    def no_weight_decay(self):
        ret = self.vit.no_weight_decay()
        ret = set(['vit.' + x for x in ret])
        ret.add('logit_scale')
            
        return ret

    def forward(self, x):
        if self.logit_scale.requires_grad:
            self.logit_scale.data = self.logit_scale.clip(max=self.cfg.model.log_scale_max)

        if self.return_attn:
            image_features, a = self.vit(x)
            a = a.sum(1) / a.shape[1]
        else:
            image_features = self.vit(x)
  
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
            
        if self.return_attn:
            return logits_per_image, a

        return logits_per_image

def build_model(cfg, return_attn=False):
    return VisionTransformer(cfg, return_attn)

def resize_model(model, patch_size, in_im_size, out_im_size):
    key = 'pos_embed'
    origin_pos_embed = model[key]
    grid_before = in_im_size // patch_size
    assert (in_im_size % patch_size) == 0
    grid_after = out_im_size // patch_size
    assert (out_im_size % patch_size) == 0
    embed_dim = origin_pos_embed.shape[2]
    assert origin_pos_embed.shape[1] == grid_before * grid_before + 1

    pos_embed = origin_pos_embed[0, 1:, :].reshape((grid_before, grid_before, embed_dim))
    new_size = (grid_after, grid_after)
    pos_embed = torch.nn.functional.interpolate(pos_embed.permute((2, 0, 1)).unsqueeze(0), size=new_size, mode='bicubic')
    pos_embed = pos_embed.squeeze(0).permute((1, 2, 0)).reshape((-1, embed_dim))
    pos_embed = torch.cat((origin_pos_embed[0, 0:1, :], pos_embed), dim=0).unsqueeze(0)
    assert pos_embed.shape == (1, grid_after * grid_after + 1, embed_dim)
    model[key] = pos_embed

    return model

if __name__ == '__main__':
    # _resize_model()
    # exit(0)
    
    from easydict import EasyDict as edict
    cfg = edict({
        'paths': {
            'data_dir': '/mnt/4t/hico',
        },
        'model': {
            'backbone': 'ImageNet1k-ViT-B',
            'image_size': 672,
            'log_scale': 0,
            'classifier': 'clip-linear',
            'log_scale_trainable': False
        }
    })
    model = build_model(cfg, return_attn=True)
    print(model(torch.randn(1, 3, 672, 672)).shape)
    print('loaded')