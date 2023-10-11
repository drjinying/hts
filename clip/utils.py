import torch
import torch.nn as nn
import os
from typing import Union, List
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.layers import CLIP, CLIP_Visual_Encoder
from clip.layers_ori import CLIP as CLIP_ORI
from clip.layers_ori import CLIP_Visual_Encoder as CLIP_ORI_Visual_Encoder
from clip.prompt_engr import *
# from simple_tokenizer import SimpleTokenizer as _Tokenizer
# from layers import CLIP, CLIP_Visual_Encoder
# from prompt_engr import *
import numpy as np
import scipy.ndimage

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)
    
def gen_text_features(data_dir, prompts, model_file):
    model_path = os.path.join(data_dir, model_file)
    state_dict = torch.jit.load(model_path, map_location='cpu').eval().state_dict()
    clip = build_clip_model(state_dict).to('cuda')

    hoi_text = tokenize(prompts).to('cuda')
    with torch.no_grad():
        clip.eval()
        hoi_text = clip.encode_text(hoi_text)
        hoi_text = hoi_text / hoi_text.norm(dim=-1, keepdim=True)
    
    if "visual.proj" in state_dict:
        clip_proj = clip.visual.proj.detach().clone()
    else:
        clip_proj = torch.eye(hoi_text.shape[1])

    del clip
    print('prompt tensors generated.')

    return hoi_text.detach(), clip_proj
    
def tokenize(texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]

    _tokenizer = _Tokenizer()
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def build_visual_encoder(cfg, state_dict: dict, return_attn=False, return_all_patches=False):
    vit = "visual.proj" in state_dict

    if vit:
        print('Building CLIP ViT')
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution_ori = vision_patch_size * grid_size
        pe_key = 'visual.positional_embedding'
    else:
        print('Building CLIP ResNet')
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution_ori = output_width * 32
        pe_key = 'visual.attnpool.positional_embedding'

    embed_dim = state_dict["text_projection"].shape[1]
    mask_layers = cfg.model.get('mask_layers', 0)

    if 'RN' in cfg.model.backbone:
        model = CLIP_ORI_Visual_Encoder(
            embed_dim,
            cfg.model.image_size, vision_layers, vision_width, vision_patch_size,
            return_attn=return_attn, return_all_patches=return_all_patches, mask_layers=mask_layers
        )
    else:
        model = CLIP_Visual_Encoder(
            embed_dim,
            cfg.model.image_size, vision_layers, vision_width, vision_patch_size,
            return_attn=return_attn, return_all_patches=return_all_patches, mask_layers=mask_layers
        )

        if cfg.model.image_size != image_resolution_ori:
            pe_ori = state_dict[pe_key]
            ntok_ori = pe_ori.shape[0]
            ngrid_ori = int((ntok_ori-1) ** 0.5)
            ngrid_new = cfg.model.image_size * ngrid_ori // image_resolution_ori

            pe_tok, pe_grid = pe_ori[:1, :], pe_ori[1:, :]

            pe_grid = pe_grid.reshape(-1, ngrid_ori, ngrid_ori, vision_width).permute(0, 3, 1, 2)
            pe_grid = torch.nn.functional.interpolate(
                pe_grid, size=(ngrid_new, ngrid_new), mode='bicubic', align_corners=False)
            pe_grid = pe_grid.permute(0, 2, 3, 1).detach().cpu().numpy()
            
            pe = torch.tensor(np.concatenate([
                pe_tok, 
                pe_grid.reshape(ngrid_new**2, -1)]
            ))
            state_dict[pe_key] = pe

            print('Converted Position Embedding from {} to {}'.format(pe_ori.shape, pe.shape))

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    #convert_weights(model)

    model.load_state_dict(state_dict, strict=False)
    return model

def build_clip_model(state_dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution_ori = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution_ori = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution_ori, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # if cfg.model.image_size != image_resolution_ori:
    #     pe_ori = state_dict['visual.positional_embedding']
    #     ntok_ori = pe_ori.shape[0]
    #     ngrid_ori = int((ntok_ori-1) ** 0.5)
    #     ngrid_new = cfg.model.image_size * ngrid_ori // image_resolution_ori

    #     pe_tok, pe_grid = pe_ori[:1, :], pe_ori[1:, :]

    #     pe_grid = pe_grid.reshape(-1, ngrid_ori, ngrid_ori, 768).permute(0, 3, 1, 2)
    #     pe_grid = torch.nn.functional.interpolate(
    #         pe_grid, size=(ngrid_new, ngrid_new), mode='bicubic', align_corners=False)
    #     pe_grid = pe_grid.permute(0, 2, 3, 1).detach().cpu().numpy()
        
    #     pe = torch.tensor(np.concatenate([
    #         pe_tok, 
    #         pe_grid.reshape(ngrid_new**2, -1)]
    #     ))
    #     state_dict['visual.positional_embedding'] = pe

    #     print('Converted Position Embedding from {} to {}'.format(pe_ori.shape, pe.shape))

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.float()