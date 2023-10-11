import json

import numpy as np
import timm
import torch
from clip.clip_hico import build_model as build_clip_model
from clip.utils import gen_text_features
from transformers import BertModel, BertTokenizer
from simcse import SimCSE

"""
This file generates the language embeddings used for Language Embedding Initialization.
The embeddings are generated, normalized and saved to a .pt file. It is loaded when model is built.

The input sentences are converted from HOI labels, we provide an example in prompts.json.
    - you may further tune the sentences to be more descriptive and fluent. This process is often
      called 'prompt engineering' in NLP and usually leads to better results
    - you may also do ensembling by writing the sentence in multiple ways and combine their embeddings

We provide embeddings from three kinds of transformer-based models: BERT, SimCSE and CLIP.    
    - BERT is trained in an unsupervised manner
    - SimCSE fine-tunes BERT so that the embedding space is isotropic. Similar work effectively enhance
      the BERT embeddings for some NLP tasks
    - CLIP (and Google's ALIGN) are image-languge jointly trained, so that
      the vision feature and sentence embedding is in the same vector space

Language Embedding Initialization provides considerable performance gain regardless of the language model.
"""


def generate_bert_embeddings(hico=True):
    bert_model = 'bert-base-uncased'
    tokenizer =  BertTokenizer.from_pretrained(bert_model)
    bert = BertModel.from_pretrained(bert_model).cuda().eval()

    prompt_fp = './prompts.json' if hico else './mpii/prompts.json'
    with open(prompt_fp) as f:
        prompts = json.load(f)

    tokens = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    emb = bert(input_ids.cuda(), attention_mask=attention_mask.cuda()).last_hidden_state
    emb = emb[:, 0, :]
    emb = emb / emb.norm(dim=-1, keepdim=True)
    emb = emb.detach().cpu().numpy()

    print(emb.shape)
    
    return emb

def generate_simcse_embeddings(hico=True):
    model = SimCSE('princeton-nlp/sup-simcse-bert-base-uncased')

    prompt_fp = '/home/ying/Desktop/defr/diagnose/prompts.json' if hico else '/mnt/4t/mpii/prompts.json'
    with open(prompt_fp) as f:
        prompts = json.load(f)

    emb = model.encode(prompts)

    emb = emb / emb.norm(dim=-1, keepdim=True)
    emb = emb.detach().cpu().numpy()

    print(emb.shape)
    
    return emb

def generate_clip_embeddings(hico=True, model='ViT-B-32'):
    prompt_fp = './prompts.json' if hico else './mpii/prompts.json'
    with open(prompt_fp) as f:
        prompts = json.load(f)

    # CLIP provides pre-trained models in different architectures. Use the one
    # that matches the backbone for the best result
    # {ViT-B-32, ViT-B-16, RN101, RN50, ViT-L-14}

    emb, proj = gen_text_features('/mnt/4t/hico', prompts, 'clip/%s.pt' % model)

    # The CLIP text embedding is 256-D, which does not match the visual feature
    # of 758-D from a standard ViT-B backbone. CLIP has a projection layer that
    # projects the 768-D visual feature to 256-D. Save both the embedding and the
    # projection matrix. See how we deal with this in the model loading part
    emb = emb.cpu().numpy()
    proj = proj.cpu().numpy()

    print(emb.shape)

    return emb, proj

def generate_cls_weights():
    """
    Save embeddings to a file so that the training script can load
    the corresponding one when building the model. It is called the CLS weights
    because the sentence embedding is the CLS output from the transformer-based
    language models.
    """

    out_fp = '.classifier_weights.pt'
    bert_emb = generate_bert_embeddings()
    simcse_emb = generate_simcse_embeddings()
    clip_emb, clip_proj = generate_clip_embeddings()

    torch.save({
        'bert_w': bert_emb,
        'simcse_w': simcse_emb,
        'clip_w': clip_emb,
        'clip_proj': clip_proj
    }, out_fp)


if __name__ == '__main__':
    generate_cls_weights()