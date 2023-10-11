import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from dataset_stats import HICO

# https://distill.pub/2016/misread-tsne/

"""
This file generates the t-SNE visualizations in the paper
to show the embedding before the training (raw language embeddings)
and after the training (eventual weight vectors in the classifier)
"""

def extract_vit_weight(fp):
    m = torch.load(fp)['model_state_dict']
    if 'module.classifier' in m:
        return m['module.classifier'].detach().cpu().numpy()
    else:
        return m['module.classifier.weight'].detach().cpu().numpy()

def tsne(emb, do_pca=True, perplexity=70):
    X = emb # 600 x d
    n_comp = X.shape[1]
    y = np.arange(0, 600)

    cols = ['dim-%03d' % i for i in range(n_comp)]
    df = pd.DataFrame(X, columns=cols)
    df['y'] = y
    df_values = df[cols].values

    if do_pca:
        n_comp = 300
        print('PCA fitting')
        pca = PCA(n_components=n_comp)
        pca_emb = pca.fit_transform(df_values)
        print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))
        
        cols = ['pca-%03d' % i for i in range(n_comp)]
        for i in range(n_comp):
            df['pca-%03d' % i] = pca_emb[:, i]

    tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=3000)
    tsne_emb = tsne.fit_transform(df[cols].values)

    tsne_emb[:, 0] -= tsne_emb[:, 0].min()
    tsne_emb[:, 0] /= (tsne_emb[:, 0]).max()

    tsne_emb[:, 1] -= tsne_emb[:, 1].min()
    tsne_emb[:, 1] /= (tsne_emb[:, 1]).max()

    return tsne_emb

    # sns.scatterplot(
    #     x='tsne-2d-0', y='tsne-2d-1',
    #     hue='y',
    #     palette=sns.color_palette('hls', 600),
    #     data=df,
    #     legend='brief',
    #     alpha=0.7,
    #     ax=ax
    # )

    # ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    # ax.scatter(
    #     xs=df['tsne-2d-0'], 
    #     ys=df['tsne-2d-1'], 
    #     zs=df['tsne-2d-2'], 
    #     c=df["y"], 
    #     cmap='hsv'
    # )
    # ax.set_xlabel('pca-one')
    # ax.set_ylabel('pca-two')
    # ax.set_zlabel('pca-three')
    # plt.show()
    

def vis_tsne():
    w = torch.load('/mnt/4t/hico/classifier_weights')
    emb_clip = (torch.tensor(w['clip_proj']) @ torch.tensor(w['clip_w']).t()).t().detach().cpu().numpy()
    emb_clip_clip = extract_vit_weight('/mnt/4t/experiments/defr/clip-clip-sign-60.5/ckpt/checkpoint_0009.pt')
    # emb_clip_rand = extract_vit_weight('/mnt/4t/experiments/defr/clip-rand-sign-33.03/checkpoint_0019.pt')
    # emb_imnt_clip = extract_vit_weight('/mnt/4t/experiments/defr/im1k-clip-sign-51.27/checkpoint_0009.pt')
    # emb_imnt_rand = extract_vit_weight('/mnt/4t/experiments/defr/im1k-rand-sign-32.01/checkpoint_0009.pt')

    emb_clip_rand = extract_vit_weight('/mnt/4t/experiments/defr/clip-randlinear-sign-36.84/checkpoint_0019.pt')
    emb_imnt_clip = extract_vit_weight('/mnt/4t/experiments/defr/im1k-cliplinear-sign-54.73/checkpoint_0009.pt')
    emb_imnt_rand = extract_vit_weight('/mnt/4t/experiments/defr/im1k-randlinear-sign-44.07/checkpoint_0009.pt')
    
    emb_clip = tsne(emb_clip, True)
    emb_clip_clip = tsne(emb_clip_clip, True)
    emb_clip_rand = tsne(emb_clip_rand, True)
    emb_imnt_clip = tsne(emb_imnt_clip, True)
    emb_imnt_rand = tsne(emb_imnt_rand, True)
    
    xy = np.concatenate([emb_clip, emb_clip_clip, emb_imnt_clip, 
                        emb_clip, emb_clip_rand, emb_imnt_rand])
    df = pd.DataFrame(xy, columns=['TSNE dim-0', 'TSNE dim-1'])
    df['cls'] = np.concatenate([np.arange(600)]*6)
    df['model'] = ['emb_clip'] * 600 + ['emb_clip_clip'] * 600 + ['emb_imnt_clip'] * 600 + \
                    ['emb_clip2'] * 600 + ['emb_clip_rand'] * 600 + ['emb_imnt_rand'] * 600

    sns.set()
    grids = sns.relplot(
        data=df,
        x='TSNE dim-0', y='TSNE dim-1',
        col='model',
        hue='cls',
        palette=sns.color_palette('hls', 600),
        alpha=1,
        legend=False,
        col_wrap=3,
        s=80
    )

    grids.axes_dict['emb_clip'].set_title('CLIP Pretrained')
    grids.axes_dict['emb_clip_clip'].set_title('CLIP (Embedding Init.) Finetuned')
    grids.axes_dict['emb_imnt_clip'].set_title('ImageNet-1K (Embedding Init.) Finetuned')
    grids.axes_dict['emb_clip2'].set_title('Legend')
    grids.axes_dict['emb_clip_rand'].set_title('CLIP (Random Init.) Finetuned')
    grids.axes_dict['emb_imnt_rand'].set_title('ImageNet-1K (Random Init.) Finetuned')

    # hico = HICO()
    # for i in range(600):
    #     grids.axes_dict['emb_clip2'].annotate(hico.hois[i], (emb_clip[i][0], emb_clip[i][1]))

    plt.tight_layout()
    plt.show()

def vis_bert_tsne():
    emb_bert = torch.load('/mnt/4t/hico/classifier_weights')
    emb_bert = emb_bert['bert_w']

    emb_bert = tsne(emb_bert, True)

    emb_bert[:, 0] /= np.abs(emb_bert[:, 0]).max()
    emb_bert[:, 1] /= np.abs(emb_bert[:, 1]).max()

    df = pd.DataFrame(emb_bert, columns=['TSNE dim-0', 'TSNE dim-1'])
    df['cls'] = np.concatenate([np.arange(600)])
    df['model'] = ['BERT'] * 600

    sns.set()
    grids = sns.relplot(
        data=df,
        x='TSNE dim-0', y='TSNE dim-1',
        col='model',
        hue='cls',
        palette=sns.color_palette('hls', 600),
        alpha=1,
        legend=False,
        col_wrap=3
    )

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    np.random.seed(2980)
    #vis_bert_tsne()
    vis_tsne()