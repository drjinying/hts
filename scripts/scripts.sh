# im1k_bert_sign is the name of a config file. change to other configs for reproducing other results
# im1k_bert_sign means 
#     - backbone pre-training is ImageNet1K, 
#     - Language Embedding Init used BERT as the language model
#     - loss function uses LSE-Sign loss

python -m torch.distributed.launch --nproc_per_node=8 train_dist.py +exp=im1k_bert_sign