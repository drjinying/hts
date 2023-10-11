from typing import Tuple
import torch
from torch import nn, Tensor


class SignLoss(nn.Module):
    def forward(self, class_logits, targets):
        assert class_logits.size() == targets.size(), "dimension mismatch"

        batch_size, n_classes = class_logits.size()
        zeros = torch.zeros(batch_size).view(batch_size, -1).to(targets.device)
        
        class_logits[targets == 1] *= -1
        class_logits = torch.cat((zeros, class_logits), dim=1)

        loss = torch.logsumexp(class_logits, 1)

        return loss

class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def extra_repr(self) -> str:
        return 'alpha=%f, gamma=%f' % (self.alpha, self.gamma)
    
    def forward(self, pred, target):
        sigmoid_pred = pred.sigmoid()
        log_sigmoid = nn.functional.logsigmoid(pred)
        loss = (target == 1) * self.alpha * torch.pow(1. - sigmoid_pred, self.gamma) * log_sigmoid

        log_sigmoid_inv = nn.functional.logsigmoid(-pred)
        loss += (target == 0) * (1 - self.alpha) * torch.pow(sigmoid_pred, self.gamma) * log_sigmoid_inv

        return -loss