from functools import reduce
from itertools import permutations
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from packaging import version
import logging
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

def normalize(x, pos):
    norm = x.pow(2).sum(pos, keepdim=True).pow(1. / 2)
    out = x.div(norm + 1e-7)
    return out

def gram_matrix_on_channel(feat):
    B, T, D = feat.shape
    feat = normalize(feat, 1)   # normalize on T
    gram = torch.bmm(feat.transpose(1, 2), feat)    # (B, D, D)
    return gram

def gram_matrix_on_time(feat):
    B, T, D = feat.shape
    feat = normalize(feat, 2)   # normalize on D
    gram = torch.bmm(feat, feat.transpose(1, 2))    # (B, T, T)
    return gram
    
class StyleLoss(nn.Module):
    def __init__(self, on_channel=True):
        super().__init__()
        self.l2loss = nn.MSELoss(reduction='none')
        self.on_channel = on_channel

    def forward(self, src, tgt, lengths):
        B, T, D = src.shape
        assert tgt.shape == (B, T, D)
        assert lengths.shape == (B, )

        mask = make_pad_mask(lengths).unsqueeze(-1).to(src.device)
        assert mask.shape == (B, T, 1)
        src = src.masked_fill(mask, 0.0)    # (B, T, D)
        tgt = tgt.masked_fill(mask, 0.0)    # (B, T, D)

        if self.on_channel:
            gram_src = gram_matrix_on_channel(src)
            gram_tgt = gram_matrix_on_channel(tgt)
        else:
            gram_src = gram_matrix_on_time(src)
            gram_tgt = gram_matrix_on_time(tgt)
            
        loss = self.l2loss(gram_src, gram_tgt).sum() / B

        return loss    


