import torch
import logging
import torch.nn.functional as F
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from torch import nn

def padding_norm_zero(x1, x2, lengths):
    B, T, D = x1.shape
    assert x2.shape == (B, T, D)
    assert lengths.shape == (B, )
    mask = make_pad_mask(lengths).view(-1, 1).to(x1.device)   # (B*T, 1)
    x1, x2 = x1.flatten(0, 1), x2.flatten(0, 1) # (B*T, D)
    x1 = x1.masked_fill(mask, 0.0)    # (B*T, D)
    x2 = x2.masked_fill(mask, 0.0)    # (B*T, D)
    return x1, x2, mask.sum()
