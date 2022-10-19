#!/usr/bin/env python3


from  typing import Tuple
from typing import Optional

from typeguard import check_argument_types
import logging
import math
import torch
import torch.nn as nn
from espnet2.layers.ra_block import TemporalSelfAttention

'''
by Yuchen on Aug 2, 2021
reference: 'Interactive Speech and Noise Modeling for Speech Enhancement'
https://arxiv.org/pdf/2012.09408.pdf
'''

class MergeBranch(nn.Module):
    def __init__(
        self,
        num_channels: int = 4,
    ):  
        check_argument_types()
        super().__init__()
        self.before_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, (3, 3), (1, 1), (1, 1)),  # (3, 7)  
            nn.BatchNorm2d(num_channels),    
            nn.PReLU(),   
        )
        self.temp_self_att = TemporalSelfAttention(num_channels)
        self.after_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, (3, 3), (1, 1), (1, 1)),  # (3, 7)  
            nn.BatchNorm2d(num_channels),    
            nn.PReLU(),   
            nn.Conv2d(num_channels, 1, (3, 3), (1, 1), (1, 1)),  # (3, 7)
            nn.BatchNorm2d(1), 
            nn.Sigmoid(),
        )

    def forward(
        self,
        x1: torch.Tensor,   # (B, T, F)
        x2: torch.Tensor,   # (B, T, F)
        x1_ori: torch.Tensor,   # (B, T, F)
        x2_ori: torch.Tensor,   # (B, T, F)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_merged = torch.stack((x1, x2, x1_ori, x2_ori), dim=1)    # (B, 4, T, F)
        logits = self.before_conv(x_merged)     # (B, 4, T, F)
        logits = self.temp_self_att(logits)     # (B, 4, T, F)
        mask = self.after_conv(logits).squeeze(1)   # (B, T, F)
        output = x1 * mask + x2 * (1 - mask)    # (B, T, F)

        return output


