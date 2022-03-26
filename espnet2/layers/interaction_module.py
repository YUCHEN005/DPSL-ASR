#!/usr/bin/env python3


from  typing import Tuple
from typing import Optional

from typeguard import check_argument_types
import logging
import math
import torch
import torch.nn as nn

'''
by Yuchen on Aug 2, 2021
reference: 'Interactive Speech and Noise Modeling for Speech Enhancement'
https://arxiv.org/pdf/2012.09408.pdf
'''
class InteractionModule(nn.Module):
    def __init__(
        self,
        num_channels: int = 64,
    ):  
        check_argument_types()
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels),    
            nn.Sigmoid(),   
        )

    def forward(
        self,
        x1: torch.Tensor,   # (B, C, T, F)
        x2: torch.Tensor,   # (B, C, T, F)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input = torch.cat((x1, x2), dim=1)  # (B, 2*C, T, F)
        mask = self.conv(input)     # (B, C, T, F)
        output = x1 + x2 * mask     # (B, C, T, F)
        
        return output


