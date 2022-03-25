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
class TempSelfAtt(nn.Module):
    def __init__(
        self,
        num_channels: int = 4,
    ):  
        check_argument_types()
        super().__init__()
        self.conv_q = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels),    
            nn.PReLU(),  
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels),    
            nn.PReLU(),  
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels),    
            nn.PReLU(),  
        )
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels),    
            nn.PReLU(),  
        )

    def forward(
        self,
        x: torch.Tensor,   # (B, C, T, F)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, C, T, F = x.shape
        q = self.conv_q(x).permute(0, 2, 1, 3)  # (B, T, C, F)
        k = self.conv_k(x).permute(0, 2, 3, 1)  # (B, T, F, C)
        v = self.conv_v(x).permute(0, 2, 1, 3)  # (B, T, C, F)
        # print(f"q.shape: {q.shape}")
        # print(f"k.shape: {k.shape}")
        # print(f"v.shape: {v.shape}")
        qk = torch.softmax(torch.matmul(q, k) / math.sqrt(C*F), dim=-1)    # (B, T, C, C)
        logits = torch.matmul(qk, v).permute(0, 2, 1, 3)    # (B, C, T, F)
        output = self.conv(logits) + x  # (B, C, T, F)

        return output


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
        self.temp_self_att = TempSelfAtt(num_channels)
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
        # print(f"x_merged.shape: {x_merged.shape}")
        logits = self.before_conv(x_merged)     # (B, 4, T, F)
        # print(f"logits.shape: {logits.shape}")
        logits = self.temp_self_att(logits)     # (B, 4, T, F)
        # print(f"logits.shape: {logits.shape}")
        mask = self.after_conv(logits).squeeze(1)   # (B, T, F)
        # print(f"mask.shape: {mask.shape}")
        output = x1 * mask + x2 * (1 - mask)    # (B, T, F)

        return output


