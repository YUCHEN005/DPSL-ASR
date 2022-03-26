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
class ResidualBlock(nn.Module):
    def __init__(
        self,
        num_channels: int = 64,
    ):  
        check_argument_types()
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, (3, 3), (1, 1), (1, 1)),  # (5, 7)
            nn.BatchNorm2d(num_channels),    
            nn.PReLU(),   
            nn.Conv2d(num_channels, num_channels, (3, 3), (1, 1), (1, 1)),  # (5, 7)
            nn.BatchNorm2d(num_channels),    
            nn.PReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,   # (B, C, T, F)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        output = self.conv(x) + x   # (B, C, T, F)

        return output

class TemporalSelfAttention(nn.Module):
    def __init__(
        self,
        num_channels: int = 64,
    ):  
        check_argument_types()
        super().__init__()
        self.conv_q = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // 2, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels // 2),    
            nn.PReLU(),  
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // 2, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels // 2),    
            nn.PReLU(),  
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // 2, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels // 2),    
            nn.PReLU(),  
        )
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels // 2, num_channels, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels),    
            nn.PReLU(),  
        )

    def forward(
        self,
        x: torch.Tensor,   # (B, C, T, F)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, C, T, F = x.shape
        q = self.conv_q(x).permute(0, 2, 1, 3).contiguous().view(B, T, C*F//2)                                  # (B, T, C/2 * F)
        k = self.conv_k(x).permute(0, 2, 1, 3).contiguous().view(B, T, C*F//2).transpose(1, 2).contiguous()     # (B, C/2 * F, T)
        v = self.conv_v(x).permute(0, 2, 1, 3).contiguous().view(B, T, C*F//2)                                  # (B, T, C/2 * F)
        qk = torch.softmax(torch.matmul(q, k) / math.sqrt(C*F//2), dim=-1)                          # (B, T, T)
        logits = torch.matmul(qk, v).view(B, T, C//2, F).permute(0, 2, 1, 3).contiguous()           # (B, C/2, T, F)
        output = self.conv(logits) + x      # (B, C, T, F)

        return output

class FrequencySelfAttention(nn.Module):
    def __init__(
        self,
        num_channels: int = 64,
    ):  
        check_argument_types()
        super().__init__()
        self.conv_q = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // 2, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels // 2),    
            nn.PReLU(),  
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // 2, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels // 2),    
            nn.PReLU(),  
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // 2, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels // 2),    
            nn.PReLU(),  
        )
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels // 2, num_channels, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels),    
            nn.PReLU(),  
        )

    def forward(
        self,
        x: torch.Tensor,   # (B, C, T, F)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, C, T, F = x.shape
        q = self.conv_q(x).permute(0, 3, 1, 2).contiguous().view(B, F, C*T//2)                                  # (B, F, C/2 * T)
        k = self.conv_k(x).permute(0, 3, 1, 2).contiguous().view(B, F, C*T//2).transpose(1, 2).contiguous()     # (B, C/2 * T, F)
        v = self.conv_v(x).permute(0, 3, 1, 2).contiguous().view(B, F, C*T//2)                                  # (B, F, C/2 * T)
        qk = torch.softmax(torch.matmul(q, k) / math.sqrt(C*T//2), dim=-1)                      # (B, F, F)
        logits = torch.matmul(qk, v).view(B, F, C//2, T).permute(0, 2, 3, 1).contiguous()       # (B, C/2, T, F)
        output = self.conv(logits) + x      # (B, C, T, F)

        return output


class RABlock(nn.Module):
    def __init__(
        self,
        num_channels: int = 64,
    ):  
        check_argument_types()
        super().__init__()
        self.residual_blocks = nn.Sequential(
            ResidualBlock(num_channels),
            ResidualBlock(num_channels),
        )
        self.temp_self_att = TemporalSelfAttention(num_channels)
        self.freq_self_att = FrequencySelfAttention(num_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels * 3, num_channels, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels),    
            nn.PReLU(), 
        )

    def forward(
        self,
        x: torch.Tensor,   # (B, C, T, F)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        f_res = self.residual_blocks(x)     # (B, C, T, F)
        f_temp = self.temp_self_att(f_res)  # (B, C, T, F)
        f_freq = self.freq_self_att(f_res)  # (B, C, T, F)
        f_comb = torch.cat((f_res, f_temp, f_freq), dim=1)  # (B, 3C, T, F)
        f_ra = self.conv(f_comb)    # (B, C, T, F)
        
        return f_ra


