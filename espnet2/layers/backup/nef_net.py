#!/usr/bin/env python3


from  typing import Tuple
from typing import Optional

from typeguard import check_argument_types
import logging
import torch
import torch.nn as nn
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet2.layers.ra_block import RABlock
from espnet2.layers.interaction_module import InteractionModule
from espnet2.layers.merge_branch import MergeBranch

'''
by Yuchen on Aug 2, 2021
reference: 'Interactive Speech and Noise Modeling for Speech Enhancement'
https://arxiv.org/pdf/2012.09408.pdf
'''
class NEFBlock(nn.Module):
    def __init__(
        self,
        num_channels: int = 64,
    ):  
        check_argument_types()
        super().__init__()
        self.ra_block1 = RABlock(num_channels)
        self.ra_block2 = RABlock(num_channels)
        self.interaction_module1 = InteractionModule(num_channels)
        self.interaction_module2 = InteractionModule(num_channels)

    def forward(
        self,
        x1: torch.Tensor,   # (B, C, T, F)
        x2: torch.Tensor,   # (B, C, T, F)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x1_ra = self.ra_block1(x1)
        x2_ra = self.ra_block2(x2)
        output1 = self.interaction_module1(x1_ra, x2_ra)
        output2 = self.interaction_module2(x2_ra, x1_ra)

        return output1, output2
            

class NEFNet(nn.Module):
    def __init__(
        self,
        num_channels: int = 64,
        num_nef_blocks: int = 4,
    ):  
        check_argument_types()
        super().__init__()
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(1, num_channels, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels),    
            nn.PReLU(),  
        )
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(1, num_channels, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels),    
            nn.PReLU(),  
        )
        self.nef_blocks = repeat(
            num_nef_blocks,
            lambda lnum: NEFBlock(num_channels),
        )
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 1, (1, 1), (1, 1)),  
            nn.BatchNorm2d(1),    
            nn.PReLU(),  
        )
        self.down_conv2 = nn.Sequential(
            nn.Conv2d(num_channels, 1, (1, 1), (1, 1)),  
            nn.BatchNorm2d(1),    
            nn.PReLU(),  
        )
        self.merge_branch = MergeBranch(4)

    def forward(
        self,
        x1: torch.Tensor,   # (B, T, F)
        x2: torch.Tensor,   # (B, T, F)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input1 = self.up_conv1(x1.unsqueeze(1))     # (B, C, T, F)
        input2 = self.up_conv2(x2.unsqueeze(1))     # (B, C, T, F)
        # print(f"input1.shape: {input1.shape}")
        # print(f"input2.shape: {input2.shape}")
        logits1, logits2 = self.nef_blocks(input1, input2)   # (B, C, T, F)
        # print(f"logits1.shape: {logits1.shape}")
        # print(f"logits2.shape: {logits2.shape}")
        output1 = self.down_conv1(logits1).squeeze(1)       # (B, T, F)
        output2 = self.down_conv2(logits2).squeeze(1)       # (B, T, F)
        # print(f"output1.shape: {output1.shape}")
        # print(f"output2.shape: {output2.shape}")
        output = self.merge_branch(output1, output2, x1, x2)    # (B, T, F)
        # print(f"output.shape: {output.shape}")
        
        return output
