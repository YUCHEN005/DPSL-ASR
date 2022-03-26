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
class MSCAM(nn.Module):
    def __init__(
        self,
        num_channels: int = 64,
        r: int = 2,
    ):  
        check_argument_types()
        super().__init__()
        self.num_channels = num_channels
        self.left = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, num_channels // r, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels // r),    
            nn.PReLU(),
            nn.Conv2d(num_channels // r, num_channels, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels),
        )
        self.right = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // r, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels // r),    
            nn.PReLU(),
            nn.Conv2d(num_channels // r, num_channels, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x1,   # (B, C, T, F)
        x2,   # (B, C, T, F)
    ):
        x = x1 + x2
        x_left = self.left(x)                       # (B, C, 1, 1)
        x_right = self.right(x)                     # (B, C, T, F)
        mask = self.sigmoid(x_left + x_right)       # (B, C, T, F)
        output = x1 * mask + x2 * (1 - mask)        # (B, C, T, F)

        return output

class AttnFuse(nn.Module):
    def __init__(
        self,
        num_channels: int = 64,
    ):  
        check_argument_types()
        super().__init__()
        self.num_channels = num_channels
        self.conv_mask = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels, (1, 1), (1, 1)),  
            nn.BatchNorm2d(num_channels),    
            nn.Sigmoid(),
        )
        self.mscam = MSCAM(num_channels)

    def forward(
        self,
        x1,   # (B, C, T, F)
        x2,   # (B, C, T, F)
    ):
        x = torch.cat((x1, x2), dim=1)              # (B, 2C, T, F)
        mask = self.conv_mask(x)                    # (B, C, T, F)
        output = self.mscam(x1 * mask, x2)          # (B, C, T, F)
        output = torch.cat((output, x2), dim=1)     # (B, 2C, T, F)

        return output


class AttnEncoder(nn.Module):
    def __init__(
        self,
        num_channels: int = 64,
    ):  
        check_argument_types()
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, num_channels // 2, (1, 1), (1, 1), (0, 0)),  
            nn.BatchNorm2d(num_channels // 2),    
            nn.PReLU(),   
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(num_channels // 2, num_channels // 2, (3, 3), (1, 2), (1, 0)),  
            nn.BatchNorm2d(num_channels // 2),    
            nn.PReLU(),   
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(num_channels // 2, num_channels, (3, 3), (1, 2), (1, 0)),  
            nn.BatchNorm2d(num_channels),    
            nn.PReLU(),   
        )

    def forward(
        self,
        x: torch.Tensor,   # (B, 1, T, F)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x1 = self.conv_1(x)     # (B, C/2, T, F)
        x2 = self.conv_2(x1)    # (B, C/2, T, F/2)
        x3 = self.conv_3(x2)    # (B, C, T, F/4)

        return x3, [x3, x2, x1]

class AttnDecoder(nn.Module):
    def __init__(
        self,
        num_channels: int = 64,
    ):  
        check_argument_types()
        super().__init__()
        self.attn_fuse3 = AttnFuse(num_channels)
        self.conv_3 = nn.Sequential(
            nn.ConvTranspose2d(num_channels * 2, num_channels // 2, (3, 3), (1, 2), (1, 0)),  
            nn.BatchNorm2d(num_channels // 2),    
            nn.PReLU(),   
        )
        self.attn_fuse2 = AttnFuse(num_channels // 2)
        self.conv_2 = nn.Sequential(
            nn.ConvTranspose2d(num_channels, num_channels // 2, (3, 3), (1, 2), (1, 0), output_padding=(0, 1)),  
            nn.BatchNorm2d(num_channels // 2),    
            nn.PReLU(),   
        )
        self.attn_fuse1 = AttnFuse(num_channels // 2)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(num_channels, 1, (1, 1), (1, 1), (0, 0)),  
            nn.BatchNorm2d(1),    
            nn.PReLU(),   
        )

    def forward(
        self,
        x: torch.Tensor,   # (B, C, T, F/4)
        x_list,            # [(B, C, T, F/4), (B, C/2, T, F/2), (B, C/2, T, F)] 
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x3 = self.attn_fuse3(x_list[0], x)          # (B, 2C, T, F/4)   
        x3 = self.conv_3(x3)                        # (B, C/2, T, F/2)

        x2 = self.attn_fuse2(x_list[1], x3)         # (B, C, T, F/2)
        x2 = self.conv_2(x2)                        # (B, C/2, T, F)

        x1 = self.attn_fuse1(x_list[2], x2)         # (B, C, T, F)
        x1 = self.conv_1(x1)                        # (B, 1, T, F)

        return x1





class Encoder(nn.Module):
    def __init__(
        self,
        num_channels: int = 64,
    ):  
        check_argument_types()
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, num_channels // 2, (1, 1), (1, 1), (0, 0)),  
            nn.BatchNorm2d(num_channels // 2),    
            nn.PReLU(),   
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(num_channels // 2, num_channels // 2, (3, 3), (1, 2), (1, 0)),  
            nn.BatchNorm2d(num_channels // 2),    
            nn.PReLU(),   
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(num_channels // 2, num_channels, (3, 3), (1, 2), (1, 0)),  
            nn.BatchNorm2d(num_channels),    
            nn.PReLU(),   
        )

    def forward(
        self,
        x: torch.Tensor,   # (B, 1, T, F)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x1 = self.conv_1(x)     # (B, C/2, T, F)
        x2 = self.conv_2(x1)    # (B, C/2, T, F/2)
        x3 = self.conv_3(x2)    # (B, C, T, F/4)

        return x3, [x3, x2, x1]

class Decoder(nn.Module):
    def __init__(
        self,
        num_channels: int = 64,
    ):  
        check_argument_types()
        super().__init__()
        self.conv_3 = nn.Sequential(
            nn.ConvTranspose2d(num_channels * 2, num_channels // 2, (3, 3), (1, 2), (1, 0)),  
            nn.BatchNorm2d(num_channels // 2),    
            nn.PReLU(),   
        )
        self.conv_2 = nn.Sequential(
            nn.ConvTranspose2d(num_channels, num_channels // 2, (3, 3), (1, 2), (1, 0), output_padding=(0, 1)),  
            nn.BatchNorm2d(num_channels // 2),    
            nn.PReLU(),   
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(num_channels, 1, (1, 1), (1, 1), (0, 0)),  
            nn.BatchNorm2d(1),    
            nn.PReLU(),   
        )

    def forward(
        self,
        x: torch.Tensor,   # (B, C, T, F/4)
        x_list,            # [(B, C, T, F/4), (B, C/2, T, F/2), (B, C/2, T, F)] 
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x3 = torch.cat([x, x_list[0]], dim=1)       # (B, 2C, T, F/4)
        x3 = self.conv_3(x3)                        # (B, C/2, T, F/2)

        x2 = torch.cat([x3, x_list[1]], dim=1)      # (B, C, T, F/2)
        x2 = self.conv_2(x2)                        # (B, C/2, T, F)

        x1 = torch.cat([x2, x_list[2]], dim=1)      # (B, C, T, F)
        x1 = self.conv_1(x1)                        # (B, 1, T, F)

        return x1





# class Encoder(nn.Module):
#     def __init__(
#         self,
#         num_channels: int = 64,
#     ):  
#         check_argument_types()
#         super().__init__()
#         self.conv_1 = nn.Sequential(
#             nn.Conv2d(1, num_channels // 4, (1, 1), (1, 1), (0, 0)),  
#             nn.BatchNorm2d(num_channels // 4),    
#             nn.PReLU(),
#         )
#         self.conv_2 = nn.Sequential(
#             nn.Conv2d(num_channels // 4, num_channels // 4, (1, 1), (1, 1), (0, 0)),  
#             nn.BatchNorm2d(num_channels // 4),    
#             nn.PReLU(),  
#         )
#         self.conv_3 = nn.Sequential(
#             nn.Conv2d(num_channels // 4, num_channels // 2, (3, 3), (1, 2), (1, 0)),  
#             nn.BatchNorm2d(num_channels // 2),    
#             nn.PReLU(), 
#         )
#         self.conv_4 = nn.Sequential(
#             nn.Conv2d(num_channels // 2, num_channels // 2, (1, 1), (1, 1), (0, 0)),  
#             nn.BatchNorm2d(num_channels // 2),    
#             nn.PReLU(),
#         )
#         self.conv_5 = nn.Sequential(
#             nn.Conv2d(num_channels // 2, num_channels, (3, 3), (1, 2), (1, 0)),  
#             nn.BatchNorm2d(num_channels),    
#             nn.PReLU(),  
#         )
#         self.conv_6 = nn.Sequential(
#             nn.Conv2d(num_channels, num_channels, (1, 1), (1, 1), (0, 0)),  
#             nn.BatchNorm2d(num_channels),    
#             nn.PReLU(), 
#         )

#     def forward(
#         self,
#         x: torch.Tensor,   # (B, 1, T, F)
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         x1 = self.conv_1(x)     # (B, C/4, T, F)
#         x2 = self.conv_2(x1)    # (B, C/4, T, F)
#         x3 = self.conv_3(x2)    # (B, C/2, T, F/2)
#         x4 = self.conv_4(x3)    # (B, C/2, T, F/2)
#         x5 = self.conv_5(x4)    # (B, C, T, F/4)
#         x6 = self.conv_6(x5)    # (B, C, T, F/4)

#         return x6, [x6, x5, x4, x3, x2, x1]

# class Decoder(nn.Module):
#     def __init__(
#         self,
#         num_channels: int = 64,
#     ):  
#         check_argument_types()
#         super().__init__()
#         self.conv_6 = nn.Sequential(
#             nn.Conv2d(num_channels * 2, num_channels, (1, 1), (1, 1), (0, 0)),  
#             nn.BatchNorm2d(num_channels),    
#             nn.PReLU(),
#         )
#         self.conv_5 = nn.Sequential(
#             nn.ConvTranspose2d(num_channels * 2, num_channels // 2, (3, 3), (1, 2), (1, 0)),  
#             nn.BatchNorm2d(num_channels // 2),    
#             nn.PReLU(),   
#         )
#         self.conv_4 = nn.Sequential(  
#             nn.Conv2d(num_channels, num_channels // 2, (1, 1), (1, 1), (0, 0)),  
#             nn.BatchNorm2d(num_channels // 2),    
#             nn.PReLU(),  
#         )       
#         self.conv_3 = nn.Sequential(
#             nn.ConvTranspose2d(num_channels, num_channels // 4, (3, 3), (1, 2), (1, 0), output_padding=(0, 1)),  
#             nn.BatchNorm2d(num_channels // 4),    
#             nn.PReLU(), 
#         )
#         self.conv_2 = nn.Sequential(
#             nn.Conv2d(num_channels // 2, num_channels // 4, (1, 1), (1, 1), (0, 0)),  
#             nn.BatchNorm2d(num_channels // 4),    
#             nn.PReLU(),  
#         )
#         self.conv_1 = nn.Sequential(
#             nn.Conv2d(num_channels // 2, 1, (1, 1), (1, 1), (0, 0)),  
#             nn.BatchNorm2d(1),    
#             nn.PReLU(),   
#         )
        
#     def forward(
#         self,
#         x: torch.Tensor,   # (B, C, T, F/4)
#         x_list,            # [(B, C, T, F/4), (B, C, T, F/4), (B, C/2, T, F/2), (B, C/2, T, F/2), (B, C/4, T, F), (B, C/4, T, F)] 
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         x6 = torch.cat([x, x_list[0]], dim=1)       # (B, 2C, T, F/4)
#         x6 = self.conv_6(x6)                        # (B, C, T, F/4)

#         x5 = torch.cat([x6, x_list[1]], dim=1)      # (B, 2C, T, F/4)
#         x5 = self.conv_5(x5)                        # (B, C/2, T, F/2)

#         x4 = torch.cat([x5, x_list[2]], dim=1)      # (B, C, T, F/2)
#         x4 = self.conv_4(x4)                        # (B, C/2, T, F/2)

#         x3 = torch.cat([x4, x_list[3]], dim=1)      # (B, C, T, F/2)
#         x3 = self.conv_3(x3)                        # (B, C/4, T, F)

#         x2 = torch.cat([x3, x_list[4]], dim=1)      # (B, C/2, T, F)
#         x2 = self.conv_2(x2)                        # (B, C/4, T, F)

#         x1 = torch.cat([x2, x_list[5]], dim=1)      # (B, C/2, T, F)
#         x1 = self.conv_1(x1)                        # (B, 1, T, F)
        
#         return x1


