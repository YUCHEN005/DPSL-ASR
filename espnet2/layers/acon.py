import torch
from torch import nn
# modified from : https://github.com/nmaac/acon/blob/main/acon.py
import logging
class AconC(nn.Module):
    r""" ACON activation (activate or not).
    # AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    # according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """
    def __init__(self, width):
        super().__init__()

        #self.p1 = nn.Parameter(torch.randn(1, width, 1, 1))
        #self.p2 = nn.Parameter(torch.randn(1, width, 1, 1))
        #self.beta = nn.Parameter(torch.ones(1, width, 1, 1))
        self.p1 = nn.Parameter(torch.randn(1, 1, width))
        self.p2 = nn.Parameter(torch.randn(1, 1, width))
        self.beta = nn.Parameter(torch.ones(1,1, width))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, **kwargs):
        a = self.p1 * x - self.p2 * x
        logging.info(f'self.p1 * x - self.p2 * x shape is {(self.p1 * x - self.p2 * x).shape}')
        b = self.sigmoid(self.beta * (self.p1 * x - self.p2 * x))
        logging.info(f"self.sigmoid(self.beta * (self.p1 * x - self.p2 * x)) shape is {self.sigmoid(self.beta * (self.p1 * x - self.p2 * x)).shape}")
        c = self.p2 * x
        logging.info(f"self.p2 * x shape is {(self.p2 * x).shape}")
        return a * b + c

class MetaAconC(nn.Module):
    r""" ACON activation (activate or not).
    # MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is generated by a small network
    # according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """
    def __init__(self, width, r=16):
        super().__init__()
        self.fc1 = nn.Conv1d(width, max(r,width//r), kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm1d(max(r,width//r))
        #self.bn1 = nn.SyncBatchNorm(max(r,width//r)) # 2021-5-26 modified
        self.fc2 = nn.Conv1d(max(r,width//r), width, kernel_size=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm1d(width)
        #self.bn2 = nn.SyncBatchNorm(width) # 2021-5-26 modified
        self.p1 = nn.Parameter(torch.randn(1, width, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, **kwargs):
        #beta = self.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True))))))
        x = x.transpose(1, 2) # BxTxF -> BxFxT
        #logging.info(f"x.transpose(1, 2) is {x.shape}")
        beta = self.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(x.mean(dim=2, keepdims=True))))))
        #logging.info(f"self.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(x.mean(dim=2, keepdims=True)))))) shape is {beta.shape}")
        a = self.p1 * x - self.p2 * x
        b = self.sigmoid( beta * (self.p1 * x - self.p2 * x))
        c = self.p2 * x
        return (a * b + c).transpose(1,2)
