#!/usr/bin/env python3

# MA DUO

# reference: "Mish: A Self Regularized Non-Monotonic Neural Activation Function"


import torch.nn as nn
import torch
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x
