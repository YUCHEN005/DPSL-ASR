
from torch import nn
"""
modified from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
"""

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, t, c = x.size()
        x = x.transpose(1,2) # b,c,t
        y = self.avg_pool(x).view(b, c,)
        y = self.fc(y).view(b, c, 1, )
        result = x * y.expand_as(x) # b,c,t
        return result.transpose(1,2) # b,t,c
