import torch
import logging
import torch.nn.functional as F
from torch import nn


class KLLoss(nn.Module):
    """Label-smoothing loss

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(self, size, padding_idx, normalize_length=False, criterion=nn.KLDivLoss(reduce=False)):
        super(KLLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.size = size
        self.normalize_length = normalize_length

    def forward(self, x, x1, target):
        """Compute loss between x and target

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target: target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        x1 = x1.view(-1, self.size)
        target = target.view(-1)
        with torch.no_grad():
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
        x_logSoftmax = torch.log_softmax(x, dim=1)
        x1_softmax = F.softmax(x1, dim=1)
        #logging.warning('x_logSoftmax = ' + str(x_logSoftmax))
        #logging.warning('x1_softmax = ' + str(x1_logSoftmax))
        kl = self.criterion(x_logSoftmax, x1_softmax)
        #logging.warning('kl = ' + str(kl))
        denom = total if self.normalize_length else batch_size
        kl_fill = kl.masked_fill(ignore.unsqueeze(1), 0)
        #logging.warning('kl_fill = ' + str(kl_fill))
        return kl_fill.sum() / denom
