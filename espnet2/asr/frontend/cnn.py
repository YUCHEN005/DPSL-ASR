import copy
from typing import Optional
from typing import Tuple
from typing import Union

import humanfriendly
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.utils.get_default_kwargs import get_default_kwargs


class CNNFrontend(AbsFrontend):
    """ conventional fronted structure for ASR
    waveform -> cnn block -> cnn feature
    """
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        L: int=20,
        N: int=256,
        B: int=256,
        dim: int=80,
    ):
        assert check_argument_types()
        super().__init__()
        # Multi-scale Encoder
        # B x S => B x N x T, S = 4s*8000 = 32000
        self.B = B
        self.L1 = L
        self.L2 = 80
        self.L3 = 160
        self.encoder_1d_short = Conv1D(1, N, L, stride=L // 2, padding=0)
        self.encoder_1d_middle = Conv1D(1, N, 80, stride=L // 2, padding=0)
        self.encoder_1d_long = Conv1D(1, N, 160, stride=L // 2, padding=0)
        # keep T not change
        # T = int((xlen - L) / (L // 2)) + 1
        # before repeat blocks, always cLN
        self.ln = ChannelWiseLayerNorm(3*N)
         # # B x N x T => B x D x T
        self.proj = Conv1D(3*N, B, 1)
        self.proj1 = torch.nn.Linear(B, 128)
        self.proj2 = torch.nn.Linear(128, 80)
        self.dim = dim
    def forward(self,x: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        # when inference, only one utt
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        
        # Multi-scale Encoder (Mixture audio input)
        w1 = F.relu(self.encoder_1d_short(x))
        T = w1.shape[-1]
        xlen1 = x.shape[-1]
        xlen2 = (T - 1) * (self.L1 // 2) + self.L2
        xlen3 = (T - 1) * (self.L1 // 2) + self.L3
        w2 = F.relu(self.encoder_1d_middle(F.pad(x, (0, xlen2 - xlen1), "constant", 0)))
        w3 = F.relu(self.encoder_1d_long(F.pad(x, (0, xlen3 - xlen1), "constant", 0)))
        # B x 3N x T
        y = self.ln(torch.cat([w1, w2, w3], 1))
        # B x D x T  # D is feature dimension, B is batch size
        y = self.proj(y)
        y = y.permute(0,2,1) # BxTxD
        y = self.proj1(y)
        y = self.proj2(y)
         

        if torch.is_tensor(input_lengths):
            output_lengths = input_lengths.cpu().numpy()
        else:
            output_lengths = np.array(input_lengths, dtype=np.float32)
        output_lengths = np.array(output_lengths // 10 - 1, dtype=np.int64).tolist()
        output_lengths = torch.tensor(output_lengths)
        
        return y, output_lengths

    def output_size(self) -> int:
        return self.dim
 


class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x

class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x



##### test case
###if __name__ == "__main__":
###    frontend = CNNFrontend()
###    wavname = "/home/md510/w2020/espnet-recipe/asr_with_enhancement/test/fe_03_1007-02235-A-000245-000428-src.wav" 
###    import soundfile
###    import numpy as np
###    x, rate = soundfile.read(wavname)
###    if isinstance(x, np.ndarray):
###        x = torch.tensor(x)
###    x = x.unsqueeze(0).to(getattr(torch, "float32"))
###    x_length = x.new_full([1],dtype=torch.long, fill_value=x.size(1))
###    output, output_length = frontend(x, x_length)
###    print(f"output shape is {output.shape}")
###    print(f"output_length shape is {output_length}")
