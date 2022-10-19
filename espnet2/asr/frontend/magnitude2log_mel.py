import copy
from typing import Optional
from typing import Tuple
from typing import Union

import humanfriendly
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.utils.get_default_kwargs import get_default_kwargs

class Magnitude2LogmelFrontend(AbsFrontend):
    """Conventional frontend structure for ASR.

    Magnitude-spec -> Power-spec -> Mel-Fbank -> log -> logmel feature
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: int = None,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: int = 8000,
        htk: bool = False,
        log_base: float = None,
    ):
        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs) 
        
        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
            log_base=log_base,
        )
        self.n_mels = n_mels

    def output_size(self) -> int:
        return self.n_mels

    def forward(
        self, magnitude: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. magnitude -> Power spectrum
        
        # assume wavefrom -> stft -> complex_stft_feature, complex_stft_feature= a + bj
        # magnitude = √(a**2+b**2) 
        # Power spectrum = a**2+b**2
        input_power = magnitude ** 2

        # 2. Feature transform e.g. magnitude -> log-mel feature
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, input_lengths)

        return input_feats, input_lengths
 
