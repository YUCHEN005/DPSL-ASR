from collections import OrderedDict
from typing import Tuple
import logging
import pickle
import torchaudio

#from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet2.enh.abs_enh import AbsEnhancement
#from espnet2.layers.stft import Stft
from espnet2.layers.stft_check1 import Stft
from espnet2.layers.utterance_mvn import UtteranceMVN
import torch
from torch_complex.tensor import ComplexTensor
from espnet2.enh.layers.activation import Mish
from espnet2.layers.swish import Swish
from espnet2.layers.acon import AconC
from espnet2.layers.acon import MetaAconC
from espnet2.layers.senet import SELayer 
#from torch_complex.tensor import ComplexTensor
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class RNN(torch.nn.Module):
    """RNN module
    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of rnn units (resulted in cdim * 2 if bidirectional)
    :param int hdim: number of final projection units
    :param float dropout: dropout rate
    :param str typ: The RNN type
    """

    def __init__(self, idim, elayers, cdim, hdim, dropout, typ="blstm",):
        super(RNN, self).__init__()
        bidir = typ[0] == "b"
        self.nbrnn = (
            torch.nn.LSTM(
                idim,
                cdim,
                elayers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidir,
            )
            if "lstm" in typ
            else torch.nn.GRU(
                idim,
                cdim,
                elayers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidir,
            )
        )
        #if bidir:
        #    self.l_last = torch.nn.Linear(cdim * 2, last_layer_output_dim)
        #else:
        #    self.l_last = torch.nn.Linear(cdim, last_layer_output_dim)
        self.typ = typ
    def forward(self, xs_pad, ilens, prev_state=None):
        """RNN forward
        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous RNN states
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        logging.debug(self.__class__.__name__ + " input lengths: " + str(ilens))
        if not isinstance(ilens, torch.Tensor):
            ilens = torch.tensor(ilens)
        #xs_pack = pack_padded_sequence(xs_pad, ilens.cpu(), batch_first=True)
        xs_pack = pack_padded_sequence(xs_pad, ilens.cpu(), batch_first=True,enforce_sorted=False)
        self.nbrnn.flatten_parameters()
        if prev_state is not None and self.nbrnn.bidirectional:
            # We assume that when previous state is passed,
            # it means that we're streaming the input
            # and therefore cannot propagate backward BRNN state
            # (otherwise it goes in the wrong direction)
            prev_state = reset_backward_rnn_state(prev_state)
        ys, states = self.nbrnn(xs_pack, hx=prev_state)
        # ys: utts x frame x cdim x 2 (2: means bidirectional)
        ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
        #ys_pad, ilens = pad_packed_sequence(ys, batch_first=True,enforce_sorted=False)
        #logging.info(f"rnn output shape is {ys_pad.shape}")
        # (sum _utt frame_utt) x dim
        #projected = torch.tanh(
        #    self.l_last(ys_pad.contiguous().view(-1, ys_pad.size(2)))
        #)
        #xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)
        return ys_pad, ilens, states  # x: utt list of frame x dim


def reset_backward_rnn_state(states):
    """Sets backward BRNN states to zeroes
    Useful in processing of sliding windows over the inputs
    """
    if isinstance(states, (list, tuple)):
        for state in states:
            state[1::2] = 0.0
    else:
        states[1::2] = 0.0
    return states


class TFMaskingNet1(AbsEnhancement):
    """TF Masking Speech Separation Net."""

    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        rnn_type: str = "blstm",
        layer: int = 3,
        unit: int = 896,
        dropout: float = 0.0,
        num_spk: int = 1,
        nonlinear: str = "sigmoid",
        #utt_mvn: bool = False,
        mask_type: str = "IRM",
        loss_type: str = "magnitude3",
        mvn_dict=None,
    ):
        super(TFMaskingNet1, self).__init__()

        self.num_spk = num_spk
        self.num_bin = n_fft // 2 + 1
        self.mask_type = mask_type
        self.loss_type = loss_type
        if loss_type not in ("mask_mse", "magnitude","magnitude3", "spectrum"):
            raise ValueError("Unsupported loss type: %s" % loss_type)

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
        #self.stft = torch.stft(a,  n_fft=512,hop_length=256,win_length=512, center=True,window=torch.hann_window(512))


        self.rnn = RNN(
            idim=self.num_bin,
            elayers=layer,
            cdim=unit,
            hdim=unit,
            dropout=dropout,
            typ=rnn_type,
        )

        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.ModuleList(
            [
                torch.nn.Linear(unit * 2,  self.num_bin)
            ]
        )
        self.nonlinear = {
            "relu": torch.nn.functional.relu,
            "sigmoid": torch.nn.functional.sigmoid,
            "tanh": torch.nn.functional.tanh,
            "mish": Mish(),
            "swish": Swish(),
            "acon": AconC(self.num_bin),
            "metaacon": MetaAconC(self.num_bin),
            "senet": SELayer(self.num_bin),
        }[nonlinear]

        if mvn_dict:
            logging.info("Using cmvn dictionary from {}".format(mvn_dict))
            with open(mvn_dict, "rb") as f:
                self.mvn_dict = pickle.load(f)



    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            separated (list[ComplexTensor]): [(B, T, F), ...]
            ilens (torch.Tensor): (B,)
            predcited masks: OrderedDict[
                'spk1': torch.Tensor(Batch, Frames, Channel, Freq),
                'spk2': torch.Tensor(Batch, Frames, Channel, Freq),
                ...
                'spkn': torch.Tensor(Batch, Frames, Channel, Freq),
            ]
        """

        # wave -> stft -> magnitude specturm -> global cmvn -> rnn -> masks
        input_spectrum, flens = self.stft(input, ilens)

        input_magnitude, input_phase = torchaudio.functional.magphase(input_spectrum)
        # apply apply global mvn
        input_magnitude_numpy = input_magnitude.cpu().data.numpy()
        if self.mvn_dict:
            input_magnitude_mvn_numpy = apply_cmvn(input_magnitude_numpy, self.mvn_dict)
        input_magnitude_mvn = torch.tensor(input_magnitude_mvn_numpy,dtype=torch.float32,device=ilens.device)
        x, flens, _ = self.rnn(input_magnitude_mvn, flens)
        x = self.dropout(x)
        #logging.info(f"in the tf_mask_net1 forward function,output of self.drop is {x} its shape is {x.shape}")
        masks = []
        for linear in self.linear:
            y = linear(x)
            y = self.nonlinear(y)
            masks.append(y)

        # apply mask
        predicted_magnitude = [input_magnitude * m for m in masks]
        masks = OrderedDict(
            zip(["spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        #logging.info(f"in the tf_mask_net1 forward,masks['spk1'] is {masks['spk1']}, its shape is {masks['spk1'].shape}")
        return predicted_magnitude, flens, masks

    def forward_rawwav(
        self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Output with waveforms.
        I don't use it on the inference stage, so I can  remove it TODO:(md) (2021-2-23) 
        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            predcited speech [Batch, num_speaker, sample]
            output lengths
            predcited masks: OrderedDict[
                'spk1': torch.Tensor(Batch, Frames, Channel, Freq),
                'spk2': torch.Tensor(Batch, Frames, Channel, Freq),
                ...
                'spkn': torch.Tensor(Batch, Frames, Channel, Freq),
            ]
        """

        # predict spectrum for each speaker
        predicted_magnitude, flens, masks = self.forward(input, ilens)
        # wave -> stft -> phase
        input_spectrum, flens = self.stft(input, ilens)
        input_spectrum = ComplexTensor(input_spectrum[..., 0], input_spectrum[..., 1])
        input_magnitude = abs(input_spectrum)
        input_phase = input_spectrum / (input_magnitude + 10e-12)
        
        
        if predicted_spectrums is None:
            predicted_wavs = None
        elif isinstance(predicted_spectrums, list):
            # multi-speaker input
            predicted_wavs = [
                self.stft.inverse(ps, ilens)[0] for ps in predicted_spectrums
            ]
        else:
            # single-speaker input
            predicted_wavs = self.stft.inverse(predicted_spectrums, ilens)[0]

        return predicted_wavs, ilens, masks

def apply_cmvn(feats, cmvn_dict):
    if type(cmvn_dict) != dict:
        raise TypeError("Input must be a python dictionary")
    if "mean" in cmvn_dict:
        feats = feats - cmvn_dict["mean"]
    if "std" in cmvn_dict:
        feats = feats / cmvn_dict["std"]
    return feats

