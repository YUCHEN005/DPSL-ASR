from collections import OrderedDict
from typing import Tuple
import pickle

import logging
import torch
from torch_complex.tensor import ComplexTensor
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

#$from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.layers.activation import Mish

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
        xs_pack = pack_padded_sequence(xs_pad, ilens.cpu(), batch_first=True)
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




class RNNSeparator(AbsSeparator):

    """
    RNN Separator
        Args:
            input_dim: input feature dimension
            rnn_type: string, select from 'blstm', 'lstm' etc.
            bidirectional: bool, whether the inter-chunk RNN layers are bidirectional.
            num_spk: number of speakers
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid', 'mish'
            layer: int, number of stacked RNN layers. Default is 3.
            unit: int, dimension of the hidden state.
            dropout: float, dropout ratio. Default is 0.5
    """

    def __init__(
        self,
        input_dim: int,
        rnn_type: str = "blstm",
        num_spk: int = 1,
        nonlinear: str = "mish",
        layer: int = 3,
        unit: int = 896,
        dropout: float = 0.5,
        mvn_dict=None,
        #bidirectional=True,
    ):
        super().__init__()

        self._num_spk = num_spk

        #self.rnn = RNN(
        #    idim=input_dim,
        #    elayers=layer,
        #    cdim=unit,
        #    hdim=unit,
        #    dropout=dropout,
        #    typ=rnn_type,
        #)

        #self.linear = torch.nn.ModuleList(
        #    [torch.nn.Linear(unit, input_dim) for _ in range(self.num_spk)]
        #)

        if nonlinear not in ("sigmoid", "relu", "tanh", "mish"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))
        #if rnn_type not in ["RNN", "LSTM", "GRU"]:
        #    raise ValueError("Unsupported rnn type: {}".format(rnn_type)) 
        #self.rnn = getattr(torch.nn, rnn_type)(
        #    input_dim,
        #    hidden_size=unit,
        #    num_layers=layer,
        #    batch_first=True,
        #    dropout=dropout,
        #    bidirectional=bidirectional,
        #)
        self.rnn = RNN(
            idim=input_dim,
            elayers=layer,
            cdim=unit,
            hdim=unit,
            dropout=dropout,
            typ=rnn_type,
        )

        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.ModuleList(
            [
                torch.nn.Linear(unit * 2,  input_dim)
            ]
        )
        self.nonlinear = {
            "relu": torch.nn.functional.relu,
            "sigmoid": torch.nn.functional.sigmoid,
            "tanh": torch.nn.functional.tanh,
            "mish": Mish(),
        }[nonlinear]
        
        if mvn_dict:
            logging.info("Using cmvn dictionary from {}".format(mvn_dict))
            with open(mvn_dict, "rb") as f:
                self.mvn_dict = pickle.load(f)
    def forward(self, input: torch.Tensor, ilens: torch.Tensor):

        """Forward.
        Args:
            input (torch.Tensor): [B, Frames, Freq], it is abs(stft output feature), it is called noisy magnitude feature.
            ilens (torch.Tensor): input lengths [Batch]
        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, Frames, Freq), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'spk1': torch.Tensor(Batch, Frames, Freq),
                'spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """
        # magnitude specturm -> global cmvn -> rnn -> masks
        # enhanced magnitude spectrum = magnitude specturm * masks
        # predict masks for each speaker
        x = input.cpu().data.numpy()
        logging.info(f"in the separator forward function, noisy magnitude is converted to numpy, {x.dtype}")
        logging.info(f"in the separator forward function, noisy magnitude is {x}")
        if self.mvn_dict:
            x = apply_cmvn(x, self.mvn_dict)
        logging.info(f"in the separator forward function, apply mvn on noisy magnitude is {x}")
        x = torch.tensor(x,dtype=torch.float32,device=ilens.device)
        logging.info(f"in the separator forward function, noisy magnitude is converted to torch.Tensor, {x.dtype}")
        logging.info(f"in the separator forward function, input of rnn network is {x}, its shape is {x.shape} ")
        x, ilens, _ = self.rnn(x, ilens)
        logging.info(f"in the separator forward function, before self.dropout, output of rnn network is {x}, its shape is {x.shape} ")
        x = self.dropout(x)
        logging.info(f"in the separator forward function, after self.dropout, output of rnn network is {x}, its shape is {x.shape} ")
        masks = []
        for linear in self.linear:
            y = linear(x)
            y = self.nonlinear(y)
            masks.append(y)

        predicted_magnitude = [input * m for m in masks]
        logging.info(f"in the separator forward function, predicted_magnitude[0] is {predicted_magnitude[0]} and its shape is {predicted_magnitude[0].shape}")

        masks = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        logging.info(f"in the separator forward function, masks is {masks['mask_spk1']} and its shape is {masks['mask_spk1'].shape}")
        return predicted_magnitude, ilens, masks

    def inference(self, input: torch.Tensor, ilens: torch.Tensor):
        """
        Args:
            input (torch.Tensor): magnitude feature of stft (it has applyed cmvn), you can see 
                                  "espnet2/bin/enh_inference1.py" 
            ilens (torch.Tensor): input lengths [Batch]
        Retrun:
            masks: List
           
        """
        x, ilens, _ = self.rnn(input, ilens)
        x = self.dropout(x)
        masks = []
        for linear in self.linear:
            y = linear(x)
            y = self.nonlinear(y)
            masks.append(y)
 
        return masks
    


    @property
    def num_spk(self):
        return self._num_spk


def apply_cmvn(feats, cmvn_dict):
    if type(cmvn_dict) != dict:
        raise TypeError("Input must be a python dictionary")
    if "mean" in cmvn_dict:
        feats = feats - cmvn_dict["mean"]
    if "std" in cmvn_dict:
        feats = feats / cmvn_dict["std"]
    return feats


