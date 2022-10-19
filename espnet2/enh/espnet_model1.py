from functools import reduce
from itertools import permutations
from typing import Dict
from typing import Optional
from typing import Tuple
import logging
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

"""
se model:
input : wavform
output: magnitude 
"""
class ESPnetEnhancementModel(AbsESPnetModel):
    """Speech enhancement or separation Frontend model"""

    def __init__(
        self,
        enh_model: Optional[AbsEnhancement],
    ):
        assert check_argument_types()

        super().__init__()

        self.enh_model = enh_model
        self.num_spk = enh_model.num_spk
        self.num_noise_type = getattr(self.enh_model, "num_noise_type", 1)
        # get mask type for TF-domain models
        self.mask_type = getattr(self.enh_model, "mask_type", None)
        # get loss type for model training
        self.loss_type = getattr(self.enh_model, "loss_type", None)
        assert self.loss_type in (
            # mse_loss(predicted_mask, target_label)
            "mask_mse",
            # mse_loss(enhanced_magnitude_spectrum, target_magnitude_spectrum)
            "magnitude",
            "magnitude3",
            # mse_loss(enhanced_complex_spectrum, target_complex_spectrum)
            "spectrum",
            # si_snr(enhanced_waveform, target_waveform)
            "si_snr",
        ), self.loss_type
        # for multi-channel signal
        self.ref_channel = getattr(self.enh_model, "ref_channel", -1)

    def _create_mask_label(self, mix_spec, ref_spec, mask_type="IAM"):
        """Create mask label.

        :param mix_spec: ComplexTensor(B, T, F)
        :param ref_spec: [ComplexTensor(B, T, F), ...] or ComplexTensor(B, T, F)
        :param noise_spec: ComplexTensor(B, T, F)
        :return: [Tensor(B, T, F), ...] or [ComplexTensor(B, T, F), ...]
        """

        assert mask_type in [
            "IBM",
            "IRM",
            "IAM",
            "PSM",
            "NPSM",
            "PSM^2",
        ], f"mask type {mask_type} not supported"
        eps = 10e-8
        mask_label = []
        for r in ref_spec:
            mask = None
            if mask_type == "IBM":
                flags = [abs(r) >= abs(n) for n in ref_spec]
                mask = reduce(lambda x, y: x * y, flags)
                mask = mask.int()
            elif mask_type == "IRM":
                # TODO(Wangyou): need to fix this,
                #  as noise referecens are provided separately
                mask = abs(r) / (sum(([abs(n) for n in ref_spec])) + eps)
            elif mask_type == "IAM":
                mask = abs(r) / (abs(mix_spec) + eps)
                mask = mask.clamp(min=0, max=1)
            elif mask_type == "PSM" or mask_type == "NPSM":
                phase_r = r / (abs(r) + eps)
                phase_mix = mix_spec / (abs(mix_spec) + eps)
                # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
                cos_theta = (
                    phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
                )
                mask = (abs(r) / (abs(mix_spec) + eps)) * cos_theta
                mask = (
                    mask.clamp(min=0, max=1)
                    if mask_label == "NPSM"
                    else mask.clamp(min=-1, max=1)
                )
            elif mask_type == "PSM^2":
                # This is for training beamforming masks
                phase_r = r / (abs(r) + eps)
                phase_mix = mix_spec / (abs(mix_spec) + eps)
                # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
                cos_theta = (
                    phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
                )
                mask = (abs(r).pow(2) / (abs(mix_spec).pow(2) + eps)) * cos_theta
                mask = mask.clamp(min=-1, max=1)
            assert mask is not None, f"mask type {mask_type} not supported"
            mask_label.append(mask)
        return mask_label

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            speech_mix_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
        """
        # clean speech signal of each speaker
        speech_ref = [
            kwargs["speech_ref{}".format(spk + 1)] for spk in range(self.num_spk)
        ]
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        speech_ref = torch.stack(speech_ref, dim=1)
        


        batch_size = speech_mix.shape[0]
        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else torch.ones(batch_size).int() * speech_mix.shape[1]
        )
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        assert speech_mix.shape[0] == speech_ref.shape[0] == speech_lengths.shape[0], (
            speech_mix.shape,
            speech_ref.shape,
            speech_lengths.shape,
        )
        batch_size = speech_mix.shape[0]

        # for data-parallel
        speech_ref = speech_ref[:, :, : speech_lengths.max()]
        speech_mix = speech_mix[:, : speech_lengths.max()]

        
        # predict magnidude (it is list) and masks
        logging.info(f"in the espnet_model1, forward function, speech_lenghts is {speech_lengths} its shape is {speech_lengths.shape}") 
        predicted_magnitude, tf_length, mask_pre = self.enh_model(
            speech_mix, speech_lengths
        )
        logging.info(f"predicted_magnitude is {predicted_magnitude} its shape is {predicted_magnitude[0].shape}")
        # prepared ref magnitude, wave -> stft -> abs -> magnitude
        speech_ref = speech_ref.squeeze(1) # (B,1,samples) -> (B, samples)
        logging.info(f"in the espnet_model1, speech_ref is {speech_ref} its shape is {speech_ref.shape}")
        input_spectrum, flens = self.enh_model.stft(speech_ref, speech_lengths) # it need to check speech_lengths
        input_spectrum = ComplexTensor(input_spectrum[..., 0], input_spectrum[..., 1])
        magnitude_ref = abs(input_spectrum)
        logging.info(f"in the espnet_model1, magnitude_ref is {magnitude_ref} its shape is{magnitude_ref.shape}")
        if self.loss_type == "magnitude3":
            # compute loss on magnitude spectrum
            # magnitude_ref  is B x T x F
            # magnitude_pre[0] is B x T x F
                
            logging.info(f"in espnet_model1,using self.loss_type  is {self.loss_type }, magnitude_ref  shape is {magnitude_ref[0].shape}")
            logging.info(f"in  espnet_model1,, predicted_magnitude[0]  shape is {predicted_magnitude[0].shape}")
            tf_loss, perm = self._permutation_loss3(
                magnitude_ref, predicted_magnitude[0], tf_length,
            )

        loss = tf_loss
        stats = dict(
           loss=loss.detach(),
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        logging.info(f"final loss is {loss}, stats is {stats}, weight is {weight}")
        return loss, stats, weight

    @staticmethod
    def tf_mse_loss(ref, inf):
        """time-frequency MSE loss.

        :param ref: (Batch, T, F)
        :param inf: (Batch, T, F)
        :return: (Batch)
        """
        assert ref.dim() == inf.dim(), (ref.shape, inf.shape)
        if ref.dim() == 3:
            mseloss = (abs(ref - inf) ** 2).mean(dim=[1, 2])
        elif ref.dim() == 4:
            mseloss = (abs(ref - inf) ** 2).mean(dim=[1, 2, 3])
        else:
            raise ValueError("Invalid input shape: ref={}, inf={}".format(ref, inf))

        return mseloss

    @staticmethod
    def tf_l1_loss(ref, inf):
        """time-frequency L1 loss.

        :param ref: (Batch, T, F) or (Batch, T, C, F)
        :param inf: (Batch, T, F) or (Batch, T, C, F)
        :return: (Batch)
        """
        assert ref.dim() == inf.dim(), (ref.shape, inf.shape)
        if ref.dim() == 3:
            l1loss = abs(ref - inf).mean(dim=[1, 2])
        elif ref.dim() == 4:
            l1loss = abs(ref - inf).mean(dim=[1, 2, 3])
        else:
            raise ValueError("Invalid input shape: ref={}, inf={}".format(ref, inf))
        return l1loss

    @staticmethod
    def si_snr_loss(ref, inf):
        """si-snr loss

        :param ref: (Batch, samples)
        :param inf: (Batch, samples)
        :return: (Batch)
        """
        ref = ref / torch.norm(ref, p=2, dim=1, keepdim=True)
        inf = inf / torch.norm(inf, p=2, dim=1, keepdim=True)

        s_target = (ref * inf).sum(dim=1, keepdims=True) * ref
        e_noise = inf - s_target

        si_snr = 20 * torch.log10(
            torch.norm(s_target, p=2, dim=1) / torch.norm(e_noise, p=2, dim=1)
        )
        return -si_snr

    @staticmethod
    def si_snr_loss_zeromean(ref, inf):
        """si_snr loss with zero-mean in pre-processing.

        :param ref: (Batch, samples)
        :param inf: (Batch, samples)
        :return: (Batch)
        """
        eps = 1e-8

        assert ref.size() == inf.size()
        B, T = ref.size()
        # mask padding position along T

        # Step 1. Zero-mean norm
        mean_target = torch.sum(ref, dim=1, keepdim=True) / T
        mean_estimate = torch.sum(inf, dim=1, keepdim=True) / T
        zero_mean_target = ref - mean_target
        zero_mean_estimate = inf - mean_estimate

        # Step 2. SI-SNR with order
        # reshape to use broadcast
        s_target = zero_mean_target  # [B, T]
        s_estimate = zero_mean_estimate  # [B, T]
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=1, keepdim=True)  # [B, 1]
        s_target_energy = torch.sum(s_target ** 2, dim=1, keepdim=True) + eps  # [B, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, T]

        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=1) / (
            torch.sum(e_noise ** 2, dim=1) + eps
        )
        # print('pair_si_snr',pair_wise_si_snr[0,:])
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + eps)  # [B]
        # print(pair_wise_si_snr)

        return -1 * pair_wise_si_snr

    @staticmethod
    def _permutation_loss(ref, inf, criterion, perm=None):
        """The basic permutation loss function.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...]
            inf (List[torch.Tensor]): [(batch, ...), ...]
            criterion (function): Loss function
            perm: (batch)
        Returns:
            torch.Tensor: (batch)
        """
        num_spk = len(ref)

        def pair_loss(permutation):
            return sum(
                [criterion(ref[s], inf[t]) for s, t in enumerate(permutation)]
            ) / len(permutation)

        losses = torch.stack(
            [pair_loss(p) for p in permutations(range(num_spk))], dim=1
        )
        if perm is None:
            loss, perm = torch.min(losses, dim=1)
        else:
            loss = losses[torch.arange(losses.shape[0]), perm]

        return loss.mean(), perm

    @staticmethod
    def _permutation_loss3(ref, inf, magnitude_lengths, perm=None):
        logging.info(f"in _permutation_loss3, ref shape {ref.shape} and inf shape is {inf.shape}")
        logging.info(f"in _permutation_loss3, magnitude_lengths is {magnitude_lengths}")
        input_size = magnitude_lengths
        def loss():
            loss_for_permute = []
            #logging.info(f"masks_[0]  type is {type(masks_[0])}")
            #logging.info(f"ref[0] type is {type(ref[0])}")
            # N X T X F

            inf_magnitude = inf
            logging.info(f"in _permutation_loss3,inf_magnitude shape is {inf_magnitude.shape}")
            #  N X T X F
            ref_magnitude = ref
            logging.info(f"in _permutation_loss3,ref_magnitude shape is {ref_magnitude.shape}")
            # N X T X F
            mse = torch.pow(inf_magnitude - ref_magnitude, 2)
            # N X T X 1
            mse_sum1 = torch.sum(mse, -1)
            # N X 1 X1
            utt_loss = torch.sum(mse_sum1, -1)
            # utt_loss = torch.sum(torch.sum(torch.pow(masks_[int(0)]*inf - ref[int(0)], 2), -1), -1)
            loss_for_permute.append(utt_loss)
            logging.info(f"input_size device is {input_size.device}")
            logging.info(f"")
            input_size_ = torch.tensor(input_size, dtype=torch.float32, device=inf_magnitude.device) 
            logging.info(f"input_size device again is {input_size.device}")
            loss_perutt = sum(loss_for_permute) / input_size_
            return loss_perutt

        #logging.info(f"num_utts is {ref[0].shape[0]}")
        num_utts = ref.shape[0] # batch size
        logging.info(f"in _permutation_loss3,num_utts is {num_utts}")
        # O(N!), could be optimized
        # 1 x N
        pscore = torch.stack([loss()], dim=0)
        # pscore = torch.stack([loss(p) for p in permutations(range(num_spk))], dim=1)
        logging.info(f"pscore is {pscore}")
        # N
        num_spk=1
        min_perutt, _ = torch.min(pscore, dim=0)
        loss = torch.sum(min_perutt) / (num_spk * num_utts)
        """
        the loss sum freq and sum time ,then average on the time axis, then average on the number of utterances
        """
        logging.info(f"loss is {loss}")
        return loss , perm

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
