# DPSL-ASR (Dual-Path Style Learning for End-to-End Noise-Robust Automatic Speech Recognition)

[Dual-Path Style Learning for End-to-End Noise-Robust Speech Recognition](https://arxiv.org/abs/2203.14838)

[Interactive Feature Fusion for End-to-End Noise-Robust Speech Recognition](https://arxiv.org/abs/2110.05267)

## Introduction

DPSL-ASR is a novel method for end-to-end noise-robust speech recognition. It extends our prior work IFF-Net (Interactive Feature Fusion Network) with dual-path inputs and style learning, which achieves better ASR performance on [Robust Automatic Transcription of Speech (RATS)](https://github.com/YUCHEN005/RATS-Channel-A-Speech-Data) and [CHiME-4](https://spandh.dcs.shef.ac.uk/chime_challenge/CHiME4/data.html) datasets.

<img width=510 src="https://user-images.githubusercontent.com/90536618/196597886-bd3af18c-0cd7-4852-8066-5b5872531b0c.png"> &emsp; <img width=290 src="https://user-images.githubusercontent.com/90536618/196597890-55bdcd9a-e958-476a-b1d3-248b1ba563ea.png">


Left figure: (a) joint SE-ASR approach, (b) IFF-Net baseline, (c) the proposed DPSL-ASR approach.

Right figure: back-end ASR module with style learning and consistency loss in our DPSL-ASR. The dashed lines denote sharing parameters.

If you find DPSL-ASR useful in your research, please use the following BibTeX entry for citation:

```bash
@article{hu2022dual,
  title={Dual-Path Style Learning for End-to-End Noise-Robust Speech Recognition}, 
  author={Hu, Yuchen and Hou, Nana and Chen, Chen and Chng, Eng Siong},
  journal={arXiv preprint arXiv:2203.14838},
  year={2022}
}

@inproceedings{hu2022interactive,
  title={Interactive Feature Fusion for End-to-End Noise-Robust Speech Recognition},
  author={Hu, Yuchen and Hou, Nana and Chen, Chen and Chng, Eng Siong},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6292--6296},
  year={2022},
  organization={IEEE}
}
```

## Usage

Our code implementation is based on [ESPnet](https://github.com/espnet/espnet). You can intall it directly using our provided ESPnet(v.0.9.6) folder, or install from official website and then add files from our repo. Use the command `pip install -e .` to install ESPnet.

In our folder, the running scripts are at `egs2/rats_chA/asr_with_enhancement/{run_rats_chA_dpsl_asr, rats_chA_dpsl_asr}.sh`, and the network code are at `espnet2/{asr/, enh/, layers/}`. 

**Tips**: 

1. To go over the entire project, please start from the script `egs2/rats_chA/asr_with_enhancement/run_rats_chA_dpsl_asr.sh` [[link]](https://github.com/YUCHEN005/DPSL-ASR/blob/master/egs2/rats_chA/asr_with_enhancement/run_rats_chA_dpsl_asr.sh)
2. To read the network code of DPSL-ASR, please refer to the script `espnet2/asr/dpsl_asr.py` [[link]](https://github.com/YUCHEN005/DPSL-ASR/blob/master/espnet2/asr/dpsl_asr.py)


