# DPSL-ASR (Dual-Path Style Learning)

[Dual-Path Style Learning for End-to-End Noise-Robust Speech Recognition](https://arxiv.org/abs/2203.14838)

[Interactive Feature Fusion for End-to-End Noise-Robust Speech Recognition](https://arxiv.org/abs/2110.05267)

## Introduction

DPSL-ASR is a novel method for end-to-end noise-robust speech recognition. It extends our prior work IFF-Net (Interactive Feature Fusion Network) with dual-path inputs and style learning, which achieves better ASR performance on [Robust Automatic Transcription of Speech (RATS)](https://github.com/YUCHEN005/RATS-Channel-A-Speech-Data) and [CHiME-4](https://spandh.dcs.shef.ac.uk/chime_challenge/CHiME4/data.html) datasets.

<img width=510 src="https://user-images.githubusercontent.com/90536618/196597886-bd3af18c-0cd7-4852-8066-5b5872531b0c.png"> &emsp; <img width=290 src="https://user-images.githubusercontent.com/90536618/196597890-55bdcd9a-e958-476a-b1d3-248b1ba563ea.png">


Left figure: (a) joint SE-ASR approach, (b) IFF-Net baseline, (c) our proposed DPSL-ASR approach.

Right figure: back-end ASR module with style learning and consistency loss in our DPSL-ASR. The dashed arrows denote sharing parameters.

If you find DPSL-ASR or IFF-Net useful in your research, please kindly use the following BibTeX entry for citation:

```bash
@inproceedings{hu2023dual,
  title={Dual-Path Style Learning for End-to-End Noise-Robust Speech Recognition}, 
  author={Hu, Yuchen and Hou, Nana and Chen, Chen and Chng, Eng Siong},
  booktitle={INTERSPEECH},
  year={2023}
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

Our code implementation is based on [ESPnet](https://github.com/espnet/espnet) (v.0.9.6), please kindly use the following commands for installation.

```bash
git clone https://github.com/YUCHEN005/DPSL-ASR.git
cd DPSL-ASR
pip install -e .
```

Experiment directory is at `egs2/rats_chA/asr_with_enhancement/`, and the network code is at `espnet2/asr/dpsl_asr.py`. 
