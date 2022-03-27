# DPSL-ASR (Dual-Path Style Learning for Noise-Robust Automatic Speech Recognition)

[Dual-Path Style Learning for Noise-Robust Automatic Speech Recognition]()

[Interactive Feature Fusion for End-to-End Noise-Robust Speech Recognition](https://arxiv.org/abs/2110.05267)

## Introduction

DPSL-ASR is novel method for end-to-end noise-robust Speech Recognition, which has extended our prior work IFF-Net (Interactive Feature Fusion Network) with dual-path inputs and style learning, and achieved better ASR performance on [RATS Channel-A dataset](https://github.com/YUCHEN005/RATS-Channel-A-Speech-Data) and CHiME-4 1-Channel Track Dataset.

<img src="https://user-images.githubusercontent.com/90536618/160274914-a78b6752-cf5b-497d-92e8-22d6fce100ca.png" width=570> &emsp; <img src="https://user-images.githubusercontent.com/90536618/160275153-2f78ecb1-1cd4-4947-8df3-20102cf09ffb.png" width=350>

Left figure: (a) joint SE-ASR approach, (b) IFF-Net baseline, (c) the proposed DPSL-ASR approach.

Right figure: back-end ASR module with style learning and consistency loss in our DPSL-ASR. The dashed lines denote sharing parameters.

If you find DPSL-ASR or IFF-Net useful in your research, please use the following BibTeX entry for citation:

```bash
@article{hu2021interactive,
  title={Interactive Feature Fusion for End-to-End Noise-Robust Speech Recognition},
  author={Hu, Yuchen and Hou, Nana and Chen, Chen and Chng, Eng Siong},
  journal={arXiv preprint arXiv:2110.05267},
  year={2021}
}
```

## Usage

Our code implementation is based on [ESPnet](https://github.com/espnet/espnet). You can intall it directly using our provided ESPnet(v.0.9.6) folder, or install from official website and then add files from our repo. Use the install command `pip install -e .`

In our foler, the running scripts are at `egs2/rats_chA/asr_with_enhancement/{run_rats_chA_dpsl_asr, rats_chA_dpsl_asr}.sh`, and the network code are at `espnet2/{asr, enh, layers}`. Tip: start from the script `run_rats_chA_dpsl_asr.sh` and you can easily go over the entire project.
