# DPSL-ASR (Dual-Path Style Learning for End-to-End Noise-Robust Automatic Speech Recognition)

[Dual-Path Style Learning for End-to-End Noise-Robust Speech Recognition](https://arxiv.org/abs/2203.14838)

[Interactive Feature Fusion for End-to-End Noise-Robust Speech Recognition](https://arxiv.org/abs/2110.05267)

## Introduction

DPSL-ASR is a novel method for end-to-end noise-robust speech recognition. It has extended our prior work IFF-Net (Interactive Feature Fusion Network) with dual-path inputs and style learning, and thus achieved better ASR performance on both [RATS Channel-A dataset](https://github.com/YUCHEN005/RATS-Channel-A-Speech-Data) and CHiME-4 1-Channel Track Dataset.

<img src="https://user-images.githubusercontent.com/90536618/160274914-a78b6752-cf5b-497d-92e8-22d6fce100ca.png" width=510> &emsp; <img src="https://user-images.githubusercontent.com/90536618/160275153-2f78ecb1-1cd4-4947-8df3-20102cf09ffb.png" width=290>

Left figure: (a) joint SE-ASR approach, (b) IFF-Net baseline, (c) the proposed DPSL-ASR approach.

Right figure: back-end ASR module with style learning and consistency loss in our DPSL-ASR. The dashed lines denote sharing parameters.

If you find DPSL-ASR useful in your research, please use the following BibTeX entry for citation:

```bash
@article{hu2022dualpath,
  title={Dual-Path Style Learning for End-to-End Noise-Robust Speech Recognition}, 
  author={Hu, Yuchen and Hou, Nana and Chen, Chen and Chng, Eng Siong},
  journal={arXiv preprint arXiv:2203.14838},
  year={2022}
}

@article{hu2021interactive,
  title={Interactive Feature Fusion for End-to-End Noise-Robust Speech Recognition},
  author={Hu, Yuchen and Hou, Nana and Chen, Chen and Chng, Eng Siong},
  journal={arXiv preprint arXiv:2110.05267},
  year={2021}
}
```

## Usage

Our code implementation is based on [ESPnet](https://github.com/espnet/espnet). You can intall it directly using our provided ESPnet(v.0.9.6) folder, or install from official website and then add files from our repo. Use the command `pip install -e .` to install ESPnet.

In our foler, the running scripts are at `egs2/rats_chA/asr_with_enhancement/{run_rats_chA_dpsl_asr, rats_chA_dpsl_asr}.sh`, and the network code are at `espnet2/{asr/, enh/, layers/}`. 

**Tips**: 

1. To go over the entire project, please start from the script `egs2/rats_chA/asr_with_enhancement/run_rats_chA_dpsl_asr.sh`
2. To read the model code only, please start from the script `espnet2/asr/dpsl_asr.py`


