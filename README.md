# <img src="assets/crema.png" alt="Image description" class="title-icon" style="width: 40px; height: auto;"> CREMA: Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://crema-videollm.github.io/)  [![arXiv](https://img.shields.io/badge/arXiv-2402.05889-b31b1b.svg)](https://arxiv.org/abs/2402.05889)   [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20-cyan.svg)](https://huggingface.co/Shoubin/CREMA/tree/main)

### Authors: [Shoubin Yu*](https://yui010206.github.io/), [Jaehong Yoon*](https://jaehong31.github.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)
### University of North Carolina at Chapel Hill

<div align=center> 
<!-- <img src="./assets/teaser.png" alt="teaser image" width="450"/> -->
</div>
<br>
<img src="./assets/crema_method.png" alt="teaser image" width="1000"/>

# 🔥 News
- **Jun 14, 2024**. Check our new [arXiv-version2](https://arxiv.org/abs/2402.05889) for exciting additions to CREMA:
  - New modality-sequential modular training & modality-adaptive early exit strategy to handle learning with many modalities.
  - More unique/rare multimodal reasoning tasks (video-touch and video-thermal QA) to further demonstrate the generalizability of CREMA


# Code structure
```bash

# CREMA code
./lavis/

# running scripts for CREMA training/inference
./run_scripts

```

# Setup

## Install Dependencies

1. (Optional) Creating conda environment

```bash
conda create -n crema python=3.8
conda activate crema
```

2. build from source

```bash
pip install -e .
```


## Download Models

### Pre-trained Models

**Visual Encoder**: we adopt pre-trained [ViT-G](https://arxiv.org/abs/2303.15389) (1B), the codebase downloads the model automatically.

**Audio Encoder**: we use pre-trained [BEATs (iter3+)](https://arxiv.org/abs/2212.09058), please download the model [here](https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D), and update the path in the [code](https://github.com/Yui010206/CREMA/blob/main/lavis/models/beats_encoder.py#L18)

**3D Encoder**: we conduct off-line feature extraction following [3D-LLM](https://arxiv.org/abs/2307.12981), please refer to this [page](https://drive.google.com/drive/folders/1CsEt48jj5uCyelGcXXJBkGH86QYeCE8D) for per-extracted features. Please change the storage in [dataset config](https://github.com/Yui010206/CREMA/blob/main/lavis/configs/datasets/sqa3d/defaults.yaml#L66).  

**Multimodal Qformer**: We initialize query tokens and FC layer for each MMQA in Multimodal Q-Former form pre-trained BLIP-2 model checkpoints.
We hold Multimodal Q-Fromer with pre-trained MMQA-audio and MMQA-3D via [HuggingFace](https://huggingface.co/Shoubin/CREMA/resolve/main/crema_pretrained.pth),
and Multimodal Q-Fromer initilized from BLIP-2 can be found [here](https://huggingface.co/Shoubin/CREMA/resolve/main/crema_initial.pth).

### Fine-tuned Models

| Dataset | Modalities |
| :---    |    :----  | 
| SQA3D | [Video+3D+Depth+Norm](https://huggingface.co/Shoubin/CREMA/blob/main/v2_ckpt/sqa3d_v3dn.pth) |
| MUSIC-AVQA | [Video+Audio+Flow+Norm+Depth](https://huggingface.co/Shoubin/CREMA/blob/main/v2_ckpt/musicavqa_vadfn.pth) |
| NExT-QA | [Video+Flow+Depth+Normal](https://huggingface.co/Shoubin/CREMA/blob/main/v2_ckpt/nextqa_vdfn.pth) |


# Dataset Preparation & Feature Extraction
We test our model on:

+ [SQA3D](https://github.com/SilongYong/SQA3D): we follow 3D-LLM data [format](https://drive.google.com/drive/folders/14_-cWOMW-Hp6T5_c6YMjwDzQOgfiO2p9).

+ [MUSIC-AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA): we follow the orginal MUSIC-AVQA data [format](https://github.com/GeWu-Lab/MUSIC-AVQA/tree/main/data/json_update).

+ [NExT-QA](https://github.com/doc-doc/NExT-QA): we follow SeViLA data [format](https://github.com/Yui010206/SeViLA/tree/main/sevila_data).

+ Touch-QA (reformulated from [Touch&Go](https://touch-and-go.github.io/)): we follow SeViLA data [format](https://github.com/Yui010206/SeViLA/tree/main/sevila_data), and released our data [here](https://huggingface.co/Shoubin/CREMA/tree/main/data/touchqa).

+ Thermal-QA (reformulated from [Thermal-IM](https://github.com/ZitianTang/Thermal-IM)): we follow SeViLA data [format](https://github.com/Yui010206/SeViLA/tree/main/sevila_data), and released our data [here](https://huggingface.co/Shoubin/CREMA/tree/main/data/thermalqa). 

To get trimmed Touch-QA and Thermal-QA video frames, you can first download raw videos from each original data project, and preprocess with our [scripts](https://github.com/Yui010206/CREMA/tree/main/tools) after setting the custom data path, by running.

```bash
python trim_video.py

python decode_frames.py
```

We extract various extra modalities from raw video with pre-train models, please refer to each model repo and paper appendix for more details.

+ [Depth map](https://github.com/isl-org/ZoeDepth)

+ [Optical flow](https://github.com/autonomousvision/unimatch)

+ [Surface normals](https://github.com/baegwangbin/surface_normal_uncertainty)

We will share extracted features in the following table. 
| Dataset | Multimodal Features |
| :----    |    :----  | 
| SQA3D | [Video Frames](), [Depth Map](), [Surface Normals]() |
| MUSIC-AVQA | [Video Frames](), [Optical Flow]() , [Depth Map](), [Surface Normals]() |
| NExT-QA | [Video Frames](), [Depth Map](), [Optical Flow](), [Surface Normals]() |
| Touch-QA | [Video Frames](), [Surface Normals]() |
| Thermal-QA | [Video Frames](), [Depth Map]() |


We pre-train MMQA in our CRMEA framework with public modality-specific datasets:
  
+ [AudioCaps](https://audiocaps.github.io/) for MMQA-Audio

+ [3D-LLM](https://github.com/UMass-Foundation-Model/3D-LLM) for MMQA-3D


# Training and Inference
We provide CREMA training and inference script examples as follows.

## 1) Training

```bash
sh run_scripts/crema/finetune/sqa3d.sh
```

## 2) Inference

```bash
sh run_scripts/crema/inference/sqa3d.sh
```


# Acknowledgments
We thank the developers of [LAVIS](https://github.com/salesforce/LAVIS), [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [CLIP](https://github.com/openai/CLIP), [X-InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/xinstructblip), for their public code release.


# Reference
Please cite our paper if you use our models in your works:

```bibtex
@article{yu2024crema,
  title={CREMA: Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion},
  author={Yu, Shoubin and Yoon, Jaehong and Bansal, Mohit},
  journal={arXiv preprint arXiv:2402.05889},
  year={2024}
}
