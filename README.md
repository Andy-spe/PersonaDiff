# Introduction
**Diffusion-Based Style Control for Speech-Synchronized 3D Facial Animation**

## Dataset 
### BIWI
To ensure a fair comparison with previous methods, we adopt the same BIWI preprocessing and train/val/test splits used in CodeTalker. Please follow the BIWI data-preparation steps described in the CodeTalker repository ([访问 GitHub]([https://github.com])) to reproduce our setup.
### VOCASET
### Beat
### 3D-ETF
The 3D-ETF dataset and download instructions can be obtained from the project repository:https://github.com/psyai-net/EmoTalk_release.


- Motion encoder for content-related facial motion
- Two-branch audio encoders for content and emotion
- Audiovisual fusion producing expression-style vector
- Diffusion decoder with multi-head rotary attention and MotionAmplifier
- Supports vertex and blendshape targets, evaluation scripts, and visualization tools

## Quick start

### 1. Environment
```bash
conda create -n personadiff python=3.9 -y
conda activate personadiff
pip install -r requirements.txt
