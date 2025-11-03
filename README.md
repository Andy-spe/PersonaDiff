# PersonaDiff
**Style-controllable diffusion for speech-driven 3D facial animation**

Brief: PersonaDiff is a diffusion-based framework that generates high-fidelity, style-controllable 3D facial motion from speech. It disentangles dynamic and static speaking style and conditions a denoising decoder on both audio content and an expression-style vector.

## Features
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
