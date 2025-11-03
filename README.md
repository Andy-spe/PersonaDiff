# Introduction
**Diffusion-Based Style Control for Speech-Synchronized 3D Facial Animation**

## Dataset 
### BIWI
Please follow the BIWI data-preparation steps described in the [CodeTalker](https://github.com/Doubiiu/CodeTalker) repository to reproduce our setup.
### VOCASET
Request the VOCASET data from https://voca.is.tue.mpg.de/login.php.
### Beat
Please follow the Beat data-preparation steps described in the [FaceDiffuser](https://github.com/uuembodiedsocialai/FaceDiffuser) repository to reproduce our setup.
### 3D-ETF
The 3D-ETF dataset and download instructions can be obtained from the project repository:https://github.com/psyai-net/EmoTalk_release.

## Project

## Acknowledgement
We extend our sincere thanks to [FaceFormer](https://github.com/EvelynFan/FaceFormer),[CodeTalker](https://github.com/Doubiiu/CodeTalker),[FaceDiffuser](https://github.com/uuembodiedsocialai/FaceDiffuser),[huggingface-transformers ](https://huggingface.co/),and[Wav2Vec2](https://huggingface.co/r-f/wav2vec-english-speech-emotion-recognition).This project stands on the shoulders of giants. Thank you for helping us reach higher.


### 1. Environment
```bash
conda create -n personadiff python=3.9 -y
conda activate personadiff
pip install -r requirements.txt
