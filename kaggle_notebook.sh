#!/bin/bash
# ============================================================
# Kaggle Notebook: OSX + SignAvatars Pipeline Setup
# ============================================================
# Copy-paste each section into a separate cell in your Kaggle
# notebook (GPU T4 x2, Internet ON).
#
# Dataset structure required:
# /kaggle/input/<your-dataset>/
#   models/
#     human_model_files/   (SMPL-X bodies, MANO, FLAME — from original OSX repo)
#     pretrained_models/   (osx_l_wo_decoder.pth.tar)
#   videos/
#     test_sample.mp4
# ============================================================

# ---- Cell 1: Clone repo ----
git clone https://github.com/Aaron-jsx/Traduccion-De-Se-asV2.git
cd /kaggle/working/Traduccion-De-Se-asV2

# ---- Cell 2: Create legacy conda env (Python 3.8) ----
conda create -n osx python=3.8 -y

# ---- Cell 3: Install PyTorch 1.13.1 + CUDA 11.7 ----
conda run -n osx pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    -f https://download.pytorch.org/whl/torch_stable.html

# ---- Cell 4: Install mmcv-full 1.7.1 (pre-built, no compilation) ----
conda run -n osx pip install mmcv-full==1.7.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

# ---- Cell 5: Install project Python dependencies ----
conda run -n osx pip install -r requirements.txt
conda run -n osx pip install opencv-python opencv-contrib-python \
    numpy-quaternion pyrender trimesh shapely prettytable tensorboardX \
    pycocotools plotly open3d imageio-ffmpeg timm einops

# ---- Cell 6: Install vendored mmpose 0.28.0 ----
conda run -n osx pip install -e code/OSX/main/transformer_utils

# ---- Cell 7: Symlink models and videos from Kaggle Dataset ----
DATASET="/kaggle/input/traduccion-de-senas"   # ← CHANGE ME to your dataset name
ln -sfn "$DATASET/models" models
ln -sfn "$DATASET/videos" videos

# ---- Cell 8: Diagnostic ----
conda run -n osx python run.py doctor

# ---- Cell 9: Infer ----
conda run -n osx python run.py infer --video videos/test_sample.mp4

# ---- Cell 10: Render (SignAvatars) ----
conda run -n osx python run.py render --pkl outputs/test_sample.pkl
