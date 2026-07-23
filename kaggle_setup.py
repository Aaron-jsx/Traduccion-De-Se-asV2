"""Kaggle Notebook: OSX + SignAvatars Pipeline Setup.

Copy-paste the sections below into individual Kaggle notebook cells.
Requires: GPU Accelerator (T4 x2 recommended), Internet enabled.

Cell 1 — Import dataset from Kaggle
(use the Kaggle UI to add your dataset containing models/ and videos/)
"""

# ============================================================
# Cell 1 — Clone the repository
# ============================================================
import os, sys, subprocess

REPO_URL = "https://github.com/Aaron-jsx/Traduccion-De-Se-asV2.git"
REPO_DIR = "/kaggle/working/OSX_Project"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)

# ============================================================
# Cell 2 — Create legacy conda environment (Python 3.8)
# ============================================================
import subprocess

# Names are critical — train_env is the convention used by OSX authors.
# Python 3.8 matches the SignAvatars README spec.
subprocess.run([
    "conda", "create", "-n", "osx", "python=3.8", "-y"
], check=True)

# ============================================================
# Cell 3 — Install PyTorch 1.13.1 + CUDA 11.7
# ============================================================
subprocess.run([
    "conda", "run", "-n", "osx", "pip", "install",
    "torch==1.13.1+cu117", "torchvision==0.14.1+cu117",
    "-f", "https://download.pytorch.org/whl/torch_stable.html"
], check=True)

# ============================================================
# Cell 4 — Install mmcv-full 1.7.1 (pre-built wheel, no compile)
# ============================================================
subprocess.run([
    "conda", "run", "-n", "osx", "pip", "install", "mmcv-full==1.7.1",
    "-f", "https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html"
], check=True)

# ============================================================
# Cell 5 — Install project dependencies
# ============================================================
subprocess.run([
    "conda", "run", "-n", "osx", "pip", "install",
    "-r", "requirements.txt"
], check=True)

subprocess.run([
    "conda", "run", "-n", "osx", "pip", "install",
    "opencv-python", "opencv-contrib-python", "numpy-quaternion",
    "pyrender", "trimesh", "shapely", "prettytable",
    "tensorboardX", "pycocotools", "plotly", "open3d",
    "imageio-ffmpeg", "numba", "filterpy", "cython",
    "einops", "boto3", "seaborn", "setuptools"
], check=True)

# ============================================================
# Cell 6 — Install vendored mmpose 0.28.0
# ============================================================
subprocess.run([
    "conda", "run", "-n", "osx", "pip", "install", "-e",
    "code/OSX/main/transformer_utils"
], check=True)

# Install compatibility patches (will be imported automatically by run.py)
subprocess.run([
    "conda", "run", "-n", "osx", "pip", "install", "chumpy"
], check=True)

# ============================================================
# Cell 7 — Set up dataset symlinks
# ============================================================
# The Kaggle Dataset should have this structure:
# /kaggle/input/<dataset-name>/
#   models/
#     human_model_files/       (SMPL-X bodies, MANO, FLAME)
#     pretrained_models/       (osx_l_wo_decoder.pth.tar)
#   videos/
#     test_sample.mp4
#
# Adjust the path below:
DATASET_PATH = "/kaggle/input/traduccion-de-senas"  # CHANGE ME

for target, link_name in [
    (f"{DATASET_PATH}/models", f"{REPO_DIR}/models"),
    (f"{DATASET_PATH}/videos", f"{REPO_DIR}/videos"),
]:
    if not os.path.exists(link_name):
        os.symlink(target, link_name)

# ============================================================
# Cell 8 — Run diagnostic
# ============================================================
subprocess.run(["conda", "run", "-n", "osx", "python", "run.py", "doctor"])

# ============================================================
# Cell 9 — Run inference
# ============================================================
subprocess.run([
    "conda", "run", "-n", "osx", "python", "run.py", "infer",
    "--video", "videos/test_sample.mp4"
])

# ============================================================
# Cell 10 — Run render (SignAvatars visualization)
# ============================================================
subprocess.run([
    "conda", "run", "-n", "osx", "python", "run.py", "render",
    "--pkl", "outputs/test_sample.pkl"
])
