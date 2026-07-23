# OSX + SignAvatars Pipeline

End-to-end pipeline: **input video → OSX 3D pose estimation → SignAvatars rendering → avatar animation video**.

Built on [OSX](https://github.com/IDEA-Research/OSX) (one-stage SMPL-X estimation) and [SignAvatars](https://github.com/ssi-research/SignAvatars) (sign-language avatar rendering).

## Directory Structure

```
OSX_Project/
├── code/
│   ├── OSX/               # OSX inference code
│   └── SignAvatars/       # SignAvatars rendering code
├── models/
│   ├── human_model_files/ # SMPL-X / SMPL / MANO / FLAME model files  ← Kaggle Dataset
│   └── pretrained_models/ # OSX checkpoint (.pth.tar)                 ← Kaggle Dataset
├── videos/                # Input videos                               ← Kaggle Dataset
├── outputs/               # Rendering outputs (ignored by git)
├── config.yaml            # Pipeline configuration
├── run.py                 # CLI entry point
├── setup.py               # Environment validation
├── requirements.txt       # Python dependencies
└── .gitignore
```

Large files (`models/`, `videos/`, `outputs/`, `*.pth.tar`, `*.pkl`, `*.mp4`) are excluded from git and must be supplied separately as a Kaggle Dataset (see below).

## Requirements

- Python 3.8+
- PyTorch (CUDA recommended)
- See `requirements.txt` for full dependency list.

## Quick Start

### 1. Validate environment

```bash
python run.py doctor
```

Checks Python version, PyTorch/CUDA, directory structure, model files, and checkpoint.

### 2. Run inference on a video

```bash
python run.py infer --video path/to/video.mp4
```

Outputs a `.pkl` file with SMPL-X parameters for each frame.

### 3. Render the avatar

```bash
python run.py render --pkl path/to/output.pkl
```

Generates an avatar animation video.

### 4. Inspect a `.pkl` file

```bash
python run.py inspect --pkl path/to/output.pkl
```

Validates keys, shapes, and dtypes.

### 5. Convert OSX JSON to SignAvatars PKL

```bash
python run.py convert --input results.json --output results.pkl
```

## Kaggle Deployment

### Setting up the Dataset

1. Upload `models/` (human_model_files + pretrained_models) and `videos/` as a Kaggle Dataset.
2. On Kaggle, the dataset will be mounted at `/kaggle/input/<dataset-name>/`.
3. Create a symlink or copy so the project can find models and videos:

```bash
ln -s /kaggle/input/<dataset-name>/models /kaggle/working/models
ln -s /kaggle/input/<dataset-name>/videos /kaggle/working/videos
```

### Running on Kaggle

```bash
python run.py doctor
python run.py infer --video videos/sample.mp4
python run.py render --pkl outputs/sample.pkl
```

## Commands

| Command | Description |
|---------|-------------|
| `doctor` | Validate runtime environment |
| `infer` | Run OSX inference on a video file |
| `render` | Render SMPL-X parameters to avatar animation |
| `inspect` | Inspect a generated `.pkl` file |
| `convert` | Convert OSX JSON output to SignAvatars `.pkl` |

## Credits

- **OSX**: [IDEA-Research/OSX](https://github.com/IDEA-Research/OSX)
- **SignAvatars**: [ssi-research/SignAvatars](https://github.com/ssi-research/SignAvatars)
- **SMPL-X**: [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/)
