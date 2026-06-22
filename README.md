# Music2Dance

Generate dancing skeleton animations from music using a causal transformer trained on [AIST++](https://google.github.io/aistplusplus_dataset/) dance videos.

**Author:** [Emma Genthon](https://github.com/Emma5099) · genthonemma@gmail.com  
**Pretrained weights:** [Emma5099/dance2music](https://huggingface.co/Emma5099/dance2music) (epoch 180)

## Overview

```
Audio (32 kHz) → mel + onset + beat → Music2PoseTransformer → poses (33×3) → skeleton video
```

The model predicts 33 MediaPipe-style body landmarks at 25 fps from audio, then renders a skeleton animation synced to the track.

## Quick start (inference)

```bash
git clone https://github.com/Emma5099/Music2Dance
cd Music2Dance
pip install -r requirements.txt

# Download weights from Hugging Face
huggingface-cli download Emma5099/dance2music pytorch_model.pt --local-dir checkpoints

# Generate a dance video (requires ffmpeg)
python generate_from_audio.py --audio song.wav --checkpoint checkpoints/pytorch_model.pt --out dance.mp4
```

## Evaluate on the validation set

```bash
python evaluate.py --checkpoint checkpoints/pytorch_model.pt
```

Produces `outputs/predicted.mp4`, `outputs/groundtruth.mp4`, and `outputs/beat_score.txt`.

## Training pipeline

### 1. Download AIST++ videos

```bash
python download_aist.py --download_folder videos/
```

Agree to the [AIST terms of use](https://aistdancedb.ongaaccel.jp/terms_of_use/) before downloading.

### 2. Extract poses & build dataset

```bash
python data/extract_poses.py --video_dir videos/ --output_dir poses/
python data/build_dataset.py --poses_dir poses/ --videos_dir videos/ --output dataset/
```

### 3. Train

```bash
python train.py --dataset_path dataset/ --epochs 200
```

Resume from the published checkpoint:

```bash
python train.py --dataset_path dataset/ --resume checkpoints/pytorch_model.pt --epochs 200
```

## Project layout

| File | Description |
|---|---|
| `model.py` | `Music2PoseTransformer` architecture |
| `train.py` | Training loop with scheduled sampling and beat-aware losses |
| `evaluate.py` | Validation inference, metrics, and skeleton video rendering |
| `generate_from_audio.py` | Inference on any audio file |
| `download_aist.py` | AIST++ video downloader |
| `data/extract_poses.py` | MediaPipe pose extraction from videos |
| `data/build_dataset.py` | Build HuggingFace dataset of (pose, audio) pairs |
| `push_to_hf.py` | Upload weights to Hugging Face Hub |

## Model

| Property | Value |
|---|---|
| Architecture | Causal transformer decoder with audio cross-attention |
| Hidden size | 256 · 6 layers · 8 heads |
| Audio input | 80 mel bands + onset + beat (82 dims/frame) |
| Pose output | 33 landmarks × 9 channels (xyz + velocity + acceleration) |
| Context window | 60 frames (2.4 s rolling buffer) |
| Best checkpoint | Epoch 180 |

### Validation metrics (sample `sBT_c01_v7`)

| Metric | Value |
|---|---|
| MSE (xyz) | 0.624 |
| Motion speed | 18.4% of ground truth |
| Balanced score | 1.306 |

## Requirements

- Python 3.10+
- PyTorch 2.x
- ffmpeg (video rendering and audio extraction)
- GPU recommended for training

## License

MIT

## Citation

```bibtex
@misc{genthon2026music2dance,
  author = {Emma Genthon},
  title = {Music2Dance: Music-Conditioned Pose Generation},
  year = {2026},
  howpublished = {\url{https://github.com/Emma5099/Music2Dance}}
}
```
