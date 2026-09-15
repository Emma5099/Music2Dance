---
license: mit
tags:
- dance
- pose-generation
- music
- transformer
- audio-to-motion
library_name: pytorch
pipeline_tag: other
---

# Dance2Music — Music2Pose Transformer

Generate dancing skeleton animations from any audio file using a causal transformer trained on [AIST++](https://google.github.io/aistplusplus_dataset/) dance videos.

**Author:** [Emma Genthon](https://huggingface.co/Emma5099) · genthonemma@gmail.com  
**Best checkpoint:** epoch 250 (`pytorch_model.pt`)

## What it does

Given a music track, the model autoregressively predicts 33 MediaPipe-style body landmarks at 25 fps, then renders them as a skeleton video synced to the audio.

```
Audio (32 kHz) → mel + onset + beat features → Music2PoseTransformer → poses (33×3) → skeleton video
```

## Quick start

```bash
git clone https://huggingface.co/Emma5099/dance2music
cd dance2music
pip install -r requirements.txt

# Requires ffmpeg on PATH
python generate.py --audio your_song.wav --out dance.mp4
```

### Python API

```python
import soundfile as sf
import librosa
import torch

from audio_features import AUDIO_SR, audio_to_features
from inference import load_checkpoint, generate_poses

waveform, sr = sf.read("song.wav", dtype="float32")
if waveform.ndim > 1:
    waveform = waveform.mean(axis=1)
if sr != AUDIO_SR:
    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=AUDIO_SR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, ckpt = load_checkpoint("pytorch_model.pt", device)
poses = generate_poses(
    model, audio_to_features(waveform),
    ckpt["x_mean"], ckpt["x_std"],
    ckpt["y_mean"], ckpt["y_std"],
    device,
)
# poses.shape → (T, 33, 3)
```

## Model

| Property | Value |
|---|---|
| Architecture | Causal transformer decoder with audio cross-attention |
| Hidden size | 256 |
| Layers / heads | 6 / 8 |
| Audio input | 80 mel bands + onset strength + beat flag (82 dims/frame) |
| Pose output | 33 landmarks × 9 channels (xyz + velocity + acceleration), inference uses xyz only |
| Context window | 60 frames (2.4 s rolling buffer) |
| Training epochs | 250 (latest best checkpoint) |

### Validation metrics (sample `sBT_c01_v7`)

| Metric | Value |
|---|---|
| MSE (xyz) | 0.624 |
| Motion speed | 18.4% of ground truth |
| Balanced score | 1.306 |

The model produces human-shaped skeletons with visible motion. Beat alignment is still limited and improves with more training data and epochs.

## Files

| File | Description |
|---|---|
| `pytorch_model.pt` | Best checkpoint (epoch 250) with weights + normalisation stats |
| `model.py` | `Music2PoseTransformer` architecture |
| `audio_features.py` | Mel/onset/beat feature extraction |
| `inference.py` | Checkpoint loading and autoregressive generation |
| `generate.py` | End-to-end CLI: audio → skeleton MP4 |
| `config.json` | Model hyperparameters and metadata |

## Training (optional)

Training uses AIST++ pose sequences paired with audio chunks, weighted losses on position/velocity/acceleration, and scheduled sampling with a rolling 60-frame context. See the full project source for `train.py`, `evaluate.py`, and dataset tooling.

## Requirements

- Python 3.10+
- PyTorch 2.x
- ffmpeg (for video rendering and audio muxing)
- ~26 MB disk for model weights

## License

MIT

## Citation

If you use this model, please cite AIST++ and this repository:

```bibtex
@misc{genthon2026dance2music,
  author = {Emma Genthon},
  title = {Dance2Music: Music-Conditioned Pose Generation},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/Emma5099/dance2music}}
}
```
