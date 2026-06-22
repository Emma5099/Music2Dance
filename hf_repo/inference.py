"""Load checkpoint and run autoregressive pose generation."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch

from audio_features import CONTEXT_LEN
from model import Music2PoseTransformer

DEFAULT_CHECKPOINT = "pytorch_model.pt"


def load_checkpoint(
    ckpt_path: str = DEFAULT_CHECKPOINT,
    device: torch.device | None = None,
) -> Tuple[Music2PoseTransformer, dict]:
    """Load model weights and normalisation stats from a checkpoint file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Music2PoseTransformer(
        audio_dim=ckpt["audio_dim"],
        pose_dim=ckpt["pose_dim"],
        **ckpt["config"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def generate_poses(
    model: Music2PoseTransformer,
    audio_feat: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Autoregressive inference.

    Returns
    -------
    poses_xyz : (T, 33, 3) normalised MediaPipe-style joint positions.
    """
    T = audio_feat.shape[0]
    af_norm = (audio_feat - x_mean) / x_std
    af_t = torch.tensor(af_norm, dtype=torch.float32, device=device)
    pose_dim = model.pose_dim

    audio_ctx = torch.zeros(1, CONTEXT_LEN, af_t.shape[-1], device=device)
    pose_ctx = torch.zeros(1, CONTEXT_LEN, pose_dim, device=device)
    preds = []

    with torch.no_grad():
        for t in range(T):
            audio_ctx = torch.roll(audio_ctx, -1, dims=1)
            audio_ctx[0, -1] = af_t[t]
            next_pose = model.step(audio_ctx, pose_ctx)[0]
            preds.append(next_pose.cpu().numpy())
            pose_ctx = torch.roll(pose_ctx, -1, dims=1)
            pose_ctx[0, -1] = next_pose

    pred_norm = np.stack(preds)
    pred_raw = pred_norm * y_std + y_mean
    return pred_raw.reshape(T, 33, 9)[:, :, :3]


def resolve_checkpoint(path: str | None = None) -> str:
    """Return checkpoint path, defaulting to the bundled weights."""
    if path:
        return path
    if os.path.exists(DEFAULT_CHECKPOINT):
        return DEFAULT_CHECKPOINT
    raise FileNotFoundError(
        f"No checkpoint found. Download weights or pass --checkpoint explicitly."
    )
