#!/usr/bin/env python3
"""
Generate a dancing skeleton video from any audio file.

Usage:
    python generate.py --audio song.wav --out dance.mp4
"""

from __future__ import annotations

import argparse
import os
import subprocess

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from audio_features import AUDIO_SR, POSE_FPS, audio_to_features
from inference import generate_poses, load_checkpoint, resolve_checkpoint

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32),
]
FACE = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
LEFT = {11, 13, 15, 23, 25, 27, 29, 31}
RIGHT = {12, 14, 16, 24, 26, 28, 30, 32}


def load_audio(path: str) -> np.ndarray:
    """Load any audio file as mono float32 at 32 kHz."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != AUDIO_SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=AUDIO_SR)
    return mono.astype(np.float32)


def _segment_color(a: int, b: int) -> str:
    if a in FACE or b in FACE:
        return "#a0c4ff"
    if a in LEFT and b in LEFT:
        return "#ffd6a5"
    if a in RIGHT and b in RIGHT:
        return "#ffadad"
    return "#b9fbc0"


def render_skeleton_video(poses_xyz: np.ndarray, title: str, tmp_path: str) -> None:
    """Render a silent skeleton animation (720×1080)."""
    T = poses_xyz.shape[0]
    vel = np.diff(poses_xyz, axis=0)
    energy = np.concatenate([[0], np.linalg.norm(vel, axis=-1).mean(axis=-1)])
    energy = energy / (energy.max() + 1e-6)

    xmin = poses_xyz[:, :, 0].min() - 0.15
    xmax = poses_xyz[:, :, 0].max() + 0.15
    ymin = poses_xyz[:, :, 1].min() - 0.1
    ymax = poses_xyz[:, :, 1].max() + 0.1

    fig = plt.figure(figsize=(5.4, 8), facecolor="#0d1117")
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.86])
    ax.set_facecolor("#0d1117")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-ymax, -ymin)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.5, 0.97, title, color="#e6edf3", fontsize=11,
             ha="center", va="top", fontweight="bold")

    bar_ax = fig.add_axes([0.05, 0.02, 0.90, 0.04])
    bar_ax.set_xlim(0, T)
    bar_ax.set_ylim(0, 1)
    bar_ax.axis("off")
    progress_bar = bar_ax.barh(0.5, 0, height=0.8, color="#3fb950", left=0)
    time_txt = bar_ax.text(T * 0.5, 0.5, "0.0 s", color="white",
                           fontsize=7, ha="center", va="center")

    scat = ax.scatter([], [], s=18, zorder=4)
    lines = [ax.plot([], [], lw=2.2, solid_capstyle="round")[0] for _ in SKELETON_EDGES]

    def init():
        scat.set_offsets(np.empty((0, 2)))
        for line in lines:
            line.set_data([], [])
        return [scat, *lines]

    def update(t):
        kpts = poses_xyz[t]
        xs, ys = kpts[:, 0], -kpts[:, 1]
        cols = []
        for i in range(33):
            if i in FACE:
                cols.append("#a0c4ff")
            elif i in LEFT:
                cols.append("#ffd6a5")
            elif i in RIGHT:
                cols.append("#ffadad")
            else:
                cols.append("#b9fbc0")
        scat.set_offsets(np.c_[xs, ys])
        scat.set_color(cols)
        for idx, (a, b) in enumerate(SKELETON_EDGES):
            lines[idx].set_data([xs[a], xs[b]], [ys[a], ys[b]])
            lines[idx].set_color(_segment_color(a, b))
            lines[idx].set_alpha(0.85 + 0.15 * energy[t])
        progress_bar[0].set_width(t + 1)
        time_txt.set_text(f"{t / POSE_FPS:.1f} s / {T / POSE_FPS:.1f} s")
        return [scat, *lines, progress_bar[0], time_txt]

    ani = animation.FuncAnimation(
        fig, update, frames=T, init_func=init,
        interval=1000 / POSE_FPS, blit=True,
    )
    writer = animation.FFMpegWriter(
        fps=POSE_FPS, bitrate=2000,
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
    )
    ani.save(tmp_path, writer=writer, dpi=150)
    plt.close(fig)


def mux_audio(video_path: str, audio_path: str, out_path: str, duration: float) -> None:
    """Combine silent video with the original audio track."""
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-t", str(duration),
        "-shortest",
        out_path,
    ], check=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Music → dance skeleton video")
    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt weights")
    parser.add_argument("--out", default="dance.mp4", help="Output video path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_checkpoint(args.checkpoint)

    print(f"Device: {device}")
    print(f"Loading audio: {args.audio}")
    waveform = load_audio(args.audio)
    duration = len(waveform) / AUDIO_SR
    print(f"  Duration: {duration:.1f}s")

    print("Extracting audio features…")
    audio_feat = audio_to_features(waveform)
    print(f"  {audio_feat.shape[0]} frames @ {POSE_FPS} fps")

    print(f"Loading model: {ckpt_path}")
    model, ckpt = load_checkpoint(ckpt_path, device)
    print(f"  Epoch {ckpt['epoch']}")

    print("Generating poses…")
    poses_xyz = generate_poses(
        model, audio_feat,
        ckpt["x_mean"], ckpt["x_std"],
        ckpt["y_mean"], ckpt["y_std"],
        device,
    )

    title = os.path.splitext(os.path.basename(args.audio))[0]
    tmp = args.out.replace(".mp4", "_silent.mp4")
    print("Rendering video…")
    render_skeleton_video(poses_xyz, title=title, tmp_path=tmp)

    print("Mixing audio…")
    mux_audio(tmp, args.audio, args.out, duration)
    os.remove(tmp)
    print(f"Done → {args.out}")


if __name__ == "__main__":
    main()
