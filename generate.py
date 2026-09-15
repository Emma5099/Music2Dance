#!/usr/bin/env python3
"""
Music2Dance CLI — music → skeleton and/or 3D character dance.

Examples:
  python generate.py --audio song.wav
  python generate.py --audio song.wav --mode skeleton --format gif
  python generate.py --audio song.wav --mode character --format mp4
  python generate.py --audio song.wav --mode both --format both --max-seconds 15
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from generate_from_audio import (
    AUDIO_SR,
    POSE_FPS,
    audio_to_features,
    load_audio,
    load_model,
    mux_audio,
    render_skeleton_video,
    run_inference,
)
from pose_postprocess import clean_dance_poses, cut_audio_to_segments, truncate_poses

ROOT = Path(__file__).resolve().parent
DEFAULT_MESH = ROOT / "3Dmodel" / "character.glb"
DEFAULT_MAX_SECONDS = 20.0
DEFAULT_SOURCE_SECONDS = 40.0
ANIMATE_SCRIPT = ROOT / "3Dmodel" / "animate_character.py"


def _resolve_checkpoint(path: str | None) -> Path:
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    candidates = [
        ROOT / "checkpoints" / "pytorch_model.pt",
        ROOT / "checkpoints" / "epoch_250.pt",
        ROOT / "checkpoints" / "best.pt",
        ROOT / "checkpoints" / "epoch_180.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c

    scored: list[tuple[int, Path]] = []
    for p in (ROOT / "checkpoints").glob("epoch_*.pt"):
        stem = p.stem.replace("epoch_", "").split()[0]
        try:
            scored.append((int(stem), p))
        except ValueError:
            continue
    if scored:
        return max(scored, key=lambda x: x[0])[1]

    raise FileNotFoundError(
        "No checkpoint found.\n"
        "Download weights:\n"
        "  huggingface-cli download Emma5099/dance2music pytorch_model.pt "
        "--local-dir checkpoints"
    )


def _trim_waveform(y: np.ndarray, max_seconds: float) -> np.ndarray:
    return y[: int(round(max_seconds * AUDIO_SR))]


def _ffmpeg_to_gif(src: Path, dst: Path, *, width: int = 480, fps: int = 12) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"fps={fps},scale={width}:-1:flags=lanczos,"
        "split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=5"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-loop", "0", str(dst)],
        check=True,
        capture_output=True,
    )


def _export_formats(mp4_path: Path, formats: set[str]) -> list[Path]:
    """Ensure requested formats exist; return list of written paths."""
    written: list[Path] = []
    if "mp4" in formats:
        written.append(mp4_path)
    if "gif" in formats:
        gif_path = mp4_path.with_suffix(".gif")
        print(f"  Converting → {gif_path.name}")
        _ffmpeg_to_gif(mp4_path, gif_path)
        written.append(gif_path)
        if "mp4" not in formats:
            mp4_path.unlink(missing_ok=True)
            written = [p for p in written if p.suffix != ".mp4"]
    return written


def generate(
    audio_path: Path,
    *,
    mode: str = "both",
    formats: set[str] | None = None,
    out_dir: Path | None = None,
    checkpoint: Path | None = None,
    mesh: Path = DEFAULT_MESH,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    source_seconds: float = DEFAULT_SOURCE_SECONDS,
    smooth: float = 1.25,
    upsample: int = 2,
    vert_ema: float = 0.85,
) -> list[Path]:
    formats = formats or {"mp4"}
    if max_seconds <= 0:
        raise ValueError("--max-seconds must be > 0")
    if max_seconds > 20.0 + 1e-6:
        print(f"  Clamping --max-seconds {max_seconds} → 20.0")
        max_seconds = 20.0

    out_dir = out_dir or (ROOT / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    work = out_dir / f".{stem}_work"
    work.mkdir(parents=True, exist_ok=True)

    want_skeleton = mode in ("skeleton", "both")
    want_character = mode in ("character", "both")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = checkpoint or _resolve_checkpoint(None)
    print(f"\nDevice: {device}")
    print(f"Checkpoint: {ckpt_path}")

    # ── 1. Audio ────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading audio: {audio_path}")
    waveform = load_audio(str(audio_path))
    full_dur = len(waveform) / AUDIO_SR
    src_cap = min(full_dur, max(source_seconds, max_seconds))
    waveform = _trim_waveform(waveform, src_cap)
    print(f"  Source used: {len(waveform) / AUDIO_SR:.1f}s of {full_dur:.1f}s")

    # ── 2. Inference ────────────────────────────────────────────────────────
    print("\n[2/5] Audio features + pose generation…")
    audio_feat = audio_to_features(waveform)
    model, ckpt = load_model(str(ckpt_path), device)
    poses = run_inference(
        model,
        audio_feat,
        ckpt["x_mean"],
        ckpt["x_std"],
        ckpt["y_mean"],
        ckpt["y_std"],
        device,
    )
    print(f"  Raw poses: {poses.shape[0]} frames ({poses.shape[0] / POSE_FPS:.1f}s)")

    # ── 3. Clean ────────────────────────────────────────────────────────────
    print("\n[3/5] Cleaning dance (drop floor collapses)…")
    cleaned = clean_dance_poses(poses, fps=POSE_FPS)
    print(
        f"  Kept {len(cleaned.segments)} segments, "
        f"cut {cleaned.n_cut} frames → {cleaned.poses.shape[0]} "
        f"({cleaned.poses.shape[0] / POSE_FPS:.1f}s)"
    )

    clean_audio = cut_audio_to_segments(
        waveform[:, None] if waveform.ndim == 1 else waveform,
        AUDIO_SR,
        cleaned.segments,
        pose_fps=POSE_FPS,
        target_frames=len(cleaned.poses),
    )
    clean_audio_out = clean_audio[:, 0] if clean_audio.ndim == 2 else clean_audio

    # ── 4. Cap duration ─────────────────────────────────────────────────────
    print(f"\n[4/5] Truncating to ≤ {max_seconds:.0f}s…")
    poses_final = truncate_poses(cleaned.poses, max_seconds, fps=POSE_FPS)
    n_samples = int(round(len(poses_final) / POSE_FPS * AUDIO_SR))
    audio_final = np.asarray(clean_audio_out[:n_samples])
    dur = len(poses_final) / POSE_FPS
    print(f"  Final: {len(poses_final)} frames ({dur:.2f}s)")

    poses_path = work / f"{stem}_poses_clean.npy"
    np.save(poses_path, poses_final.astype(np.float32))
    audio_path_final = work / f"{stem}_clean.wav"
    sf.write(audio_path_final, audio_final, AUDIO_SR)

    results: list[Path] = []

    # ── Skeleton ────────────────────────────────────────────────────────────
    if want_skeleton:
        skel_mp4 = out_dir / f"{stem}_skeleton.mp4"
        tmp = work / "skeleton_silent.mp4"
        print(f"\n[5a] Rendering skeleton → {skel_mp4.name}")
        render_skeleton_video(poses_final, title=stem, tmp_path=str(tmp))
        mux_audio(str(tmp), str(audio_path_final), str(skel_mp4), dur)
        tmp.unlink(missing_ok=True)
        results.extend(_export_formats(skel_mp4, formats))

    # ── Character ───────────────────────────────────────────────────────────
    if want_character:
        if not mesh.is_file():
            raise FileNotFoundError(f"Character mesh not found: {mesh}")
        if not ANIMATE_SCRIPT.is_file():
            raise FileNotFoundError(f"Missing animator: {ANIMATE_SCRIPT}")

        char_mp4 = out_dir / f"{stem}_character.mp4"
        print(f"\n[5b] Character animation → {char_mp4.name}")
        subprocess.run(
            [
                sys.executable,
                str(ANIMATE_SCRIPT),
                "--poses", str(poses_path),
                "--mesh", str(mesh),
                "--audio", str(audio_path_final),
                "--out", str(char_mp4),
                "--smooth", str(smooth),
                "--upsample", str(upsample),
                "--vert-ema", str(vert_ema),
            ],
            check=True,
        )
        results.extend(_export_formats(char_mp4, formats))

    print("\nDone!")
    for p in results:
        print(f"  → {p}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate a dancing skeleton and/or 3D character from music",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--audio", type=Path, required=True, help="Input music file")
    p.add_argument(
        "--mode",
        choices=("skeleton", "character", "both"),
        default="both",
        help="What to render",
    )
    p.add_argument(
        "--format",
        choices=("mp4", "gif", "both"),
        default="mp4",
        help="Output format(s)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: ./outputs)",
    )
    p.add_argument("--checkpoint", type=Path, default=None, help="Model .pt weights")
    p.add_argument("--mesh", type=Path, default=DEFAULT_MESH, help="Character GLB")
    p.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help="Hard cap on output duration (clamped to ≤20)",
    )
    p.add_argument(
        "--source-seconds",
        type=float,
        default=DEFAULT_SOURCE_SECONDS,
        help="How much audio to feed the model before cleaning",
    )
    p.add_argument("--smooth", type=float, default=1.25)
    p.add_argument("--upsample", type=int, default=2)
    p.add_argument("--vert-ema", type=float, default=0.85)
    args = p.parse_args()

    if not args.audio.is_file():
        raise SystemExit(f"Audio not found: {args.audio}")

    fmt = {"mp4", "gif"} if args.format == "both" else {args.format}
    generate(
        args.audio,
        mode=args.mode,
        formats=fmt,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        mesh=args.mesh,
        max_seconds=args.max_seconds,
        source_seconds=args.source_seconds,
        smooth=args.smooth,
        upsample=args.upsample,
        vert_ema=args.vert_ema,
    )


if __name__ == "__main__":
    main()
