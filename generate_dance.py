#!/usr/bin/env python3
"""
generate_dance.py — End-to-end music → clean character dance (CLI).

Pipeline:
  1. Load audio (cap source length for inference)
  2. Run Music2PoseTransformer → poses
  3. Remove on-ground collapses, stitch + smooth
  4. Hard-cap output to --max-seconds (default 20)
  5. Render character GLB animation with muxed audio

Usage:
  python generate_dance.py --audio Freak.wav
  python generate_dance.py --audio song.mp3 --out outputs/my_dance.mp4 --max-seconds 20
  python generate_dance.py --audio MA.wav --skeleton   # also write skeleton preview
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
from pose_postprocess import (
    clean_dance_poses,
    cut_audio_to_segments,
    truncate_poses,
)
from train import BASELINE_CKPT

DEFAULT_MESH = Path("3Dmodel/character.glb")
DEFAULT_MAX_SECONDS = 20.0
# Generate extra source so cleaning still leaves ~max_seconds of usable dance.
DEFAULT_SOURCE_SECONDS = 40.0


def _resolve_checkpoint(path: str | None) -> str:
    if path and Path(path).is_file():
        return path
    candidates = [
        BASELINE_CKPT,
        "checkpoints/epoch_180.pt",
        "checkpoints/best.pt",
        "checkpoints/pytorch_model.pt",
        "checkpoints/epoch_185.pt",
        "checkpoints/epoch_175.pt",
    ]
    for c in candidates:
        if Path(c).is_file():
            return c
    # newest epoch_*.pt
    ckpts = sorted(Path("checkpoints").glob("epoch_*.pt"))
    # prefer numeric, skip weird names
    scored = []
    for p in ckpts:
        stem = p.stem.replace("epoch_", "").split()[0]
        try:
            scored.append((int(stem), p))
        except ValueError:
            continue
    if scored:
        return str(max(scored, key=lambda x: x[0])[1])
    raise FileNotFoundError(
        "No checkpoint found. Place weights in checkpoints/ or pass --checkpoint."
    )


def _trim_waveform(y: np.ndarray, max_seconds: float) -> np.ndarray:
    n = int(round(max_seconds * AUDIO_SR))
    return y[:n]


def run_pipeline(
    audio_path: Path,
    out_path: Path,
    *,
    checkpoint: str,
    mesh: Path,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    source_seconds: float = DEFAULT_SOURCE_SECONDS,
    skeleton_out: Path | None = None,
    save_poses: Path | None = None,
    skip_character: bool = False,
    smooth: float = 1.25,
    upsample: int = 2,
    vert_ema: float = 0.85,
) -> Path:
    if max_seconds <= 0:
        raise ValueError("--max-seconds must be > 0")
    if max_seconds > 20.0 + 1e-6:
        # Hard product rule: never longer than 20s.
        print(f"  Clamping --max-seconds {max_seconds} → 20.0")
        max_seconds = 20.0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    work = out_path.parent / f".{out_path.stem}_work"
    work.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── 1. Audio ────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading audio: {audio_path}")
    waveform = load_audio(str(audio_path))
    full_dur = len(waveform) / AUDIO_SR
    # Pull enough source to survive cleaning, but never need more than a few
    # minutes for a ≤20s output.
    src_cap = min(full_dur, max(source_seconds, max_seconds))
    waveform = _trim_waveform(waveform, src_cap)
    print(f"  Source used: {len(waveform)/AUDIO_SR:.1f}s of {full_dur:.1f}s")

    # ── 2. Features + inference ─────────────────────────────────────────────
    print("\n[2/5] Audio features + pose generation…")
    audio_feat = audio_to_features(waveform)
    print(f"  Features: {audio_feat.shape}")

    print(f"  Checkpoint: {checkpoint}")
    model, ckpt = load_model(checkpoint, device)
    poses = run_inference(
        model,
        audio_feat,
        ckpt["x_mean"],
        ckpt["x_std"],
        ckpt["y_mean"],
        ckpt["y_std"],
        device,
    )
    print(f"  Raw poses: {poses.shape[0]} frames ({poses.shape[0]/POSE_FPS:.1f}s)")

    # ── 3. Clean ────────────────────────────────────────────────────────────
    print("\n[3/5] Cleaning dance (drop floor collapses)…")
    cleaned = clean_dance_poses(poses, fps=POSE_FPS)
    print(
        f"  Kept {len(cleaned.segments)} segments, "
        f"cut {cleaned.n_cut} frames → {cleaned.poses.shape[0]} "
        f"({cleaned.poses.shape[0]/POSE_FPS:.1f}s)"
    )

    # Audio aligned to cleaned pose timeline (before truncate)
    clean_audio = cut_audio_to_segments(
        waveform[:, None] if waveform.ndim == 1 else waveform,
        AUDIO_SR,
        cleaned.segments,
        pose_fps=POSE_FPS,
        target_frames=len(cleaned.poses),
    )
    if clean_audio.ndim == 2 and clean_audio.shape[1] == 1:
        clean_audio_out = clean_audio[:, 0]
    else:
        clean_audio_out = clean_audio

    # ── 4. Cap ≤ max_seconds ────────────────────────────────────────────────
    print(f"\n[4/5] Truncating to ≤ {max_seconds:.0f}s…")
    poses_final = truncate_poses(cleaned.poses, max_seconds, fps=POSE_FPS)
    n_samples = int(round(len(poses_final) / POSE_FPS * AUDIO_SR))
    audio_final = np.asarray(clean_audio_out[:n_samples])
    dur = len(poses_final) / POSE_FPS
    print(f"  Final: {len(poses_final)} frames ({dur:.2f}s)")

    poses_path = save_poses or (work / f"{stem}_poses_clean.npy")
    poses_path = Path(poses_path)
    poses_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(poses_path, poses_final.astype(np.float32))
    print(f"  Poses → {poses_path}")

    audio_path_final = work / f"{stem}_clean.wav"
    sf.write(audio_path_final, audio_final, AUDIO_SR)

    # ── Optional skeleton preview ───────────────────────────────────────────
    if skeleton_out is not None:
        skeleton_out = Path(skeleton_out)
        skeleton_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = work / "skeleton_silent.mp4"
        print(f"\n  Rendering skeleton → {skeleton_out}")
        render_skeleton_video(poses_final, title=stem, tmp_path=str(tmp))
        mux_audio(str(tmp), str(audio_path_final), str(skeleton_out), dur)
        tmp.unlink(missing_ok=True)

    if skip_character:
        print(f"\nDone (poses only) → {poses_path}")
        return poses_path

    if not mesh.is_file():
        raise FileNotFoundError(f"Character mesh not found: {mesh}")

    # ── 5. Character render ─────────────────────────────────────────────────
    print(f"\n[5/5] Character animation → {out_path}")
    cmd = [
        sys.executable,
        "3Dmodel/animate_character.py",
        "--poses", str(poses_path),
        "--mesh", str(mesh),
        "--audio", str(audio_path_final),
        "--out", str(out_path),
        "--smooth", str(smooth),
        "--upsample", str(upsample),
        "--vert-ema", str(vert_ema),
    ]
    subprocess.run(cmd, check=True)

    # Safety: if somehow longer, re-trim with ffmpeg
    try:
        import json
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "json", str(out_path),
            ],
            capture_output=True, text=True, check=True,
        )
        out_dur = float(json.loads(probe.stdout)["format"]["duration"])
        if out_dur > max_seconds + 0.15:
            print(f"  Trimming output {out_dur:.2f}s → {max_seconds:.2f}s")
            trimmed = out_path.with_suffix(".trim.mp4")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(out_path),
                    "-t", str(max_seconds),
                    "-c", "copy", str(trimmed),
                ],
                check=True, capture_output=True,
            )
            trimmed.replace(out_path)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] duration check skipped: {exc}")

    print(f"\nDone! → {out_path}  ({dur:.2f}s)")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Music → clean character dance (max 20s)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--audio", type=Path, required=True, help="Input music file")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output character MP4 (default: outputs/<stem>_character.mp4)",
    )
    p.add_argument("--checkpoint", default=None, help="Model .pt weights")
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
    p.add_argument(
        "--skeleton",
        action="store_true",
        help="Also write a cleaned skeleton preview MP4",
    )
    p.add_argument(
        "--skeleton-out",
        type=Path,
        default=None,
        help="Skeleton preview path (implies --skeleton)",
    )
    p.add_argument("--save-poses", type=Path, default=None, help="Save cleaned poses .npy")
    p.add_argument("--skip-character", action="store_true", help="Stop after poses/skeleton")
    p.add_argument("--smooth", type=float, default=1.25)
    p.add_argument("--upsample", type=int, default=2)
    p.add_argument("--vert-ema", type=float, default=0.85)
    args = p.parse_args()

    if not args.audio.is_file():
        raise SystemExit(f"Audio not found: {args.audio}")

    out = args.out or Path("outputs") / f"{args.audio.stem}_character.mp4"
    skel = args.skeleton_out
    if args.skeleton and skel is None:
        skel = out.with_name(out.stem.replace("_character", "") + "_skeleton.mp4")
        if skel == out:
            skel = out.with_name(out.stem + "_skeleton.mp4")

    ckpt = _resolve_checkpoint(args.checkpoint)
    run_pipeline(
        args.audio,
        out,
        checkpoint=ckpt,
        mesh=args.mesh,
        max_seconds=args.max_seconds,
        source_seconds=args.source_seconds,
        skeleton_out=skel,
        save_poses=args.save_poses,
        skip_character=args.skip_character,
        smooth=args.smooth,
        upsample=args.upsample,
        vert_ema=args.vert_ema,
    )


if __name__ == "__main__":
    main()
