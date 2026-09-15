"""Post-processing helpers for predicted MediaPipe-style skeletons."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Ankles, heels, foot indices (MediaPipe Pose)
_FOOT_JOINTS = (27, 28, 29, 30, 31, 32)
_HIP_L, _HIP_R = 23, 24
_SH_L, _SH_R = 11, 12
_ANK_L, _ANK_R = 27, 28


def contact_y(poses_xyz: np.ndarray) -> np.ndarray:
    """Per-frame Y of the lowest foot joint (MediaPipe Y increases downward)."""
    return poses_xyz[:, _FOOT_JOINTS, 1].max(axis=1)


def ground_poses(
    poses_xyz: np.ndarray,
    floor: float | None = None,
    smooth: float = 0.0,
) -> np.ndarray:
    """
    Translate each frame in Y so the supporting foot sits on a shared floor.

    Hip-centered training leaves absolute height free; crouches then look like
    sitting in mid-air. Planting the lowest foot restores a stable ground plane
    without changing relative limb geometry.

    Args:
        poses_xyz: (T, 33, 3) xyz joints
        floor: target contact Y (default: median contact over the clip)
        smooth: EMA factor in [0, 1) on the Y shift (0 = hard plant every frame)

    Returns:
        (T, 33, 3) copy with feet planted
    """
    out = np.array(poses_xyz, dtype=np.float32, copy=True)
    contact = contact_y(out)
    if floor is None:
        floor = float(np.median(contact))

    shift = floor - contact
    if smooth > 0.0:
        s = float(np.clip(smooth, 0.0, 0.999))
        filt = np.empty_like(shift)
        filt[0] = shift[0]
        for t in range(1, len(shift)):
            filt[t] = s * filt[t - 1] + (1.0 - s) * shift[t]
        shift = filt

    out[:, :, 1] += shift[:, None]
    return out


def _hip_center(poses: np.ndarray) -> np.ndarray:
    return 0.5 * (poses[:, _HIP_L] + poses[:, _HIP_R])


def on_ground_mask(
    poses_xyz: np.ndarray,
    *,
    hip_clearance_min: float = 0.35,
    upright_min: float = 0.40,
    close_iters: int = 8,
    open_iters: int = 3,
    pad_iters: int = 5,
) -> np.ndarray:
    """
    Boolean mask (T,) True where the dancer has collapsed / is on the floor.

    MediaPipe Y increases downward. Upright torso → hip→shoulder has negative Y.
    """
    from scipy.ndimage import binary_closing, binary_dilation, binary_opening

    poses = np.asarray(poses_xyz, dtype=np.float32)
    hip = 0.5 * (poses[:, _HIP_L] + poses[:, _HIP_R])
    shoulder = 0.5 * (poses[:, _SH_L] + poses[:, _SH_R])
    ankle = 0.5 * (poses[:, _ANK_L] + poses[:, _ANK_R])

    torso = shoulder - hip
    upright = -torso[:, 1] / (np.linalg.norm(torso, axis=1) + 1e-6)
    floor = float(np.median(ankle[:, 1]))
    hip_above_floor = floor - hip[:, 1]

    raw = (hip_above_floor < hip_clearance_min) | (upright < upright_min)
    mask = binary_closing(raw, iterations=close_iters)
    mask = binary_opening(mask, iterations=open_iters)
    mask = binary_dilation(mask, iterations=pad_iters)
    return np.asarray(mask, dtype=bool)


@dataclass
class CleanResult:
    poses: np.ndarray          # (T', 33, 3) cleaned
    segments: list[tuple[int, int]]  # kept [start, end) in original frame index
    n_cut: int


def clean_dance_poses(
    poses_xyz: np.ndarray,
    *,
    fps: float = 25.0,
    min_keep_s: float = 1.8,
    blend_frames: int = 14,
    ground_smooth: float = 0.18,
    gauss_sigma: float = 1.25,
    center_xz: bool = True,
) -> CleanResult:
    """
    Remove on-ground collapses, stitch remaining segments with ease blends,
    plant feet, and lightly smooth for continuous motion.
    """
    from scipy.ndimage import gaussian_filter1d

    poses = np.asarray(poses_xyz, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (33, 3):
        raise ValueError(f"Expected (T,33,3) poses, got {poses.shape}")

    T = len(poses)
    mask = on_ground_mask(poses)
    keep = ~mask
    min_keep = max(8, int(round(min_keep_s * fps)))

    segs: list[tuple[int, int]] = []
    i = 0
    while i < T:
        if keep[i]:
            j = i
            while j < T and keep[j]:
                j += 1
            if j - i >= min_keep:
                segs.append((i, j))
            i = j
        else:
            i += 1

    if not segs:
        # Fallback: keep the most upright contiguous window
        hip = 0.5 * (poses[:, _HIP_L] + poses[:, _HIP_R])
        shoulder = 0.5 * (poses[:, _SH_L] + poses[:, _SH_R])
        torso = shoulder - hip
        upright = -torso[:, 1] / (np.linalg.norm(torso, axis=1) + 1e-6)
        win = min(T, max(min_keep, int(round(5 * fps))))
        best_i = int(np.argmax([upright[k : k + win].mean() for k in range(0, max(1, T - win + 1))]))
        segs = [(best_i, min(T, best_i + win))]

    parts = [poses[a:b].copy() for a, b in segs]
    stitched = [parts[0]]
    for seg in parts[1:]:
        prev = stitched[-1]
        d = _hip_center(prev)[-1, [0, 2]] - _hip_center(seg)[0, [0, 2]]
        seg = seg.copy()
        seg[:, :, 0] += d[0]
        seg[:, :, 2] += d[1]

        n = min(blend_frames, len(prev) // 3, len(seg) // 3, 24)
        if n >= 2:
            for k in range(n):
                a = (k + 1) / (n + 1)
                w = 0.5 - 0.5 * np.cos(np.pi * a)
                target = (1.0 - w) * prev[-n + k] + w * seg[k]
                prev[-n + k] = target
                seg[k] = target
        stitched.append(seg)

    clean = np.concatenate(stitched, axis=0)
    clean = ground_poses(clean, smooth=ground_smooth)
    if gauss_sigma > 0:
        for c in range(3):
            clean[:, :, c] = gaussian_filter1d(
                clean[:, :, c], sigma=gauss_sigma, axis=0
            ).astype(np.float32)

    if center_xz:
        hc = _hip_center(clean)
        clean[:, :, 0] -= hc[:, 0:1]
        clean[:, :, 2] -= hc[:, 2:3]

    return CleanResult(
        poses=clean.astype(np.float32),
        segments=segs,
        n_cut=int(mask.sum()),
    )


def cut_audio_to_segments(
    audio: np.ndarray,
    sr: int,
    segments: list[tuple[int, int]],
    *,
    pose_fps: float = 25.0,
    crossfade_s: float = 0.05,
    target_frames: int | None = None,
) -> np.ndarray:
    """Concatenate audio slices matching kept pose segments; optional length match."""
    if audio.ndim == 1:
        audio = audio[:, None]

    chunks: list[np.ndarray] = []
    for a, b in segments:
        s0 = int(round(a / pose_fps * sr))
        s1 = int(round(b / pose_fps * sr))
        s0 = max(0, min(s0, len(audio)))
        s1 = max(0, min(s1, len(audio)))
        if s1 > s0:
            chunks.append(audio[s0:s1])

    if not chunks:
        n = int(round((target_frames or 1) / pose_fps * sr))
        return np.zeros((n, audio.shape[1]), dtype=audio.dtype)

    xf = max(1, int(round(crossfade_s * sr)))
    out = chunks[0].copy()
    for ch in chunks[1:]:
        n = min(xf, len(out), len(ch))
        if n > 1:
            fade = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
            mixed = out[-n:] * (1.0 - fade) + ch[:n] * fade
            out = np.concatenate([out[:-n], mixed, ch[n:]], axis=0)
        else:
            out = np.concatenate([out, ch], axis=0)

    if target_frames is not None:
        target = int(round(target_frames / pose_fps * sr))
        if len(out) > target:
            out = out[:target]
        elif len(out) < target:
            pad = np.zeros((target - len(out), out.shape[1]), dtype=out.dtype)
            out = np.concatenate([out, pad], axis=0)

    return out


def truncate_poses(
    poses_xyz: np.ndarray,
    max_seconds: float,
    *,
    fps: float = 25.0,
) -> np.ndarray:
    """Hard-cap pose sequence length (never longer than max_seconds)."""
    n = max(1, int(round(max_seconds * fps)))
    return np.asarray(poses_xyz[:n], dtype=np.float32)
