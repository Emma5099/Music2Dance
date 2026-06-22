"""
data/extract_poses.py

Extract body pose coordinates from dance videos using MediaPipe Pose.
Outputs: per-video .npy files with shape (T, 33, 9)
         where T = number of frames, 33 = body landmarks (no hands), 9 = xyz + Δxyz + Δ²xyz
"""

import cv2
import numpy as np
import json
import urllib.request
from pathlib import Path
from tqdm import tqdm
import argparse

import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

# BlazePose body model: 33 landmarks (no extra hand/face points in this pipeline).
N_LANDMARKS = 33

# Official lite model (downloaded once into ~/.cache/dance-music/ unless --model_path is set).
POSE_LANDMARKER_LITE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def _ensure_default_pose_model() -> Path:
    cache = Path.home() / ".cache" / "dance-music"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "pose_landmarker_lite.task"
    if not path.exists():
        print(f"Downloading pose model to {path} ...")
        urllib.request.urlretrieve(POSE_LANDMARKER_LITE_URL, path)
    return path


def extract_landmarks_frame(results) -> np.ndarray:
    """
    Extract body landmarks from a MediaPipe Pose result frame.
    Returns array of shape (33, 3) — (x, y, z) normalized coords.
    Missing landmarks are filled with zeros.
    """
    frame = np.zeros((N_LANDMARKS, 3), dtype=np.float32)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks[0]
        for i, lm in enumerate(landmarks):
            if i >= N_LANDMARKS:
                break
            frame[i] = [lm.x, lm.y, lm.z]

    return frame


def normalize_pose_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Normalize pose sequence to be translation/scale invariant:
    - Center on hip midpoint (hip-centered)
    - Scale by torso height (shoulder–hip distance)

    Args:
        seq: (T, 33, 3) raw landmark coords

    Returns:
        (T, 33, 3) normalized coords
    """
    # Hip midpoint — MediaPipe Pose indices: left_hip=23, right_hip=24
    left_hip, right_hip = 23, 24
    hip = (seq[:, left_hip:left_hip + 1] + seq[:, right_hip:right_hip + 1]) / 2.0  # (T, 1, 3)

    # Center all landmarks on the hip midpoint
    seq_centered = seq - hip

    # Scale factor: distance from hip midpoint to shoulder midpoint
    left_shoulder, right_shoulder = 11, 12
    shoulder = (seq[:, left_shoulder:left_shoulder + 1] + seq[:, right_shoulder:right_shoulder + 1]) / 2.0
    torso_height = np.linalg.norm(shoulder - hip, axis=-1, keepdims=True)  # (T, 1, 1)
    torso_height = np.maximum(torso_height, 1e-6)  # guard against div-by-zero

    seq_normalized = seq_centered / torso_height
    return seq_normalized


def compute_motion_features(seq: np.ndarray) -> np.ndarray:
    """
    Augment raw coordinates with first- and second-order finite differences.

    Args:
        seq: (T, 33, 3) normalized landmark positions

    Returns:
        (T, 33, 9): [xyz | Δxyz (velocity) | Δ²xyz (acceleration)]
    """
    velocity = np.zeros_like(seq)
    velocity[1:] = seq[1:] - seq[:-1]          # forward difference

    acceleration = np.zeros_like(seq)
    acceleration[1:] = velocity[1:] - velocity[:-1]

    return np.concatenate([seq, velocity, acceleration], axis=-1)  # (T, 33, 9)


def extract_video_poses(
    video_path: str,
    target_fps: float = 25.0,
    add_motion_features: bool = True,
    model_path: str | None = None,
) -> dict:
    """
    Extract pose sequence from a single video file.

    Args:
        video_path:          Path to input video
        target_fps:          Target sample rate in fps (source is resampled to this)
        add_motion_features: Append velocity + acceleration channels

    Returns:
        dict with keys:
            'poses'             : np.ndarray (T, 33, 3 or 9)
            'fps'               : float
            'n_frames'          : int
            'video_duration_s'  : float
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_skip = max(1, round(source_fps / target_fps))

    model = Path(model_path) if model_path else _ensure_default_pose_model()
    if not model.is_file():
        raise FileNotFoundError(f"Pose model not found: {model}")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames_data = []
    frame_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(round(1000.0 * frame_idx / source_fps))
                results = landmarker.detect_for_video(mp_image, timestamp_ms)
                lm = extract_landmarks_frame(results)
                frames_data.append(lm)

            frame_idx += 1

    cap.release()

    if not frames_data:
        raise ValueError(f"No frames extracted from {video_path}")

    seq = np.stack(frames_data, axis=0)   # (T, 33, 3)
    seq = normalize_pose_sequence(seq)

    if add_motion_features:
        seq = compute_motion_features(seq)  # (T, 33, 9)

    return {
        "poses": seq,
        "fps": target_fps,
        "n_frames": len(seq),
        "video_duration_s": len(seq) / target_fps,
    }


def extract_dataset(
    video_dir: str,
    output_dir: str,
    target_fps: float = 25.0,
    model_path: str | None = None,
    extensions: tuple = (".mp4", ".avi", ".mov", ".mkv"),
):
    """
    Batch-extract poses from all videos in a directory.

    Expected directory layout:
        video_dir/
            dance_001.mp4
            dance_002.mp4
            ...

    Output:
        output_dir/
            dance_001_poses.npy      # (T, 33, 9)
            dance_001_meta.json      # fps, duration, n_frames
            ...
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(f for f in video_dir.iterdir() if f.suffix.lower() in extensions)
    print(f"Found {len(videos)} videos in {video_dir}")

    succeeded, failed = 0, 0
    for video_path in tqdm(videos, desc="Extracting poses"):
        stem = video_path.stem
        pose_path = output_dir / f"{stem}_poses.npy"
        meta_path = output_dir / f"{stem}_meta.json"

        if pose_path.exists():
            succeeded += 1
            continue  # already processed

        try:
            result = extract_video_poses(
                str(video_path), target_fps=target_fps, model_path=model_path
            )
            np.save(pose_path, result["poses"])
            meta = {k: v for k, v in result.items() if k != "poses"}
            meta_path.write_text(json.dumps(meta, indent=2))
            succeeded += 1
        except Exception as e:
            print(f"  ERROR on {video_path.name}: {e}")
            failed += 1

    print(f"Done. {succeeded} succeeded, {failed} failed. Poses saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract body pose sequences from dance videos.")
    parser.add_argument("--video_dir",  required=True, help="Directory containing video files")
    parser.add_argument("--output_dir", required=True, help="Where to write .npy + .json files")
    parser.add_argument("--fps", type=float, default=25.0, help="Target extraction frame rate")
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to pose_landmarker *.task file (default: download lite model to ~/.cache/dance-music/)",
    )
    args = parser.parse_args()

    extract_dataset(
        args.video_dir,
        args.output_dir,
        target_fps=args.fps,
        model_path=args.model_path,
    )
