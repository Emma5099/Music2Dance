"""
data/build_dataset.py

Build a HuggingFace Dataset of (pose_sequence, audio_chunk) pairs.

For each video we:
  1. Load the extracted pose .npy file
  2. Extract audio from the video (or load pre-extracted .wav)
  3. Chunk both into CHUNK_DURATION_S windows with STRIDE_S overlap
  4. Store as a datasets.Dataset on disk
"""

import numpy as np
import soundfile as sf
import librosa
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from tqdm import tqdm
import argparse

CHUNK_DURATION_S = 10.0    # seconds per training window
STRIDE_S = 5.0             # stride between windows (50% overlap)
SAMPLE_RATE = 32_000       # EnCodec / MusicGen sample rate
POSE_FPS = 25.0


def extract_audio_from_video(video_path: Path, output_dir: Path) -> Path:
    """Extract audio track from video using ffmpeg."""
    audio_path = output_dir / f"{video_path.stem}.wav"
    if audio_path.exists():
        return audio_path
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        str(audio_path)
    ], check=True, capture_output=True)
    return audio_path


def load_mono_waveform(path: Path, sample_rate: int) -> np.ndarray:
    """
    Load WAV as mono float32 with shape (1, n_samples) at ``sample_rate``.

    Uses soundfile + librosa instead of torchaudio so we never hit PyTorch's
    tensor→numpy path (which can raise RuntimeError: Numpy is not available on
    some torch builds or backends).
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # (n_samples, n_channels)
    if data.shape[1] > 1:
        mono = np.mean(data, axis=1)
    else:
        mono = data[:, 0]
    if sr != sample_rate:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=sample_rate)
    return mono[np.newaxis, :].astype(np.float32)


def chunk_sequence(
    poses: np.ndarray,
    audio: np.ndarray,
    pose_fps: float = POSE_FPS,
    chunk_s: float = CHUNK_DURATION_S,
    stride_s: float = STRIDE_S,
    sample_rate: int = SAMPLE_RATE,
) -> list[dict]:
    """
    Slice (poses, audio) into overlapping chunks.

    poses: (T_pose, N_landmarks, C)
    audio: (1, T_audio)  — mono waveform, float32 numpy

    Returns list of dicts with:
        - pose_chunk: (T_chunk_pose, N_landmarks, C) as float32
        - audio_chunk: (1, T_chunk_audio) as float32
    """
    pose_chunk_len = int(chunk_s * pose_fps)
    pose_stride = int(stride_s * pose_fps)
    audio_chunk_len = int(chunk_s * sample_rate)
    audio_stride = int(stride_s * sample_rate)

    T_pose = poses.shape[0]
    chunks = []
    pose_start = 0

    while pose_start + pose_chunk_len <= T_pose:
        audio_start = int(pose_start / pose_fps * sample_rate)
        audio_end = audio_start + audio_chunk_len

        if audio_end > audio.shape[-1]:
            break

        chunks.append({
            "pose_chunk": poses[pose_start:pose_start + pose_chunk_len].astype(np.float32),
            "audio_chunk": audio[:, audio_start:audio_end].astype(np.float32),
        })

        pose_start += pose_stride

    return chunks


def _load_existing_stems(output_dir: Path) -> tuple[Optional[DatasetDict], set[str]]:
    """If a DatasetDict already exists on disk, return it and the set of video stems it contains."""
    marker = output_dir / "dataset_dict.json"
    if not marker.is_file():
        return None, set()
    dd = load_from_disk(str(output_dir))
    stems = set(dd["train"]["stem"]) | set(dd["validation"]["stem"])
    return dd, stems


def _staged_stems(staging_dir: Path) -> set[str]:
    """Stems already written to on-disk staging shards (for resume after OOM)."""
    stems: set[str] = set()
    if not staging_dir.is_dir():
        return stems
    for shard in staging_dir.glob("batch_*"):
        if not _shard_is_valid(shard):
            continue
        ds = load_from_disk(str(shard))
        stems.update(ds["stem"])
    return stems


def _shard_is_valid(shard_path: Path) -> bool:
    return shard_path.is_dir() and any(shard_path.glob("*.arrow"))


def _save_staging_shard(staging_dir: Path, shard_idx: int, records: list[dict]) -> None:
    if not records:
        return
    staging_dir.mkdir(parents=True, exist_ok=True)
    shard_path = staging_dir / f"batch_{shard_idx:04d}"
    tmp_path = staging_dir / f"_tmp_{shard_idx:04d}"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    Dataset.from_list(records).save_to_disk(str(tmp_path))
    if shard_path.exists():
        shutil.rmtree(shard_path)
    tmp_path.rename(shard_path)


def _merge_staging_shards(
    staging_dir: Path,
    existing_dd: Optional[DatasetDict],
    output_dir: Path,
    val_split: float,
    seed: int,
) -> Optional[DatasetDict]:
    """Concatenate staging shards (+ optional existing dataset) without holding all chunks in RAM."""
    shard_paths = sorted(p for p in staging_dir.glob("batch_*") if _shard_is_valid(p))
    if not shard_paths and existing_dd is None:
        return None

    buf_a = output_dir / "_merge_a"
    buf_b = output_dir / "_merge_b"
    for p in (buf_a, buf_b):
        if p.is_dir():
            shutil.rmtree(p)

    combined = None
    if existing_dd is not None:
        combined = concatenate_datasets([existing_dd["train"], existing_dd["validation"]])

    write_buf = buf_a
    for shard_path in shard_paths:
        try:
            shard = load_from_disk(str(shard_path))
        except Exception as e:
            print(f"  Warning: skipping corrupt shard {shard_path.name}: {e}")
            continue
        combined = shard if combined is None else concatenate_datasets([combined, shard])
        del shard
        if write_buf.is_dir():
            shutil.rmtree(write_buf)
        combined.save_to_disk(str(write_buf))
        del combined
        combined = load_from_disk(str(write_buf))
        write_buf = buf_b if write_buf is buf_a else buf_a

    combined = combined.shuffle(seed=seed)
    n_val = max(1, int(len(combined) * val_split))
    split = combined.train_test_split(test_size=n_val, seed=seed)

    dataset_dict = DatasetDict({"train": split["train"], "validation": split["test"]})
    dataset_dict.save_to_disk(str(output_dir))

    for p in (buf_a, buf_b):
        if p.is_dir():
            shutil.rmtree(p)
    shutil.rmtree(staging_dir)

    return dataset_dict


def build_dataset(
    video_dir: str,
    pose_dir: str,
    output_dir: str,
    audio_dir: Optional[str] = None,
    val_split: float = 0.05,
    seed: int = 42,
    rebuild: bool = False,
    batch_size: int = 25,
):
    """
    Build a HuggingFace Dataset from extracted poses + video audio.

    Args:
        video_dir:  Directory containing original .mp4 files
        pose_dir:   Directory containing _poses.npy files (from extract_poses.py)
        output_dir: Where to save the HuggingFace dataset
        audio_dir:  Optional pre-extracted audio dir. If None, extracts on the fly.
        val_split:  Fraction for validation set
        rebuild:    If False and ``output_dir`` already has a dataset, only add chunks from
                    videos whose stem is not yet present (skip re-chunking / re-audio for those).
        batch_size: Flush chunks to disk every N videos to avoid OOM on large runs.
    """
    video_dir = Path(video_dir)
    pose_dir = Path(pose_dir)
    output_dir = Path(output_dir)
    staging_dir = output_dir / "_staging"
    audio_cache = Path(audio_dir) if audio_dir else output_dir / "_audio_cache"
    audio_cache.mkdir(parents=True, exist_ok=True)

    existing_dd, existing_stems = (None, set()) if rebuild else _load_existing_stems(output_dir)
    staged = set() if rebuild else _staged_stems(staging_dir)
    skip_stems = existing_stems | staged

    if existing_stems:
        print(f"Incremental mode: {len(existing_stems)} video stems already in dataset; skipping those.")
    if staged:
        print(f"Resuming: {len(staged)} video stems already in staging shards.")

    pose_files = sorted(pose_dir.glob("*_poses.npy"))
    print(f"Found {len(pose_files)} pose files")

    batch_records: list[dict] = []
    batch_videos = 0
    batch_count = len(list(staging_dir.glob("batch_*"))) if staging_dir.is_dir() else 0
    new_chunks = 0

    for pose_path in tqdm(pose_files, desc="Building dataset"):
        stem = pose_path.stem.replace("_poses", "")

        if stem in skip_stems:
            continue

        # Find matching video
        video_path = None
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            candidate = video_dir / f"{stem}{ext}"
            if candidate.exists():
                video_path = candidate
                break

        if video_path is None:
            print(f"  Warning: no video found for {stem}, skipping")
            continue

        # Load poses
        poses = np.load(pose_path)  # (T, 33, 9)

        # Extract / load audio
        try:
            audio_path = extract_audio_from_video(video_path, audio_cache)
            waveform = load_mono_waveform(audio_path, SAMPLE_RATE)
        except Exception as e:
            print(f"  Audio error for {stem}: {e}")
            continue

        # Chunk
        chunks = chunk_sequence(poses, waveform)
        for chunk in chunks:
            batch_records.append({
                "stem": stem,
                "pose_chunk": chunk["pose_chunk"].tolist(),      # store as nested list
                "audio_chunk": chunk["audio_chunk"].tolist(),
                "pose_shape": list(chunk["pose_chunk"].shape),
                "audio_shape": list(chunk["audio_chunk"].shape),
            })
            new_chunks += 1

        batch_videos += 1
        if batch_videos >= batch_size:
            _save_staging_shard(staging_dir, batch_count, batch_records)
            batch_count += 1
            batch_records = []
            batch_videos = 0

    if batch_records:
        _save_staging_shard(staging_dir, batch_count, batch_records)
        batch_count += 1

    print(f"New chunks this run: {new_chunks}")

    shard_paths = sorted(p for p in staging_dir.glob("batch_*") if _shard_is_valid(p))
    if not shard_paths and new_chunks == 0:
        if existing_dd is not None:
            print("Nothing new to add; leaving existing dataset unchanged.")
            return existing_dd
        print("No chunks produced (no pose files or all videos missing?).")
        return None

    if shard_paths:
        print(f"Merging {len(shard_paths)} staging shard(s) into {output_dir} ...")
        dataset_dict = _merge_staging_shards(
            staging_dir, existing_dd, output_dir, val_split, seed
        )
    else:
        dataset_dict = existing_dd

    if dataset_dict is not None:
        print(f"Dataset saved to {output_dir}")
        print(f"  Train: {len(dataset_dict['train'])} chunks")
        print(f"  Val:   {len(dataset_dict['validation'])} chunks")

    return dataset_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--pose_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--audio_dir", default=None)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore any existing dataset on disk and rebuild from all pose files (re-chunks everything).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=25,
        help="Flush staging shard every N videos (default: 25, lowers peak RAM).",
    )
    args = parser.parse_args()

    build_dataset(
        args.video_dir,
        args.pose_dir,
        args.output_dir,
        audio_dir=args.audio_dir,
        val_split=args.val_split,
        rebuild=args.rebuild,
        batch_size=args.batch_size,
    )
