"""
Prepare a HuggingFace dataset for Colab training.

Google Drive cannot memory-map `.arrow` shards (HuggingFace `load_from_disk`
uses mmap). Training must read the dataset from local Colab disk (`/content`),
while checkpoints can stay on Drive so they survive runtime disconnects.

Expected Drive layout (MyDrive/Code/Music2Dance/):

    dataset.zip                  # preferred — one complete archive
    dataset/                     # optional; often incomplete on Drive
      dataset_dict.json
      train/state.json
      train/data-00000-of-00016.arrow … data-00015-of-00016.arrow
      validation/data-00000-of-00001.arrow
    checkpoints/epoch_120.pt     # resume point

Usage (Colab):

    from google.colab import drive
    drive.mount("/content/drive")
    !python data/prepare_colab.py --drive_root /content/drive/MyDrive/Code/Music2Dance
    !python train.py --dataset_path /content/dataset \\
        --resume /content/drive/MyDrive/Code/Music2Dance/checkpoints/epoch_120.pt \\
        --ckpt_dir /content/drive/MyDrive/Code/Music2Dance/checkpoints --epochs 250
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from typing import Iterable

COLAB_LOCAL_DATASET = "/content/dataset"
SKIP_PREFIXES = ("dataset/_audio_cache", "__MACOSX")


def _is_colab() -> bool:
    return os.path.isdir("/content")


def _is_gdrive_path(path: str) -> bool:
    p = os.path.abspath(path)
    return "/content/drive/" in p.replace("\\", "/")


def dataset_missing_files(dataset_path: str) -> list[str]:
    """Return paths that HuggingFace `load_from_disk` would need but are missing."""
    missing: list[str] = []
    marker = os.path.join(dataset_path, "dataset_dict.json")
    if not os.path.isfile(marker):
        return [marker]
    with open(marker, encoding="utf-8") as f:
        splits: Iterable[str] = json.load(f).get("splits", ["train", "validation"])
    for split in splits:
        split_dir = os.path.join(dataset_path, split)
        info = os.path.join(split_dir, "dataset_info.json")
        state_path = os.path.join(split_dir, "state.json")
        for p in (info, state_path):
            if not os.path.isfile(p):
                missing.append(p)
        if not os.path.isfile(state_path):
            continue
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        for entry in state.get("_data_files", []):
            fp = os.path.join(split_dir, entry["filename"])
            if not os.path.isfile(fp) or os.path.getsize(fp) == 0:
                missing.append(fp)
    return missing


def _copy_file(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return
    shutil.copy2(src, dst)


def copy_hf_dataset(src: str, dst: str) -> None:
    """Copy only the shards listed in state.json (skip leftover/old arrows and audio cache)."""
    os.makedirs(dst, exist_ok=True)
    _copy_file(
        os.path.join(src, "dataset_dict.json"),
        os.path.join(dst, "dataset_dict.json"),
    )
    with open(os.path.join(src, "dataset_dict.json"), encoding="utf-8") as f:
        splits = json.load(f)["splits"]
    for split in splits:
        src_split = os.path.join(src, split)
        dst_split = os.path.join(dst, split)
        os.makedirs(dst_split, exist_ok=True)
        for name in ("dataset_info.json", "state.json"):
            _copy_file(os.path.join(src_split, name), os.path.join(dst_split, name))
        with open(os.path.join(src_split, "state.json"), encoding="utf-8") as f:
            state = json.load(f)
        for entry in state["_data_files"]:
            filename = entry["filename"]
            print(f"  copy {split}/{filename}")
            _copy_file(os.path.join(src_split, filename), os.path.join(dst_split, filename))


def extract_dataset_zip(zip_path: str, dest_parent: str) -> str:
    dest = os.path.join(dest_parent, "dataset")
    os.makedirs(dest_parent, exist_ok=True)
    print(f"Extracting {zip_path} → {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            info for info in zf.infolist()
            if info.filename.startswith("dataset/")
            and not any(info.filename.startswith(p) or f"/{p}/" in info.filename
                        for p in SKIP_PREFIXES)
            and "__MACOSX" not in info.filename
        ]
        if not members:
            raise FileNotFoundError(f"No dataset/ files inside {zip_path}")
        for i, info in enumerate(members, 1):
            if i == 1 or i % 20 == 0 or i == len(members):
                print(f"  [{i}/{len(members)}] {info.filename}")
            zf.extract(info, dest_parent)
    return dest


def find_dataset_zip(dataset_path: str, drive_root: str | None = None) -> str | None:
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(dataset_path.rstrip("/"))), "dataset.zip"),
        os.path.join(os.getcwd(), "dataset.zip"),
    ]
    if drive_root:
        candidates.insert(0, os.path.join(drive_root, "dataset.zip"))
    for path in candidates:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
    return None


def materialize_dataset(dataset_path: str, drive_root: str | None = None) -> str:
    """
    Return a local path that `datasets.load_from_disk` can mmap.

    On Colab, Drive paths are always copied/unzipped to /content/dataset.
    """
    dataset_path = os.path.abspath(dataset_path)
    missing = dataset_missing_files(dataset_path)
    on_drive = _is_gdrive_path(dataset_path)
    use_local_ssd = _is_colab() and on_drive

    if use_local_ssd:
        local = COLAB_LOCAL_DATASET
        if not dataset_missing_files(local):
            print(f"Using local dataset cache: {local}")
            return local

        zip_path = find_dataset_zip(dataset_path, drive_root)
        if zip_path:
            extract_dataset_zip(zip_path, "/content")
            leftover = dataset_missing_files(local)
            if leftover:
                raise FileNotFoundError(
                    "dataset.zip extracted but still missing:\n  " + "\n  ".join(leftover)
                )
            print(f"Dataset ready at {local}")
            return local

        if missing:
            raise FileNotFoundError(_missing_help(dataset_path, missing, drive_root))

        print(
            "Copying dataset from Drive → /content/dataset "
            "(Arrow memory-mapping does not work on Google Drive)."
        )
        copy_hf_dataset(dataset_path, local)
        leftover = dataset_missing_files(local)
        if leftover:
            raise FileNotFoundError(
                "Copy finished but still missing:\n  " + "\n  ".join(leftover)
            )
        return local

    if missing:
        zip_path = find_dataset_zip(dataset_path, drive_root)
        if zip_path:
            dest_parent = os.path.dirname(dataset_path.rstrip("/")) or "."
            extract_dataset_zip(zip_path, dest_parent)
            leftover = dataset_missing_files(dataset_path)
            if leftover:
                raise FileNotFoundError(
                    "dataset.zip extracted but still missing:\n  " + "\n  ".join(leftover)
                )
            return dataset_path
        raise FileNotFoundError(_missing_help(dataset_path, missing, drive_root))

    return dataset_path


def _missing_help(dataset_path: str, missing: list[str], drive_root: str | None) -> str:
    root = drive_root or os.path.dirname(dataset_path.rstrip("/"))
    shown = missing[:8]
    extra = f"\n  … and {len(missing) - 8} more" if len(missing) > 8 else ""
    return (
        f"HuggingFace dataset at {dataset_path} is incomplete "
        f"({len(missing)} missing files). First missing:\n  "
        + "\n  ".join(shown)
        + extra
        + "\n\nPut the full archive on Drive and re-run:\n"
        f"  {os.path.join(root, 'dataset.zip')}\n"
        "Then train from local disk, e.g.\n"
        "  python train.py --dataset_path /content/dataset \\\n"
        f"    --resume {os.path.join(root, 'checkpoints', 'epoch_120.pt')} \\\n"
        f"    --ckpt_dir {os.path.join(root, 'checkpoints')} --epochs 250"
    )


def resolve_ckpt_dir(resume: str | None, ckpt_dir: str | None) -> str:
    if ckpt_dir:
        return ckpt_dir
    if resume:
        parent = os.path.dirname(os.path.abspath(resume))
        if parent:
            return parent
    return "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Music2Dance dataset for Colab")
    parser.add_argument(
        "--drive_root",
        default="/content/drive/MyDrive/Code/Music2Dance",
        help="Project folder on Google Drive",
    )
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Override dataset folder (default: <drive_root>/dataset)",
    )
    args = parser.parse_args()
    dataset_path = args.dataset_path or os.path.join(args.drive_root, "dataset")
    local = materialize_dataset(dataset_path, drive_root=args.drive_root)
    resume = os.path.join(args.drive_root, "checkpoints", "epoch_120.pt")
    ckpt_dir = os.path.join(args.drive_root, "checkpoints")
    print("\nDataset ready:", local)
    if os.path.isfile(resume):
        print("Resume checkpoint found:", resume)
    else:
        print("WARNING: resume checkpoint not found:", resume)
    print("\nTrain with:")
    print(
        f"  python train.py --dataset_path {local} "
        f"--resume {resume} --ckpt_dir {ckpt_dir} --epochs 250"
    )


if __name__ == "__main__":
    main()
