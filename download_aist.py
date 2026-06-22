"""
download_aist.py — Download AIST++ videos and build the training dataset.

This script:
  1. Downloads the AIST++ video list from Google
  2. Filters to camera c01 only (~1408 videos, ~14 GB) to avoid redundant
     multi-angle copies of the same dance sequence
  3. Skips videos already present on disk (safe to re-run / resume)
  4. Optionally runs extract_poses.py + build_dataset.py automatically

Usage:
  # Download only
  python download_aist.py --download_folder videos/

  # Download + extract poses + build dataset in one go
  python download_aist.py --download_folder videos/ --auto_pipeline

  # Faster download with more parallel workers
  python download_aist.py --download_folder videos/ --num_processes 10

  # Limit to N videos (useful for a quick test)
  python download_aist.py --download_folder videos/ --max_videos 50

Estimated sizes:
  ~1408 videos × ~10 MB each ≈ 14 GB
  ~1408 videos × ~6 chunks/min ≈ 8 000+ training chunks

Before running, read and agree to the AIST terms of use:
  https://aistdancedb.ongaaccel.jp/terms_of_use/
"""

import argparse
import multiprocessing
import os
import sys
import subprocess
import urllib.request
from functools import partial
from pathlib import Path

SOURCE_URL = "https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M/"
# Original AIST++ subset list (GCS bucket removed as of 2025).
LIST_URL   = "https://storage.googleapis.com/aist_plusplus_public/20121228/video_list.txt"
# Fallback: full AIST DB URL list (filter by camera in fetch_video_list).
AIST_CSV_URL = (
    "https://aistdancedb.ongaaccel.jp/v1.0.0/data/video_raw/10M/"
    "raw_10M_all_video_url.csv"
)

# Camera angle to download — c01 gives one view per unique sequence.
# Change to "cAll" to get all 9 angles (9× more data but highly redundant).
CAMERA_FILTER = "_c01_"


def _fetch_from_aist_csv(camera_filter: str) -> list[str]:
    """Fallback when the AIST++ GCS video_list.txt is unavailable."""
    print(f"  Fetching from AIST CSV: {AIST_CSV_URL}")
    raw = urllib.request.urlopen(AIST_CSV_URL)
    names = set()
    for line in raw:
        url = line.strip().decode("utf-8")
        if not url.endswith(".mp4"):
            continue
        name = os.path.basename(url).removesuffix(".mp4")
        if camera_filter in name:
            names.add(name)
    return sorted(names)


def fetch_video_list(camera_filter: str) -> list[str]:
    """Download the master video list and filter to one camera angle."""
    print("Fetching video list...")
    try:
        raw = urllib.request.urlopen(LIST_URL)
        all_names = [line.strip().decode("utf-8") for line in raw if line.strip()]
    except Exception as e:
        print(f"  AIST++ list unavailable ({e}); using AIST DB CSV fallback.")
        filtered = _fetch_from_aist_csv(camera_filter)
        print(f"  After filter  : {len(filtered)}  (camera '{camera_filter.strip('_')}')")
        return filtered

    filtered = [n for n in all_names if camera_filter in n]
    print(f"  Total in list : {len(all_names)}")
    print(f"  After filter  : {len(filtered)}  (camera '{camera_filter.strip('_')}')")
    return filtered


def _download_one(seq_name: str, download_folder: str):
    """Download a single video; skip if already present."""
    filename = seq_name + ".mp4"
    save_path = os.path.join(download_folder, filename)

    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        return "skip"

    url = SOURCE_URL + filename
    try:
        urllib.request.urlretrieve(url, save_path)
        return "ok"
    except Exception as e:
        # Remove partial file
        if os.path.exists(save_path):
            os.remove(save_path)
        return f"error: {e}"


def download(seq_names: list[str], download_folder: str, num_processes: int = 5):
    os.makedirs(download_folder, exist_ok=True)

    # Skip already-downloaded
    missing = [n for n in seq_names
               if not (Path(download_folder) / (n + ".mp4")).exists()
               or (Path(download_folder) / (n + ".mp4")).stat().st_size == 0]
    already  = len(seq_names) - len(missing)
    print(f"  Already on disk : {already}")
    print(f"  To download     : {len(missing)}")

    if not missing:
        print("All videos already downloaded.")
        return

    download_func = partial(_download_one, download_folder=download_folder)
    pool = multiprocessing.Pool(processes=num_processes)

    ok, skipped, errors = 0, 0, 0
    for i, result in enumerate(pool.imap_unordered(download_func, missing)):
        if result == "ok":
            ok += 1
        elif result == "skip":
            skipped += 1
        else:
            errors += 1
            sys.stderr.write(f"\n  [warn] {result}")
        sys.stderr.write(f"\r  Progress: {i+1}/{len(missing)}  "
                         f"ok={ok}  skip={skipped}  err={errors}   ")
    sys.stderr.write("\n")
    pool.close()
    pool.join()
    print(f"Download complete: {ok} downloaded, {skipped} skipped, {errors} errors.")


def run_pipeline(download_folder: str, pose_dir: str, dataset_dir: str):
    """Run extract_poses → build_dataset on the downloaded videos."""
    script_dir = Path(__file__).parent

    print("\n── Step 1: Extract poses ──────────────────────────────────────")
    subprocess.run([
        sys.executable,
        str(script_dir / "data" / "extract_poses.py"),
        "--video_dir", download_folder,
        "--output_dir", pose_dir,
    ], check=True)

    print("\n── Step 2: Build HuggingFace dataset ──────────────────────────")
    subprocess.run([
        sys.executable,
        str(script_dir / "data" / "build_dataset.py"),
        "--video_dir",  download_folder,
        "--pose_dir",   pose_dir,
        "--output_dir", dataset_dir,
    ], check=True)

    print(f"\nDone! Dataset saved to: {dataset_dir}")
    print("Next step → upload to Colab and run: python train.py --dataset_path dataset/")


def main():
    parser = argparse.ArgumentParser(
        description="Download AIST++ videos + optionally build the training dataset."
    )
    parser.add_argument("--download_folder", required=True,
                        help="Where to save .mp4 files")
    parser.add_argument("--pose_dir",        default="poses/",
                        help="Where to save extracted pose .npy files (default: poses/)")
    parser.add_argument("--dataset_dir",     default="dataset/",
                        help="Where to save HuggingFace dataset (default: dataset/)")
    parser.add_argument("--num_processes",   type=int, default=5,
                        help="Parallel download workers (default: 5)")
    parser.add_argument("--max_videos",      type=int, default=None,
                        help="Limit to first N videos — useful for testing")
    parser.add_argument("--camera",          default=CAMERA_FILTER.strip("_"),
                        help="Camera filter string (default: c01)")
    parser.add_argument("--auto_pipeline",   action="store_true",
                        help="After downloading, automatically run extract_poses + build_dataset")
    args = parser.parse_args()

    # ── Terms of use ─────────────────────────────────────────────────────────
    print("\nBefore downloading, please read the AIST Terms of Use:")
    print("  https://aistdancedb.ongaaccel.jp/terms_of_use/\n")
    ans = input("Do you agree with the Terms of Use? [Y/N] ").strip()
    if ans.lower() not in ("y", "yes"):
        print("Exiting. Please read and agree to the Terms of Use first.")
        sys.exit(0)

    # ── Fetch list ───────────────────────────────────────────────────────────
    camera_filter = f"_{args.camera}_"
    seq_names = fetch_video_list(camera_filter)

    if args.max_videos:
        seq_names = seq_names[:args.max_videos]
        print(f"  Limited to {args.max_videos} videos (--max_videos)")

    # ── Download ─────────────────────────────────────────────────────────────
    print(f"\nDownloading to: {args.download_folder}")
    download(seq_names, args.download_folder, args.num_processes)

    # ── Pipeline ─────────────────────────────────────────────────────────────
    if args.auto_pipeline:
        run_pipeline(args.download_folder, args.pose_dir, args.dataset_dir)
    else:
        print(f"\nNext steps:")
        print(f"  python data/extract_poses.py --video_dir {args.download_folder} --output_dir {args.pose_dir}")
        print(f"  python data/build_dataset.py --video_dir {args.download_folder} --pose_dir {args.pose_dir} --output_dir {args.dataset_dir}")


if __name__ == "__main__":
    main()
