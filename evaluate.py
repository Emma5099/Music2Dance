"""
evaluate.py — Dry-run evaluation: render predicted skeleton as video + beat score.

Works on the existing small dataset. Use this to validate the full pipeline
before gathering more data and doing a full training run.

Usage:
  # Full dry run: quick train (20 epochs) then evaluate
  python evaluate.py --dry_run

  # Evaluate the baseline checkpoint (default)
  python evaluate.py

  # Evaluate on a specific sample index from the validation set
  python evaluate.py --sample_idx 0

  # Seed first 2 s with ground-truth poses (momentum primer)
  python evaluate.py --primer gt --primer_frames 50

Output:
  outputs/predicted.mp4   — predicted skeleton animated to music
  outputs/groundtruth.mp4 — ground truth skeleton for comparison
  outputs/beat_score.txt  — beat alignment score (0–1, higher = better)

Requirements:
  pip install torch librosa datasets soundfile matplotlib opencv-python tqdm
"""

import argparse
import os
import subprocess
import sys
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datasets import load_from_disk
from tqdm import tqdm

from model import Music2PoseTransformer
from pose_postprocess import ground_poses
from train import (
    audio_to_features, DanceDataset, compute_norm_stats, CONTEXT_LEN,
    BASELINE_CKPT,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = "dataset"
CKPT_DIR     = "checkpoints"
OUTPUT_DIR   = "outputs"
POSE_FPS     = 25
AUDIO_SR     = 32_000
HOP_SAMPLES  = AUDIO_SR // POSE_FPS   # 1280

# MediaPipe skeleton edges (same as td_skeleton_renderer.py)
SKELETON_EDGES = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (11,12),(11,23),(12,24),(23,24),
    (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]
# ─────────────────────────────────────────────────────────────────────────────


# ── Quick training for dry run ────────────────────────────────────────────────

def default_checkpoint() -> str:
    """Project baseline (epoch 180), then best.pt / best_motion.pt."""
    if os.path.exists(BASELINE_CKPT):
        return BASELINE_CKPT
    for name in ("best.pt", "best_motion.pt"):
        path = os.path.join(CKPT_DIR, name)
        if os.path.exists(path):
            return path
    return BASELINE_CKPT


def quick_train(dataset_path: str, epochs: int = 30) -> str:
    """Train for a small number of epochs and return checkpoint path."""
    print(f"\n── Quick training ({epochs} epochs on existing data) ─────────────")
    result = subprocess.run([
        sys.executable, "train.py",
        "--dataset_path", dataset_path,
        "--epochs", str(epochs),
    ])

    if result.returncode != 0:
        raise RuntimeError(
            f"train.py exited with code {result.returncode}.\n"
            "Scroll up to see the error from training."
        )

    ckpt = default_checkpoint()
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Training finished but no checkpoint was created in {CKPT_DIR}/.\n"
            "This usually means the dataset has 0 valid batches — "
            "check that dataset/ is in the same folder as train.py."
        )
    return ckpt


# ── Inference ─────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device):
    # Checkpoint stores numpy norm stats alongside model weights (trusted local file).
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Music2PoseTransformer(
        audio_dim=ckpt["audio_dim"],
        pose_dim =ckpt["pose_dim"],
        **ckpt["config"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def motion_energy_xyz(poses_xyz: np.ndarray) -> float:
    """Mean joint speed from xyz positions."""
    vel = np.diff(poses_xyz, axis=0)
    return float(np.linalg.norm(vel, axis=-1).mean())


def run_inference(
    model, audio_feat: np.ndarray,
    x_mean, x_std, y_mean, y_std,
    device: torch.device,
    context_len: int = CONTEXT_LEN,
    primer_frames: int = 0,
    primer_poses_norm: np.ndarray | None = None,
) -> np.ndarray:
    """
    Autoregressive inference over a full audio sequence.
    Returns (T, 33, 3) predicted xyz positions.

    primer_frames: force the first N outputs to match primer_poses_norm (normalized 297-d).
    primer_poses_norm: (T, pose_dim) — use GT or mean pose in normalised space.
    """
    T = audio_feat.shape[0]

    # Normalise audio
    af_norm = (audio_feat - x_mean) / x_std          # (T, 82)
    af_t    = torch.tensor(af_norm, dtype=torch.float32, device=device)

    pose_dim = model.pose_dim
    pred_poses_norm = []

    # Rolling context (matches training / evaluate default)
    audio_ctx = torch.zeros(1, context_len, af_t.shape[-1], device=device)
    pose_ctx  = torch.zeros(1, context_len, pose_dim, device=device)

    if primer_poses_norm is not None:
        primer_t = torch.tensor(primer_poses_norm, dtype=torch.float32, device=device)

    with torch.no_grad():
        for t in tqdm(range(T), desc="Inferring frames", leave=False):
            audio_ctx = torch.roll(audio_ctx, -1, dims=1)
            audio_ctx[0, -1] = af_t[t]

            if primer_frames > 0 and primer_poses_norm is not None and t < primer_frames:
                next_pose = primer_t[t]
            else:
                next_pose = model.step(audio_ctx, pose_ctx)[0]

            pred_poses_norm.append(next_pose.cpu().numpy())

            pose_ctx = torch.roll(pose_ctx, -1, dims=1)
            pose_ctx[0, -1] = next_pose

    pred_norm = np.stack(pred_poses_norm)              # (T, pose_dim)
    pred_raw = pred_norm * y_std + y_mean              # (T, 297)
    pred_xyz = pred_raw.reshape(T, 33, 9)[:, :, :3]   # (T, 33, 3)
    return pred_xyz


# ── Skeleton rendering ────────────────────────────────────────────────────────

def draw_skeleton(ax, kpts: np.ndarray, color="cyan"):
    """Draw one skeleton frame on a matplotlib axis. kpts: (33, 3)."""
    ax.scatter(kpts[:, 0], -kpts[:, 1], s=15, c=color, zorder=3)
    for a, b in SKELETON_EDGES:
        ax.plot(
            [kpts[a, 0], kpts[b, 0]],
            [-kpts[a, 1], -kpts[b, 1]],
            c=color, linewidth=1.2, alpha=0.8,
        )


def render_video(
    poses_xyz: np.ndarray,    # (T, 33, 3)
    audio_raw: np.ndarray,    # (N_samples,) waveform
    out_path: str,
    title: str = "Skeleton",
    fps: int = POSE_FPS,
):
    """Render skeleton animation as .mp4 using matplotlib + ffmpeg."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    T = poses_xyz.shape[0]

    xmin, xmax = poses_xyz[:, :, 0].min() - 0.1, poses_xyz[:, :, 0].max() + 0.1
    ymin, ymax = poses_xyz[:, :, 1].min() - 0.1, poses_xyz[:, :, 1].max() + 0.1

    fig, ax = plt.subplots(figsize=(4, 6), facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-ymax, -ymin)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=10)

    def update(frame_idx):
        ax.cla()
        ax.set_facecolor("black")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-ymax, -ymin)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{title}  frame {frame_idx}/{T}", color="white", fontsize=9)
        draw_skeleton(ax, poses_xyz[frame_idx])

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000 / fps)
    writer = animation.FFMpegWriter(fps=fps, bitrate=1200)
    try:
        ani.save(out_path, writer=writer)
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  [warn] ffmpeg not available ({e}), saving frames as PNG instead")
        frames_dir = out_path.replace(".mp4", "_frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i in range(T):
            update(i)
            fig.savefig(os.path.join(frames_dir, f"{i:04d}.png"),
                        facecolor="black", dpi=80)
        print(f"  Frames saved to: {frames_dir}")
        print(f"  Combine with: ffmpeg -r {fps} -i {frames_dir}/%04d.png -c:v libx264 {out_path}")

    plt.close(fig)


# ── Beat alignment score ──────────────────────────────────────────────────────

def beat_alignment_score(poses_xyz: np.ndarray, audio_raw: np.ndarray) -> float:
    """
    Score how well motion energy peaks align with musical beats.
    Returns a value in [0, 1]: 1.0 = perfect alignment, ~0.1 = random.

    Method: correlate beat signal with kinematic energy signal.
    """
    T = poses_xyz.shape[0]

    # Motion energy: mean joint velocity magnitude per frame
    velocity   = np.diff(poses_xyz, axis=0)                  # (T-1, 33, 3)
    energy     = np.linalg.norm(velocity, axis=-1).mean(axis=-1)  # (T-1,)
    energy     = np.concatenate([[0], energy])                # (T,)

    # Beat signal from audio
    _, beats = librosa.beat.beat_track(y=audio_raw, sr=AUDIO_SR, hop_length=HOP_SAMPLES)
    beat_sig = np.zeros(T, dtype=np.float32)
    beat_sig[beats[beats < T].astype(int)] = 1.0

    if beat_sig.sum() == 0:
        return 0.0

    # Soft beat window (±2 frames)
    from scipy.ndimage import gaussian_filter1d
    beat_soft  = gaussian_filter1d(beat_sig.astype(float), sigma=2)
    energy_n   = (energy - energy.mean()) / (energy.std() + 1e-6)
    beat_n     = (beat_soft - beat_soft.mean()) / (beat_soft.std() + 1e-6)

    score = float(np.corrcoef(energy_n, beat_n)[0, 1])
    return max(0.0, score)   # clip negatives to 0


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(
    ckpt_path: str,
    dataset_path: str,
    sample_idx: int = 0,
    primer: str = "none",
    primer_frames: int = 0,
    character_mesh: str | None = None,
    ground: bool = False,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    dd       = load_from_disk(dataset_path)
    val_set  = dd["validation"]
    n_val    = len(val_set)
    sample_idx = min(sample_idx, n_val - 1)
    sample   = val_set[sample_idx]

    print(f"Val set: {n_val} samples  |  Evaluating sample #{sample_idx} ({sample['stem']})")

    pose_gt  = np.array(sample["pose_chunk"],  dtype=np.float32)   # (T, 33, 9)
    audio_raw = np.array(sample["audio_chunk"], dtype=np.float32)[0] # (N,)

    T        = pose_gt.shape[0]
    pose_gt_full = pose_gt.reshape(T, -1)
    pose_gt_xyz = pose_gt[:, :, :3]                                 # (T, 33, 3)

    # Audio features
    audio_feat = audio_to_features(audio_raw)                       # (T, 82)
    T = min(T, audio_feat.shape[0])
    pose_gt_full = pose_gt_full[:T]
    pose_gt_xyz = pose_gt_xyz[:T]
    audio_feat  = audio_feat[:T]

    # ── Load checkpoint & norm stats ─────────────────────────────────────────
    model, ckpt = load_model(ckpt_path, device)
    x_mean, x_std = ckpt["x_mean"], ckpt["x_std"]
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

    print(f"Checkpoint epoch: {ckpt['epoch']}")
    if "val_motion_ratio" in ckpt:
        print(f"  Saved motion%: {100 * ckpt['val_motion_ratio']:.1f}")

    primer_poses_norm = None
    if primer_frames > 0:
        if primer == "gt":
            primer_poses_norm = (pose_gt_full - y_mean) / y_std
            print(f"Primer: {primer_frames} frames of ground-truth poses")
        elif primer == "mean":
            primer_poses_norm = np.zeros((T, pose_gt_full.shape[-1]), dtype=np.float32)
            print(f"Primer: {primer_frames} frames of dataset-mean pose")
        else:
            print(f"Unknown primer '{primer}' — ignoring primer_frames")

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\nRunning autoregressive inference...")
    pred_xyz = run_inference(
        model, audio_feat, x_mean, x_std, y_mean, y_std, device,
        primer_frames=primer_frames if primer_poses_norm is not None else 0,
        primer_poses_norm=primer_poses_norm,
    )

    # ── Reconstruction loss (sanity check) ───────────────────────────────────
    pred_flat = pred_xyz.reshape(T, -1)
    gt_flat   = pose_gt_xyz.reshape(T, -1)
    mse       = float(np.mean((pred_flat - gt_flat) ** 2))
    print(f"\nReconstruction MSE (xyz only): {mse:.4f}")
    if mse > 1.0:
        print("  → High MSE expected on small dataset / few epochs — normal for a dry run.")
    else:
        print("  → Low MSE — model is fitting well.")

    # ── Beat alignment score ──────────────────────────────────────────────────
    score_pred = beat_alignment_score(pred_xyz, audio_raw[:T * HOP_SAMPLES])
    score_gt   = beat_alignment_score(pose_gt_xyz, audio_raw[:T * HOP_SAMPLES])
    speed_pred = motion_energy_xyz(pred_xyz)
    speed_gt   = motion_energy_xyz(pose_gt_xyz)
    motion_pct = 100.0 * speed_pred / (speed_gt + 1e-6)
    print(f"\nMotion energy (mean joint speed):")
    print(f"  Predicted  : {speed_pred:.4f}  ({motion_pct:.1f}% of GT)")
    print(f"  Ground truth: {speed_gt:.4f}")
    print(f"\nBeat alignment score:")
    print(f"  Predicted  : {score_pred:.3f}")
    print(f"  Ground truth: {score_gt:.3f}")
    if score_pred > 0.1:
        print("  → Some beat correlation detected.")
    else:
        print("  → No beat correlation yet — expected with small data / few epochs.")

    with open(os.path.join(OUTPUT_DIR, "beat_score.txt"), "w") as f:
        f.write(f"sample: {sample['stem']}\n")
        f.write(f"epoch:  {ckpt['epoch']}\n")
        f.write(f"MSE (xyz): {mse:.4f}\n")
        f.write(f"motion_speed_predicted:   {speed_pred:.4f}\n")
        f.write(f"motion_speed_groundtruth: {speed_gt:.4f}\n")
        f.write(f"motion_pct_of_gt: {motion_pct:.1f}\n")
        f.write(f"beat_alignment_predicted:   {score_pred:.3f}\n")
        f.write(f"beat_alignment_groundtruth: {score_gt:.3f}\n")
        if primer_frames > 0 and primer_poses_norm is not None:
            f.write(f"primer: {primer} x {primer_frames} frames\n")

    # ── Render videos ─────────────────────────────────────────────────────────
    print(f"\nRendering videos to {OUTPUT_DIR}/ ...")
    pred_render = ground_poses(pred_xyz) if ground else pred_xyz
    gt_render   = ground_poses(pose_gt_xyz) if ground else pose_gt_xyz
    if ground:
        print("  Grounding: planted lowest foot on median floor plane")
    pred_path = os.path.join(OUTPUT_DIR, "predicted_poses.npy")
    np.save(pred_path, pred_render.astype(np.float32))
    print(f"  Saved poses: {pred_path}")
    render_video(pred_render, audio_raw, os.path.join(OUTPUT_DIR, "predicted.mp4"),    title="Predicted")
    render_video(gt_render,   audio_raw, os.path.join(OUTPUT_DIR, "groundtruth.mp4"), title="Ground Truth")

    if character_mesh:
        char_out = os.path.join(OUTPUT_DIR, "predicted_character.mp4")
        # Write a short wav for muxing this sample
        wav_path = os.path.join(OUTPUT_DIR, "_eval_audio.wav")
        import soundfile as sf
        sf.write(wav_path, audio_raw, AUDIO_SR)
        print(f"\nRendering character mesh → {char_out}")
        subprocess.run([
            sys.executable, "3Dmodel/animate_character.py",
            "--poses", pred_path,
            "--mesh", character_mesh,
            "--audio", wav_path,
            "--out", char_out,
        ], check=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"RESULTS for dry run on '{sample['stem']}'")
    print(f"  MSE (xyz only)      : {mse:.4f}")
    print(f"  Motion speed (pred) : {speed_pred:.4f}  ({motion_pct:.1f}% of GT)")
    print(f"  Beat score (pred)   : {score_pred:.3f}  (GT: {score_gt:.3f})")
    print(f"  Videos              : {OUTPUT_DIR}/predicted.mp4")
    print(f"                        {OUTPUT_DIR}/groundtruth.mp4")
    print(f"\nWhat to look for in the videos:")
    print(f"  ✓ Skeleton is roughly human-shaped (not exploded)")
    print(f"  ✓ There is visible motion (not a frozen pose)")
    print(f"  ✓ Motion has some rhythm (even if not perfect)")
    print(f"  → Full beat alignment only develops with more data + more epochs.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    default=None,
                        help="Path to .pt checkpoint. If omitted with --dry_run, trains first.")
    parser.add_argument("--dataset_path",  default=DATASET_PATH)
    parser.add_argument("--sample_idx",    type=int, default=0,
                        help="Which validation sample to evaluate (default: 0)")
    parser.add_argument("--dry_run",       action="store_true",
                        help="Quick-train for 30 epochs then evaluate.")
    parser.add_argument("--epochs",        type=int, default=30,
                        help="Epochs for --dry_run (default: 30)")
    parser.add_argument("--primer",        default="none", choices=["none", "gt", "mean"],
                        help="Pose primer for first N frames (gt or mean in normalised space)")
    parser.add_argument("--primer_frames", type=int, default=0,
                        help="Number of primer frames (e.g. 50 = 2 s at 25 fps)")
    parser.add_argument("--character_mesh", default=None,
                        help="Optional GLB to skin onto predicted poses")
    parser.add_argument("--ground", action="store_true",
                        help="Plant lowest foot on a floor plane (post-process; prefer training fixes)")
    args = parser.parse_args()

    if args.dry_run and args.checkpoint is None:
        ckpt_path = quick_train(args.dataset_path, epochs=args.epochs)
    elif args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = default_checkpoint()
        if not os.path.exists(ckpt_path):
            print(f"No checkpoint found in {CKPT_DIR}/. Run with --dry_run or --checkpoint.")
            sys.exit(1)

    evaluate(
        ckpt_path, args.dataset_path,
        sample_idx=args.sample_idx,
        primer=args.primer,
        primer_frames=args.primer_frames,
        character_mesh=args.character_mesh,
        ground=args.ground,
    )


if __name__ == "__main__":
    main()
