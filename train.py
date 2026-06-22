"""
train.py — Train Music2Pose on Colab or any cloud GPU.

Loads directly from the HuggingFace dataset built by data/build_dataset.py.

Quick start (Colab):
  !pip install torch librosa numpy tqdm datasets soundfile
  !python train.py --dataset_path dataset/

Resume from the baseline checkpoint:
  !python train.py --dataset_path dataset/ --resume checkpoints/epoch_180.pt --epochs 200

Loss: weighted xyz + velocity/accel channels, temporal motion losses, beat alignment.
Training uses scheduled sampling and rolling 60-frame windows.
Saves best.pt (lowest val), best_motion.pt (most motion), best_balanced.pt (sweet spot).

Dataset format (per sample):
  pose_chunk  : (250, 33, 9)  — 10 s at 25 fps, xyz + Δxyz + Δ²xyz
  audio_chunk : (1, 320000)   — 10 s raw mono waveform at 32 kHz

Pose is flattened to (T, 297). Audio is converted to mel+onset+beat (T, 82).
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import librosa
from datasets import load_from_disk

from model import Music2PoseTransformer

# ── Hyperparameters ──────────────────────────────────────────────────────────
DATASET_PATH = "dataset"
CKPT_DIR       = "checkpoints"
BASELINE_CKPT  = os.path.join(CKPT_DIR, "epoch_180.pt")
BATCH_SIZE   = 16          # lower than before: mel extraction adds RAM
EPOCHS       = int(os.environ.get("EPOCHS", 200))   # override via env for dry run
LR           = 3e-4
WEIGHT_DECAY = 1e-4
SAVE_EVERY   = 5

# Audio
POSE_FPS     = 25
AUDIO_SR     = 32_000
N_MELS       = 80
# hop so that 1 spectrogram frame == 1 pose frame
HOP_SAMPLES  = AUDIO_SR // POSE_FPS   # 1280

# Model
D_MODEL      = 256
NHEAD        = 8
NUM_LAYERS   = 6
DROPOUT      = 0.1

# Pose layout: (T, 297) = 33 landmarks × [xyz | Δxyz | Δ²xyz]
POSE_XYZ_SL   = slice(0, 99)
POSE_VEL_SL   = slice(99, 198)
POSE_ACCEL_SL = slice(198, 297)
ONSET_FEAT_IDX = 80
BEAT_FEAT_IDX  = 81

# Loss weights — favour motion over static pose regression
W_XYZ         = 1.0
W_VEL_CH      = 0.3
W_ACCEL_CH    = 0.3
W_TEMP_VEL    = 1.0
W_TEMP_ACCEL  = 0.5
W_BEAT        = 0.2

# Scheduled sampling + rolling-context training (matches inference)
CONTEXT_LEN    = 60
SS_START       = 15
SS_END         = 100
SS_MAX         = 0.5
CTX_TRAIN_PROB = 0.25

# Balanced checkpoint — between static (ep 120 ~16%) and hyperactive (ep 85 ~91%)
TARGET_MOTION_RATIO = 0.32   # aim for ~32% of GT motion
MOTION_RATIO_LOW    = 0.18   # below this → mannequin penalty
MOTION_RATIO_HIGH   = 0.60   # above this → too wild (ep 85 territory)
W_BALANCED_MOTION   = 5.0    # weight of motion-band penalty vs val loss
W_MOTION_TARGET     = 0.15   # soft training nudge toward TARGET_MOTION_RATIO
# ─────────────────────────────────────────────────────────────────────────────


def audio_to_features(waveform: np.ndarray) -> np.ndarray:
    """
    Convert raw mono waveform (n_samples,) → (T, 82) feature matrix.
    T matches the number of pose frames (one feature frame per pose frame).
    """
    y = waveform.astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=y, sr=AUDIO_SR, n_mels=N_MELS, hop_length=HOP_SAMPLES
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)          # (80, T)

    onset = librosa.onset.onset_strength(
        y=y, sr=AUDIO_SR, hop_length=HOP_SAMPLES
    )                                                        # (T,)

    _, beats = librosa.beat.beat_track(
        y=y, sr=AUDIO_SR, hop_length=HOP_SAMPLES
    )

    # mel and onset can differ by 1 frame due to librosa rounding — align to min length
    T = min(mel_db.shape[1], len(onset))
    mel_db = mel_db[:, :T]
    onset = onset[:T]

    beat_sig = np.zeros(T, dtype=np.float32)
    beat_sig[beats[beats < T]] = 1.0

    feat = np.concatenate([mel_db.T, onset[:, None], beat_sig[:, None]], axis=-1)
    return feat.astype(np.float32)       # (T, 82)


class DanceDataset(Dataset):
    def __init__(self, hf_split, x_mean=None, x_std=None, y_mean=None, y_std=None):
        self.data = hf_split
        self.x_mean = x_mean
        self.x_std  = x_std
        self.y_mean = y_mean
        self.y_std  = y_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # ── Poses: (T, 33, 9) → (T, 297) ─────────────────────────────────
        pose = np.array(sample["pose_chunk"], dtype=np.float32)  # (T, 33, 9)
        T    = pose.shape[0]
        pose = pose.reshape(T, -1)                               # (T, 297)

        # ── Audio: (1, N) raw waveform → (T, 82) features ─────────────────
        audio_raw = np.array(sample["audio_chunk"], dtype=np.float32)[0]  # (N,)
        audio_feat = audio_to_features(audio_raw)                         # (T_audio, 82)

        # Align lengths (mel frame count may differ by 1 due to rounding)
        T_audio = audio_feat.shape[0]
        T = min(T, T_audio)
        pose       = pose[:T]
        audio_feat = audio_feat[:T]

        # ── Normalise ──────────────────────────────────────────────────────
        if self.x_mean is not None:
            audio_feat = (audio_feat - self.x_mean) / self.x_std
        if self.y_mean is not None:
            pose = (pose - self.y_mean) / self.y_std

        audio_t = torch.tensor(audio_feat, dtype=torch.float32)
        pose_t  = torch.tensor(pose,       dtype=torch.float32)

        # Decoder input: poses shifted right (first frame = zeros)
        poses_in  = torch.cat([torch.zeros(1, pose_t.size(-1)), pose_t[:-1]], dim=0)
        poses_tgt = pose_t

        return audio_t, poses_in, poses_tgt


def compute_norm_stats(hf_split, n_samples: int = 500):
    """Estimate mean/std from a subset of the training data."""
    print(f"Computing normalisation stats from {n_samples} samples...")
    X_acc, Y_acc = [], []
    indices = np.random.choice(len(hf_split), min(n_samples, len(hf_split)), replace=False)
    for i in tqdm(indices, desc="Norm stats"):
        sample = hf_split[int(i)]
        pose = np.array(sample["pose_chunk"], dtype=np.float32)
        T = pose.shape[0]
        pose = pose.reshape(T, -1)

        audio_raw = np.array(sample["audio_chunk"], dtype=np.float32)[0]
        audio_feat = audio_to_features(audio_raw)
        T_audio = audio_feat.shape[0]
        T = min(T, T_audio)

        X_acc.append(audio_feat[:T])
        Y_acc.append(pose[:T])

    X = np.concatenate(X_acc, axis=0)
    Y = np.concatenate(Y_acc, axis=0)

    x_mean, x_std = X.mean(axis=0), X.std(axis=0) + 1e-6
    y_mean, y_std = Y.mean(axis=0), Y.std(axis=0) + 1e-6
    return x_mean, x_std, y_mean, y_std


def velocity_loss(pred, target):
    d_pred   = pred[:, 1:] - pred[:, :-1]
    d_target = target[:, 1:] - target[:, :-1]
    return F.mse_loss(d_pred, d_target)


def temporal_velocity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match frame-to-frame xyz motion."""
    xyz_p = pred[..., POSE_XYZ_SL]
    xyz_t = target[..., POSE_XYZ_SL]
    return F.mse_loss(xyz_p[:, 1:] - xyz_p[:, :-1], xyz_t[:, 1:] - xyz_t[:, :-1])


def temporal_acceleration_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match xyz acceleration (second temporal difference)."""
    xyz_p = pred[..., POSE_XYZ_SL]
    xyz_t = target[..., POSE_XYZ_SL]
    vel_p = xyz_p[:, 1:] - xyz_p[:, :-1]
    vel_t = xyz_t[:, 1:] - xyz_t[:, :-1]
    return F.mse_loss(vel_p[:, 1:] - vel_p[:, :-1], vel_t[:, 1:] - vel_t[:, :-1])


def motion_ratio_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample motion energy as a fraction of ground truth. Returns (B,)."""
    return batch_motion_energy(pred) / (batch_motion_energy(target) + 1e-6)


def motion_target_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Nudge batch motion ratio toward TARGET_MOTION_RATIO."""
    ratio = motion_ratio_batch(pred, target)
    return (ratio - TARGET_MOTION_RATIO).pow(2).mean()


def motion_deviation_penalty(motion_ratio: float) -> float:
    """
    Penalty for being outside the sweet spot (linear — stronger signal than squared).
    Low at TARGET_MOTION_RATIO, rises toward static or hyperactive extremes.
    """
    if motion_ratio < MOTION_RATIO_LOW:
        return MOTION_RATIO_LOW - motion_ratio
    if motion_ratio > MOTION_RATIO_HIGH:
        return motion_ratio - MOTION_RATIO_HIGH
    return abs(motion_ratio - TARGET_MOTION_RATIO)


def balanced_score(val_loss: float, motion_ratio: float) -> float:
    """Lower is better. Combines reconstruction quality with motion sweet spot."""
    return val_loss + W_BALANCED_MOTION * motion_deviation_penalty(motion_ratio)


def beat_alignment_loss(pred: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
    """
    Encourage predicted motion energy to correlate with onset + beat features.
    audio: (B, T, 82) normalised features; last two dims are onset and beat.
    """
    T = pred.size(1)
    if audio.size(1) != T:
        audio = audio[:, :T]

    xyz = pred[..., POSE_XYZ_SL].view(pred.size(0), T, 33, 3)
    energy = (xyz[:, 1:] - xyz[:, :-1]).pow(2).sum(-1).sqrt().mean(-1)   # (B, T-1)
    energy = F.pad(energy, (1, 0), value=0.0)

    music = audio[..., ONSET_FEAT_IDX] + audio[..., BEAT_FEAT_IDX]
    energy = energy - energy.mean(dim=1, keepdim=True)
    music  = music  - music.mean(dim=1, keepdim=True)
    num = (energy * music).sum(dim=1)
    den = energy.norm(dim=1) * music.norm(dim=1) + 1e-6
    corr = num / den
    return (1.0 - corr).mean()


def batch_motion_energy(pose: torch.Tensor) -> torch.Tensor:
    """Mean joint speed from xyz channels. Returns (B,)."""
    xyz = pose[..., POSE_XYZ_SL].view(pose.size(0), pose.size(1), 33, 3)
    vel = xyz[:, 1:] - xyz[:, :-1]
    return vel.pow(2).sum(-1).sqrt().mean(dim=(1, 2))


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    audio: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    l_xyz   = F.mse_loss(pred[..., POSE_XYZ_SL],   target[..., POSE_XYZ_SL])
    l_vel   = F.mse_loss(pred[..., POSE_VEL_SL],   target[..., POSE_VEL_SL])
    l_acc   = F.mse_loss(pred[..., POSE_ACCEL_SL], target[..., POSE_ACCEL_SL])
    l_tvel  = temporal_velocity_loss(pred, target)
    l_tacc  = temporal_acceleration_loss(pred, target)
    l_beat  = beat_alignment_loss(pred, audio)
    l_mtgt  = motion_target_loss(pred, target)

    loss = (
        W_XYZ * l_xyz + W_VEL_CH * l_vel + W_ACCEL_CH * l_acc
        + W_TEMP_VEL * l_tvel + W_TEMP_ACCEL * l_tacc + W_BEAT * l_beat
        + W_MOTION_TARGET * l_mtgt
    )
    motion_ratio = motion_ratio_batch(pred, target).mean().item()
    metrics = {
        "xyz": l_xyz.item(),
        "vel_ch": l_vel.item(),
        "acc_ch": l_acc.item(),
        "t_vel": l_tvel.item(),
        "t_acc": l_tacc.item(),
        "beat": l_beat.item(),
        "motion_tgt": l_mtgt.item(),
        "motion": batch_motion_energy(pred).mean().item(),
        "motion_ratio": motion_ratio,
    }
    return loss, metrics


def scheduled_sampling_prob(epoch: int) -> float:
    if epoch < SS_START:
        return 1.0
    if epoch >= SS_END:
        return 1.0 - SS_MAX
    ramp = (epoch - SS_START) / (SS_END - SS_START)
    return 1.0 - SS_MAX * ramp


def training_forward(
    model: Music2PoseTransformer,
    audio: torch.Tensor,
    poses_tgt: torch.Tensor,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass with optional rolling-window crop and scheduled sampling.
    Returns (pred, poses_tgt, audio) — all possibly cropped to the same T.
    """
    B, T, D = poses_tgt.shape

    if model.training and T > CONTEXT_LEN and torch.rand(1).item() < CTX_TRAIN_PROB:
        start = torch.randint(0, T - CONTEXT_LEN + 1, (1,)).item()
        end = start + CONTEXT_LEN
        audio = audio[:, start:end]
        poses_tgt = poses_tgt[:, start:end]

    zeros = torch.zeros(B, 1, D, device=audio.device, dtype=audio.dtype)
    poses_in = torch.cat([zeros, poses_tgt[:, :-1]], dim=1)

    if model.training and torch.rand(1).item() > scheduled_sampling_prob(epoch):
        with torch.no_grad():
            teacher_pred = model(audio, poses_in)
        poses_in = torch.cat([zeros, teacher_pred[:, :-1]], dim=1)

    return model(audio, poses_in), poses_tgt, audio


def train(
    dataset_path: str = DATASET_PATH,
    epochs: int = None,
    resume: str = None,
    target_motion: float = TARGET_MOTION_RATIO,
):
    global TARGET_MOTION_RATIO
    TARGET_MOTION_RATIO = target_motion
    n_epochs = epochs if epochs is not None else EPOCHS
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    resume_ckpt = None
    if resume:
        if not os.path.isfile(resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        resume_ckpt = torch.load(resume, map_location=device, weights_only=False)
        print(f"Resuming from {resume} (completed epoch {resume_ckpt['epoch']})")

    # ── Data ────────────────────────────────────────────────────────────────
    dd = load_from_disk(dataset_path)
    train_hf = dd["train"]
    val_hf   = dd["validation"]
    print(f"Train: {len(train_hf)} chunks | Val: {len(val_hf)} chunks")

    norm_cache = os.path.join(CKPT_DIR, "norm_stats.npz")
    if resume_ckpt is not None:
        x_mean = resume_ckpt["x_mean"]
        x_std  = resume_ckpt["x_std"]
        y_mean = resume_ckpt["y_mean"]
        y_std  = resume_ckpt["y_std"]
        np.savez(norm_cache, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
        print("Loaded normalisation stats from checkpoint.")
    elif os.path.exists(norm_cache):
        norm = np.load(norm_cache)
        x_mean, x_std, y_mean, y_std = norm["x_mean"], norm["x_std"], norm["y_mean"], norm["y_std"]
        print("Loaded normalisation stats from cache.")
    else:
        x_mean, x_std, y_mean, y_std = compute_norm_stats(train_hf)
        np.savez(norm_cache, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    train_ds = DanceDataset(train_hf, x_mean, x_std, y_mean, y_std)
    val_ds   = DanceDataset(val_hf,   x_mean, x_std, y_mean, y_std)

    # num_workers=0 on Colab/Windows to avoid multiprocessing issues with librosa
    n_workers = int(os.environ.get("DATALOADER_WORKERS", 0))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=n_workers, pin_memory=True)

    if resume_ckpt is not None:
        audio_dim = resume_ckpt["audio_dim"]
        pose_dim  = resume_ckpt["pose_dim"]
        seq_len   = resume_ckpt["config"]["max_seq_len"] - 10
    else:
        sample_audio, sample_poses_in, _ = train_ds[0]
        audio_dim = sample_audio.shape[-1]    # 82
        pose_dim  = sample_poses_in.shape[-1] # 297
        seq_len   = sample_audio.shape[0]     # 250
    print(f"audio_dim={audio_dim}  pose_dim={pose_dim}  seq_len={seq_len}")

    # ── Model ────────────────────────────────────────────────────────────────
    model_cfg = resume_ckpt["config"] if resume_ckpt is not None else dict(
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        dropout=DROPOUT, max_seq_len=seq_len + 10,
    )
    model = Music2PoseTransformer(
        audio_dim=audio_dim,
        pose_dim=pose_dim,
        **model_cfg,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    start_epoch = 1
    best_val = float("inf")
    best_motion = 0.0
    best_balanced = float("inf")
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_val = resume_ckpt.get("best_val", float("inf"))
        best_motion = resume_ckpt.get("best_motion", 0.0)
        best_balanced = resume_ckpt.get("best_balanced", float("inf"))
        if start_epoch > n_epochs:
            print(f"Checkpoint already at epoch {resume_ckpt['epoch']} "
                  f"(>= --epochs {n_epochs}). Nothing to do.")
            return

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if resume_ckpt is not None and "optimizer_state" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        print("Restored optimizer state.")
    elif resume_ckpt is not None:
        for pg in optimizer.param_groups:
            pg["initial_lr"] = LR
        print("No optimizer state in checkpoint — using fresh optimizer with restored weights.")

    last_epoch = resume_ckpt["epoch"] - 1 if resume_ckpt is not None else -1
    scheduler = CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=LR * 0.01, last_epoch=last_epoch,
    )

    print(f"Balanced target: {100 * TARGET_MOTION_RATIO:.0f}% motion  "
          f"(band {100 * MOTION_RATIO_LOW:.0f}–{100 * MOTION_RATIO_HIGH:.0f}%)")

    if resume_ckpt is not None:
        best_str = f"{best_val:.4f}" if best_val != float("inf") else "unknown"
        motion_str = f"{100 * best_motion:.0f}%" if best_motion > 0 else "unknown"
        bal_str = f"{best_balanced:.4f}" if best_balanced != float("inf") else "unknown"
        print(f"Continuing from epoch {start_epoch} → {n_epochs}  "
              f"(best val: {best_str}, best motion: {motion_str}, best balanced: {bal_str})")

    for epoch in range(start_epoch, n_epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_motion = 0.0
        for audio, poses_in, poses_tgt in tqdm(train_loader,
                                               desc=f"Ep {epoch:03d} train", leave=False):
            audio, _, poses_tgt = (t.to(device) for t in (audio, poses_in, poses_tgt))
            pred, poses_tgt, audio = training_forward(model, audio, poses_tgt, epoch)

            loss, metrics = compute_loss(pred, poses_tgt, audio)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_motion += metrics["motion"]

        scheduler.step()
        train_loss /= len(train_loader)
        train_motion /= len(train_loader)

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_motion = 0.0
        val_motion_ratio = 0.0
        with torch.no_grad():
            for audio, poses_in, poses_tgt in val_loader:
                audio, _, poses_tgt = (t.to(device) for t in (audio, poses_in, poses_tgt))
                zeros = torch.zeros(
                    poses_tgt.size(0), 1, poses_tgt.size(-1),
                    device=device, dtype=poses_tgt.dtype,
                )
                poses_in = torch.cat([zeros, poses_tgt[:, :-1]], dim=1)
                pred = model(audio, poses_in)
                loss, metrics = compute_loss(pred, poses_tgt, audio)
                val_loss += loss.item()
                pred_energy = batch_motion_energy(pred)
                tgt_energy  = batch_motion_energy(poses_tgt)
                val_motion += pred_energy.mean().item()
                val_motion_ratio += (pred_energy / (tgt_energy + 1e-6)).mean().item()
        val_loss /= len(val_loader)
        val_motion /= len(val_loader)
        val_motion_ratio /= len(val_loader)

        epoch_balanced = balanced_score(val_loss, val_motion_ratio)
        ss_p = scheduled_sampling_prob(epoch)
        print(f"Epoch {epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"motion%={100*val_motion_ratio:.1f}  balanced={epoch_balanced:.4f}  "
              f"ss_p_gt={ss_p:.2f}  lr={scheduler.get_last_lr()[0]:.2e}")

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        is_best_motion = val_motion_ratio > best_motion
        if is_best_motion:
            best_motion = val_motion_ratio

        is_best_balanced = epoch_balanced < best_balanced
        if is_best_balanced:
            best_balanced = epoch_balanced

        # ── Save ─────────────────────────────────────────────────────────────
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "audio_dim": audio_dim,
            "pose_dim":  pose_dim,
            "x_mean": x_mean, "x_std": x_std,
            "y_mean": y_mean, "y_std": y_std,
            "best_val": best_val,
            "best_motion": best_motion,
            "best_balanced": best_balanced,
            "val_motion_ratio": val_motion_ratio,
            "target_motion_ratio": TARGET_MOTION_RATIO,
            "config": model_cfg,
        }
        if is_best:
            torch.save(ckpt, os.path.join(CKPT_DIR, "best.pt"))

        if is_best_motion:
            torch.save(ckpt, os.path.join(CKPT_DIR, "best_motion.pt"))

        if is_best_balanced:
            torch.save(ckpt, os.path.join(CKPT_DIR, "best_balanced.pt"))

        if epoch % SAVE_EVERY == 0:
            torch.save(ckpt, os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pt"))

    print(f"\nDone. Best val: {best_val:.4f}  |  Best motion: {100*best_motion:.1f}%  "
          f"|  Best balanced: {best_balanced:.4f}")
    print(f"Checkpoints → {CKPT_DIR}/best_balanced.pt (recommended)  |  "
          f"{CKPT_DIR}/best.pt  |  {CKPT_DIR}/best_motion.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=DATASET_PATH)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs (default: 200)")
    parser.add_argument("--resume", default=None,
                        help="Path to .pt checkpoint to resume from (e.g. checkpoints/epoch_180.pt)")
    parser.add_argument("--target_motion", type=float, default=TARGET_MOTION_RATIO,
                        help="Target motion ratio for balanced checkpoint (default: 0.32 = 32%% of GT)")
    args = parser.parse_args()
    train(args.dataset_path, epochs=args.epochs, resume=args.resume,
          target_motion=args.target_motion)
