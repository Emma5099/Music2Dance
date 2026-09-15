"""
generate_from_audio.py — Run inference on any audio file and render a skeleton video.

Usage:
    python generate_from_audio.py --audio MA.wav --out outputs/ma_dance.mp4
"""
import os, sys, argparse, subprocess
import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from model import Music2PoseTransformer
from pose_postprocess import ground_poses
from train import BASELINE_CKPT

# ── Constants (must match training) ──────────────────────────────────────────
AUDIO_SR     = 32_000
POSE_FPS     = 25
HOP_SAMPLES  = AUDIO_SR // POSE_FPS   # 1280
N_MELS       = 80
CONTEXT_LEN  = 60

SKELETON_EDGES = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),
]
FACE  = {0,1,2,3,4,5,6,7,8,9,10}
LEFT  = {11,13,15,23,25,27,29,31}
RIGHT = {12,14,16,24,26,28,30,32}

def seg_color(a, b):
    if a in FACE or b in FACE: return "#a0c4ff"
    if a in LEFT  and b in LEFT:  return "#ffd6a5"
    if a in RIGHT and b in RIGHT: return "#ffadad"
    return "#b9fbc0"

# ── Audio pipeline ────────────────────────────────────────────────────────────
def load_audio(path: str) -> np.ndarray:
    """Load any audio file → mono float32 at AUDIO_SR."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != AUDIO_SR:
        print(f"  Resampling {sr} Hz → {AUDIO_SR} Hz…")
        mono = librosa.resample(mono, orig_sr=sr, target_sr=AUDIO_SR)
    return mono.astype(np.float32)


def audio_to_features(y: np.ndarray) -> np.ndarray:
    """Raw waveform (N,) → (T, 82) feature matrix."""
    mel    = librosa.feature.melspectrogram(y=y, sr=AUDIO_SR, n_mels=N_MELS, hop_length=HOP_SAMPLES)
    mel_db = librosa.power_to_db(mel, ref=np.max)                   # (80, T)
    onset  = librosa.onset.onset_strength(y=y, sr=AUDIO_SR, hop_length=HOP_SAMPLES)  # (T,)
    _, beats = librosa.beat.beat_track(y=y, sr=AUDIO_SR, hop_length=HOP_SAMPLES)

    T = min(mel_db.shape[1], len(onset))
    mel_db = mel_db[:, :T]
    onset  = onset[:T]
    beat_sig = np.zeros(T, dtype=np.float32)
    beat_sig[beats[beats < T].astype(int)] = 1.0

    return np.concatenate([mel_db.T, onset[:, None], beat_sig[:, None]], axis=-1).astype(np.float32)


# ── Model + inference ─────────────────────────────────────────────────────────
def load_model(ckpt_path: str, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Music2PoseTransformer(
        audio_dim=ckpt["audio_dim"],
        pose_dim =ckpt["pose_dim"],
        **ckpt["config"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}  |  audio_dim={ckpt['audio_dim']}  pose_dim={ckpt['pose_dim']}")
    return model, ckpt


def run_inference(model, audio_feat, x_mean, x_std, y_mean, y_std, device) -> np.ndarray:
    """Autoregressive inference → (T, 33, 3) xyz poses."""
    T        = audio_feat.shape[0]
    af_norm  = (audio_feat - x_mean) / x_std
    af_t     = torch.tensor(af_norm, dtype=torch.float32, device=device)
    pose_dim = model.pose_dim

    audio_ctx = torch.zeros(1, CONTEXT_LEN, af_t.shape[-1], device=device)
    pose_ctx  = torch.zeros(1, CONTEXT_LEN, pose_dim,        device=device)
    preds     = []

    with torch.no_grad():
        for t in tqdm(range(T), desc="Generating poses", unit="frame"):
            audio_ctx        = torch.roll(audio_ctx, -1, dims=1)
            audio_ctx[0, -1] = af_t[t]
            next_pose        = model.step(audio_ctx, pose_ctx)[0]
            preds.append(next_pose.cpu().numpy())
            pose_ctx        = torch.roll(pose_ctx, -1, dims=1)
            pose_ctx[0, -1] = next_pose

    pred_norm = np.stack(preds)                             # (T, 297)
    pred_raw  = pred_norm * y_std + y_mean                  # (T, 297)
    return pred_raw.reshape(T, 33, 9)[:, :, :3]            # (T, 33, 3)


# ── Renderer ──────────────────────────────────────────────────────────────────
def render_skeleton_video(poses_xyz: np.ndarray, title: str, tmp_path: str):
    """Render skeleton animation to a silent mp4 (720×1080)."""
    T = poses_xyz.shape[0]

    vel    = np.diff(poses_xyz, axis=0)
    energy = np.concatenate([[0], np.linalg.norm(vel, axis=-1).mean(axis=-1)])
    energy = energy / (energy.max() + 1e-6)

    xmin = poses_xyz[:,:,0].min() - 0.15
    xmax = poses_xyz[:,:,0].max() + 0.15
    ymin = poses_xyz[:,:,1].min() - 0.1
    ymax = poses_xyz[:,:,1].max() + 0.1

    fig = plt.figure(figsize=(5.4, 8), facecolor="#0d1117")
    ax  = fig.add_axes([0.05, 0.08, 0.90, 0.86])
    ax.set_facecolor("#0d1117")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-ymax, -ymin)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.text(0.5, 0.97, title, color="#e6edf3", fontsize=11,
             ha="center", va="top", fontweight="bold")
    bar_ax = fig.add_axes([0.05, 0.02, 0.90, 0.04])
    bar_ax.set_xlim(0, T); bar_ax.set_ylim(0, 1)
    bar_ax.axis("off")
    progress_bar = bar_ax.barh(0.5, 0, height=0.8, color="#3fb950", left=0)
    time_txt     = bar_ax.text(T * 0.5, 0.5, "0.0 s", color="white",
                               fontsize=7, ha="center", va="center")

    scat  = ax.scatter([], [], s=18, zorder=4)
    lines = [ax.plot([], [], lw=2.2, solid_capstyle="round")[0] for _ in SKELETON_EDGES]

    def init():
        scat.set_offsets(np.empty((0,2)))
        for l in lines: l.set_data([], [])
        return [scat, *lines]

    def update(t):
        kpts = poses_xyz[t]
        xs, ys = kpts[:,0], -kpts[:,1]

        cols = []
        for i in range(33):
            if i in FACE:  cols.append("#a0c4ff")
            elif i in LEFT: cols.append("#ffd6a5")
            elif i in RIGHT: cols.append("#ffadad")
            else: cols.append("#b9fbc0")
        scat.set_offsets(np.c_[xs, ys])
        scat.set_color(cols)

        for idx, (a, b) in enumerate(SKELETON_EDGES):
            lines[idx].set_data([xs[a], xs[b]], [ys[a], ys[b]])
            lines[idx].set_color(seg_color(a, b))
            lines[idx].set_alpha(0.85 + 0.15 * energy[t])

        progress_bar[0].set_width(t + 1)
        time_txt.set_text(f"{t / POSE_FPS:.1f} s / {T / POSE_FPS:.1f} s")
        return [scat, *lines, progress_bar[0], time_txt]

    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init,
                                  interval=1000/POSE_FPS, blit=True)
    writer = animation.FFMpegWriter(fps=POSE_FPS, bitrate=2000,
                                    extra_args=["-vcodec","libx264","-pix_fmt","yuv420p"])
    ani.save(tmp_path, writer=writer, dpi=150)
    plt.close(fig)
    print(f"  Silent video saved: {tmp_path}")


# ── Mix audio in ──────────────────────────────────────────────────────────────
def mux_audio(video_path: str, audio_path: str, out_path: str, duration: float):
    """Combine silent video + original audio → final mp4."""
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
    print(f"  Final video (with audio): {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",      required=True,  help="Input audio file (any format)")
    parser.add_argument("--checkpoint", default=BASELINE_CKPT)
    parser.add_argument("--out",        default="outputs/dance.mp4")
    parser.add_argument("--save_poses", default=None,
                        help="Optional path to save predicted poses as .npy (T,33,3)")
    parser.add_argument("--character_mesh", default=None,
                        help="Optional GLB to animate with 3Dmodel/animate_character.py")
    parser.add_argument("--character_out", default=None,
                        help="Output MP4 for character render (default: <out> with _character)")
    parser.add_argument("--ground", action="store_true",
                        help="Plant lowest foot on a floor plane (post-process; prefer training fixes)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print(f"\n[1/4] Loading audio: {args.audio}")
    waveform = load_audio(args.audio)
    duration = len(waveform) / AUDIO_SR
    print(f"  Duration: {duration:.1f}s  ({len(waveform)} samples @ {AUDIO_SR} Hz)")

    print("\n[2/4] Extracting audio features…")
    audio_feat = audio_to_features(waveform)
    T = audio_feat.shape[0]
    print(f"  Feature shape: {audio_feat.shape}  ({T} frames @ {POSE_FPS} fps)")

    print(f"\n[3/4] Loading model: {args.checkpoint}")
    model, ckpt = load_model(args.checkpoint, device)
    x_mean, x_std = ckpt["x_mean"], ckpt["x_std"]
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

    print(f"\n[4/4] Running inference ({T} frames)…")
    poses_xyz = run_inference(model, audio_feat, x_mean, x_std, y_mean, y_std, device)
    print(f"  Poses shape: {poses_xyz.shape}")

    if args.ground:
        poses_xyz = ground_poses(poses_xyz)
        print("  Grounding: planted lowest foot on median floor plane")

    if args.save_poses:
        os.makedirs(os.path.dirname(args.save_poses) or ".", exist_ok=True)
        np.save(args.save_poses, poses_xyz.astype(np.float32))
        print(f"  Saved poses: {args.save_poses}")

    title  = os.path.splitext(os.path.basename(args.audio))[0]
    tmp    = args.out.replace(".mp4", "_silent.mp4")
    print(f"\nRendering skeleton animation…")
    render_skeleton_video(poses_xyz, title=title, tmp_path=tmp)

    print(f"\nMixing audio…")
    mux_audio(tmp, args.audio, args.out, duration)
    os.remove(tmp)
    print(f"\nDone! → {args.out}")

    if args.character_mesh:
        from pathlib import Path
        char_out = args.character_out or args.out.replace(".mp4", "_character.mp4")
        poses_path = args.save_poses or args.out.replace(".mp4", "_poses.npy")
        if not args.save_poses:
            np.save(poses_path, poses_xyz.astype(np.float32))
        print(f"\nRendering character mesh → {char_out}")
        subprocess.run([
            sys.executable, "3Dmodel/animate_character.py",
            "--poses", poses_path,
            "--mesh", args.character_mesh,
            "--audio", args.audio,
            "--out", char_out,
        ], check=False)


if __name__ == "__main__":
    main()
