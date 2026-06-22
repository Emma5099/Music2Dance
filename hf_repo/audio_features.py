"""Audio feature extraction for Music2Pose inference."""

import numpy as np
import librosa

AUDIO_SR = 32_000
POSE_FPS = 25
HOP_SAMPLES = AUDIO_SR // POSE_FPS  # 1280 samples per pose frame
N_MELS = 80
CONTEXT_LEN = 60


def audio_to_features(waveform: np.ndarray) -> np.ndarray:
    """Convert mono waveform (n_samples,) → (T, 82) mel + onset + beat features."""
    y = waveform.astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=y, sr=AUDIO_SR, n_mels=N_MELS, hop_length=HOP_SAMPLES
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    onset = librosa.onset.onset_strength(
        y=y, sr=AUDIO_SR, hop_length=HOP_SAMPLES
    )
    _, beats = librosa.beat.beat_track(
        y=y, sr=AUDIO_SR, hop_length=HOP_SAMPLES
    )

    T = min(mel_db.shape[1], len(onset))
    mel_db = mel_db[:, :T]
    onset = onset[:T]

    beat_sig = np.zeros(T, dtype=np.float32)
    beat_sig[beats[beats < T].astype(int)] = 1.0

    return np.concatenate(
        [mel_db.T, onset[:, None], beat_sig[:, None]], axis=-1
    ).astype(np.float32)
