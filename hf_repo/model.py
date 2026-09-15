"""
model.py — Music-conditioned causal transformer for pose generation.

Architecture:
  - Audio encoder: linear projection of mel features → d_model
  - Pose embedding: linear projection of previous pose → d_model
  - Causal decoder: GPT-style transformer (masked self-attention)
    attends to both past poses and current+past audio via cross-attention
  - Output head: linear → pose_dim

At inference: autoregressive — feed one frame at a time.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPE(nn.Module):
    """Standard fixed sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class Music2PoseTransformer(nn.Module):
    """
    Args
    ----
    audio_dim   : number of audio features per frame  (e.g. 82 = 80 mel + 2)
    pose_dim    : flattened pose vector size           (e.g. 99 = 33 kpts × 3)
    d_model     : transformer hidden size
    nhead       : attention heads
    num_layers  : decoder layers
    dropout     : dropout probability
    max_seq_len : maximum sequence length during training
    """
    def __init__(
        self,
        audio_dim:   int = 82,    # 80 mel + onset + beat = 82
        pose_dim:    int = 297,   # 33 landmarks × 9 channels (xyz + vel + accel)
        d_model:     int = 256,
        nhead:       int = 8,
        num_layers:  int = 6,
        dropout:     float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.pose_dim  = pose_dim
        self.d_model   = d_model

        # ── Encoders ────────────────────────────────────────────────────────
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_enc = SinusoidalPE(d_model, max_len=max_seq_len)

        # ── Causal Decoder ───────────────────────────────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm (more stable)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # ── Output ───────────────────────────────────────────────────────────
        self.out_head = nn.Linear(d_model, pose_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask so position i cannot attend to j > i."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(
        self,
        audio: torch.Tensor,   # (B, T, audio_dim)
        poses: torch.Tensor,   # (B, T, pose_dim)  — teacher-forced targets shifted right
    ) -> torch.Tensor:         # (B, T, pose_dim)  — predicted poses
        T = audio.size(1)

        # Memory: audio features (encoder side of cross-attention)
        memory = self.pos_enc(self.audio_proj(audio))   # (B, T, d_model)

        # Target: previous poses (decoder input, shifted right by 1)
        tgt = self.pos_enc(self.pose_proj(poses))       # (B, T, d_model)

        causal_mask = self._causal_mask(T, audio.device)

        out = self.decoder(tgt, memory, tgt_mask=causal_mask)  # (B, T, d_model)
        return self.out_head(out)                               # (B, T, pose_dim)

    # ── Autoregressive inference (single step) ───────────────────────────────
    def step(
        self,
        audio_ctx: torch.Tensor,   # (1, T_ctx, audio_dim)  — full audio context so far
        pose_ctx:  torch.Tensor,   # (1, T_ctx, pose_dim)   — poses generated so far
    ) -> torch.Tensor:             # (1, pose_dim)           — next pose
        """Return the predicted pose for the NEXT frame given context."""
        with torch.no_grad():
            pred = self.forward(audio_ctx, pose_ctx)
        return pred[:, -1, :]   # last frame prediction
