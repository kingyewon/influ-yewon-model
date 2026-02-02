# -*- coding: utf-8 -*-
"""
PatchTST model architecture
"""

import math
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class PatchTSTDataset(Dataset):
    """Multivariate X (N,F) + y (N,) -> (patchified) windows."""
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len:int, pred_len:int, patch_len:int, stride:int):
        assert len(X) == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len, self.pred_len = seq_len, pred_len
        self.patch_len, self.stride = patch_len, stride
        max_start = len(self.y) - (seq_len + pred_len)
        self.indices = list(range(max(0, max_start + 1)))

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        seq_X = self.X[i:i+self.seq_len, :]                      # (L, F)
        tgt_y = self.y[i+self.seq_len:i+self.seq_len+self.pred_len]  # (H,)

        # patchify along time axis
        patches = []
        pos = 0
        while pos + self.patch_len <= self.seq_len:
            patches.append(seq_X[pos:pos+self.patch_len, :])     # (patch_len, F)
            pos += self.stride
        X_patch = np.stack(patches, axis=0)                      # (P, patch_len, F)
        return torch.from_numpy(X_patch).float(), torch.from_numpy(tgt_y).float(), i

class MultiScaleCNNPatchEmbed(nn.Module):
    """
    (B, P, L, F) -> [각 패치] 멀티스케일 Conv1d 분기(k=2/3/5, 또 하나는 dilation=2) → GAP → (B, P, D)
    - 분기 4개 출력 concat → D_MODEL
    - 패치 내부의 급격/완만/잔진동 패턴을 동시에 포착
    """
    def __init__(self, in_features: int, patch_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % 4 == 0, "d_model은 4의 배수가 되어야 멀티스케일 분기 합산이 맞습니다."
        out_ch = d_model // 4
        # 커널 크기를 patch_len에 비례하게 설정
        self.b2 = nn.Conv1d(in_features, out_ch, kernel_size=1, padding=0)
        self.b3 = nn.Conv1d(in_features, out_ch, kernel_size=3, padding=1)
        self.b5 = nn.Conv1d(in_features, out_ch, kernel_size=5, padding=2)
        self.bd = nn.Conv1d(in_features, out_ch, kernel_size=3, padding=2, dilation=2)

        self.bn   = nn.BatchNorm1d(d_model)
        self.act  = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)   # (B*P, D, L) → (B*P, D, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, P, L, F)
        B, P, L, F = x.shape
        x = x.view(B*P, L, F).permute(0, 2, 1)        # (B*P, F, L)

        z = torch.cat([self.b2(x), self.b3(x), self.b5(x), self.bd(x)], dim=1)  # (B*P, D, L)
        z = self.act(self.bn(z))
        z = self.pool(z).squeeze(-1)                  # (B*P, D)
        z = self.drop(z)
        return z.view(B, P, -1)                       # (B, P, D)

class TokenConvMixer(nn.Module):
    """
    패치 토큰 간(P 축) 로컬 연속성 강화: DepthwiseConv1d(P-축) + PointwiseConv1d
    입력/출력: (B, P, D)
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, z):              # (B, P, D)
        y = z.permute(0, 2, 1)         # (B, D, P)
        y = self.dw(y)
        y = self.pw(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.drop(y)
        y = y.permute(0, 2, 1)         # (B, P, D)
        return z + y                   # Residual

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div)
        if d_model % 2 == 1:
            pe[:,1::2] = torch.cos(position*div)[:, :pe[:,1::2].shape[1]]
        else:
            pe[:,1::2] = torch.cos(position*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        P = x.size(1)
        return x + self.pe[:, :P, :]

class AttnPool(nn.Module):
    """Learnable-query attention pooling over patch tokens."""
    def __init__(self, d_model:int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model))
        self.proj = nn.Linear(d_model, d_model, bias=False)
    def forward(self, z):           # z: (B, P, D)
        B,P,D = z.shape
        q = self.q.expand(B, -1, -1)                       # (B,1,D)
        k = self.proj(z)                                   # (B,P,D)
        attn = torch.softmax((q @ k.transpose(1,2)) / (D**0.5), dim=-1)  # (B,1,P)
        pooled = attn @ z                                  # (B,1,D)
        return pooled.squeeze(1)                           # (B,D)

class PatchTSTModel(nn.Module):
    def __init__(self, in_features:int, patch_len:int, d_model:int, n_heads:int,
                 n_layers:int, ff_dim:int, dropout:float, pred_len:int, head_hidden:List[int]):
        super().__init__()
        # ① 멀티스케일 CNN 패치 임베딩
        self.embed = MultiScaleCNNPatchEmbed(in_features, patch_len, d_model, dropout=dropout*0.5)
        # ② 패치 토큰 간 로컬 연속성 믹서
        self.mixer = nn.Sequential(
            TokenConvMixer(d_model, dropout=dropout),
            TokenConvMixer(d_model, dropout=dropout),
        )
        # ③ PatchTST 인코더
        self.posenc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool = AttnPool(d_model)

        # ④ 예측 헤드
        mlp, in_dim = [], d_model
        for h in head_hidden[:2]:
            mlp += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        mlp.append(nn.Linear(in_dim, pred_len))
        self.head = nn.Sequential(*mlp)

    def forward(self, x):
        # x: (B, P, L, F)
        z = self.embed(x)      # (B,P,D)
        z = self.mixer(z)      # (B,P,D)
        z = self.posenc(z)
        z = self.encoder(z)
        z = self.pool(z)       # (B,D)
        return self.head(z)    # (B,H)

    def correlation_loss(pred, true):
        # pred, true: (B, H)
        pred = pred - pred.mean(dim=1, keepdim=True)
        true = true - true.mean(dim=1, keepdim=True)
        corr = (pred * true).sum(dim=1) / (
            (pred.norm(dim=1) * true.norm(dim=1)) + 1e-6
        )
        return 1 - corr.mean()
