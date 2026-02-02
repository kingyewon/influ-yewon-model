# -*- coding: utf-8 -*-
"""
Configuration file for PatchTST model
"""

from pathlib import Path

# =========================
# Paths & device
# =========================
BASE_DIR = Path.cwd()
# 우선순위로 탐색 (새 파일 -> 구 파일들)
CANDIDATE_CSVS = [
    BASE_DIR / "data/processed/3_merged_influenza_vaccine_respiratory_weather_filled.csv",
    BASE_DIR / "data/processed/3_merged_influenza_vaccine_respiratory_weather.csv",
]

def pick_csv_path():
    for p in CANDIDATE_CSVS:
        if p.exists():
            return p
    raise FileNotFoundError("No input CSV found among:\n" + "\n".join(map(str, CANDIDATE_CSVS)))

CSV_PATH = pick_csv_path()

def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = pick_device()
SEED   = 42

# =========================
# Hyperparameters
# =========================
EPOCHS      = 100
BATCH_SIZE  = 64        # 소규모 시계열에서도 안정적으로 학습되도록 약간 낮춤
SEQ_LEN     = 12
PRED_LEN    = 3
PATCH_LEN   = 4          # ← CNN이 최소 3~5 커널 적용 가능하도록 확대
STRIDE      = 1

D_MODEL     = 128        # 4의 배수(멀티스케일 분기 4개 합산)
N_HEADS     = 2
ENC_LAYERS  = 4
FF_DIM      = 128
DROPOUT     = 0.3        # 약간 강화
HEAD_HIDDEN = [64, 64]

LR              = 5e-4
WEIGHT_DECAY    = 5e-4
PATIENCE        = 60
WARMUP_EPOCHS   = 30

SCALER_TYPE     = "robust"   # 노이즈/꼬리값 대응에 유리 (원하면 "standard"로 변경)

# 외생 특징 사용 모드: "auto"|"none"|"vax"|"resp"|"both"
USE_EXOG        = "all"

OUT_CSV          = str(BASE_DIR / "data/results/ili_predictions.csv")
PLOT_LAST_WINDOW = str(BASE_DIR / "plot_last_window.png")
PLOT_TEST_RECON  = str(BASE_DIR / "plot_test_reconstruction.png")
PLOT_MA_CURVES   = str(BASE_DIR / "plot_ma_curves.png")

# overlap 재구성 가중치 (t+1을 조금 더 신뢰)
RECON_W_START, RECON_W_END = 2.0, 0.5

# --- Feature switches ---
INCLUDE_SEASONAL_FEATS = True   # week_sin, week_cos를 입력 피처에 포함할지
