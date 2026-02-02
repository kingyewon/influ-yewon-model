# -*- coding: utf-8 -*-
"""
Feature Importance computation functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import config
import model
import train

def _eval_mae_on_split(model_obj, X_split_sc, y_split_sc, scaler_y, feat_names, 
                       seq_len=config.SEQ_LEN, pred_len=config.PRED_LEN, patch_len=config.PATCH_LEN, stride=config.STRIDE,
                       batch_size=config.BATCH_SIZE):
    """현재 모델로 한 분할(va/test) 세트에서 MAE(원 단위) 계산"""
    ds = model.PatchTSTDataset(X_split_sc, y_split_sc, seq_len, pred_len, patch_len, stride)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model_obj.eval()
    mae_sum, n = 0.0, 0
    with torch.no_grad():
        for Xb, yb, _ in dl:
            Xb = Xb.to(config.DEVICE); yb = yb.to(config.DEVICE)
            pred = model_obj(Xb)  # (B, H)
            mae_sum += train.batch_mae_in_original_units(pred, yb, scaler_y) * yb.size(0)
            n += yb.size(0)
    return float(mae_sum / max(1, n))


def compute_feature_importance(model_obj, 
                               X_va_sc, y_va_sc, 
                               X_te_sc=None, y_te_sc=None,
                               scaler_y=None, feat_names=None, 
                               random_state=42):
    """
    퍼뮤테이션(열 섞기) 중요도와 평균 대체(그 특징을 평균으로 고정) 중요도를 계산.
    반환: 중요도 DataFrame (ΔMAE가 클수록 중요)
    """
    assert scaler_y is not None and feat_names is not None
    rng = np.random.RandomState(random_state)

    # --- 기준선(baseline MAE) ---
    baseline_val = _eval_mae_on_split(model_obj, X_va_sc, y_va_sc, scaler_y, feat_names)
    print(f"[FI] Baseline Val MAE: {baseline_val:.6f}")

    baseline_tst = None
    if X_te_sc is not None and y_te_sc is not None:
        baseline_tst = _eval_mae_on_split(model_obj, X_te_sc, y_te_sc, scaler_y, feat_names)
        print(f"[FI] Baseline Test MAE: {baseline_tst:.6f}")

    perm_deltas_val, mean_deltas_val = [], []
    perm_deltas_tst, mean_deltas_tst = [], []

    for j, name in enumerate(feat_names):
        # ① 퍼뮤테이션(열 섞기)
        Xp = X_va_sc.copy()
        col = Xp[:, j].copy()
        rng.shuffle(col)
        Xp[:, j] = col
        mae_perm_val = _eval_mae_on_split(model_obj, Xp, y_va_sc, scaler_y, feat_names)
        perm_deltas_val.append(mae_perm_val - baseline_val)

        # ② 평균 대체(특징 제거 효과)
        Xz = X_va_sc.copy()
        Xz[:, j] = X_va_sc[:, j].mean()
        mae_mean_val = _eval_mae_on_split(model_obj, Xz, y_va_sc, scaler_y, feat_names)
        mean_deltas_val.append(mae_mean_val - baseline_val)

        if X_te_sc is not None and y_te_sc is not None:
            Xp_te = X_te_sc.copy()
            col_te = Xp_te[:, j].copy()
            rng.shuffle(col_te)
            Xp_te[:, j] = col_te
            mae_perm_tst = _eval_mae_on_split(model_obj, Xp_te, y_te_sc, scaler_y, feat_names)
            perm_deltas_tst.append(mae_perm_tst - baseline_tst)

            Xz_te = X_te_sc.copy()
            Xz_te[:, j] = X_te_sc[:, j].mean()
            mae_mean_tst = _eval_mae_on_split(model_obj, Xz_te, y_te_sc, scaler_y, feat_names)
            mean_deltas_tst.append(mae_mean_tst - baseline_tst)

        print(f"[FI] {name:>20s} | ΔMAE(val) perm={perm_deltas_val[-1]:+.6f}  mean={mean_deltas_val[-1]:+.6f}")

    df = pd.DataFrame({
        "feature": feat_names,
        "delta_mae_val_perm": perm_deltas_val,
        "delta_mae_val_mean": mean_deltas_val,
    })
    if baseline_tst is not None:
        df["delta_mae_test_perm"] = perm_deltas_tst
        df["delta_mae_test_mean"] = mean_deltas_tst

    # ΔMAE가 클수록 중요 → 내림차순 정렬
    df = df.sort_values("delta_mae_val_perm", ascending=False).reset_index(drop=True)
    return df


def save_feature_importance(df: pd.DataFrame, out_csv="feature_importance.csv", out_png="feature_importance.png"):
    """중요도 테이블 저장 + 막대 그래프 저장"""
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[FI] Saved -> {out_csv}")

    top = min(20, len(df))
    plt.figure(figsize=(10, max(4, 0.4*top)))
    plt.barh(df["feature"][:top][::-1], df["delta_mae_val_perm"][:top][::-1])
    plt.title("Permutation Feature Importance (ΔMAE on Val)")
    plt.xlabel("ΔMAE (higher = more important)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[FI] Saved -> {out_png}")
