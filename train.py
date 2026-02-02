# -*- coding: utf-8 -*-
"""
Training and evaluation functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config
import utils
import model

def warmup_lr(ep:int, base_lr:float, warmup_epochs:int):
    if ep <= warmup_epochs:
        return base_lr * (ep / max(1, warmup_epochs))
    return base_lr

def batch_mae_in_original_units(pred_b: torch.Tensor, y_b: torch.Tensor, scaler_y) -> float:
    p = pred_b.detach().cpu().numpy().reshape(-1, 1)
    t = y_b.detach().cpu().numpy().reshape(-1, 1)
    p_orig = scaler_y.inverse_transform(p).reshape(-1)
    t_orig = scaler_y.inverse_transform(t).reshape(-1)
    return float(np.mean(np.abs(p_orig - t_orig)))

def batch_corrcoef(pred_b: torch.Tensor, y_b: torch.Tensor, scaler_y) -> float:
    """
    Pearson correlation coefficient (batch 평균)
    pred_b, y_b: (B, H)
    """
    p = pred_b.detach().cpu().numpy().reshape(-1, 1)
    t = y_b.detach().cpu().numpy().reshape(-1, 1)
    p_orig = scaler_y.inverse_transform(p).reshape(-1)
    t_orig = scaler_y.inverse_transform(t).reshape(-1)

    if np.std(p_orig) < 1e-6 or np.std(t_orig) < 1e-6:
        return 0.0
    return float(np.corrcoef(p_orig, t_orig)[0,1])

def train_and_eval(X: np.ndarray, y: np.ndarray, labels: list, feat_names: list, 
                   compute_fi: bool = False, save_fi: bool = False):
    """
    X: (N,F), y: (N,), feat_names: ['ili', 'vaccine_rate', 'respiratory_index'] 등
    compute_fi: True면 검증/테스트 기반 피처 중요도 계산 및 저장
    save_fi: True면 feature_importance.csv/png 저장
    """
    utils.set_seed(config.SEED)
    (s0,e0),(s1,e1),(s2,e2) = utils.make_splits(len(y))
    X_tr, X_va, X_te = X[s0:e0], X[s1:e1], X[s2:e2]
    y_tr, y_va, y_te = y[s0:e0], y[s1:e1], y[s2:e2]
    lab_tr, lab_va, lab_te = labels[s0:e0], labels[s1:e1], labels[s2:e2]

    # ==== Scaling ====
    # Target scaler
    scaler_y = utils.get_scaler()
    y_tr_sc = scaler_y.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_va_sc = scaler_y.transform(y_va.reshape(-1,1)).ravel()
    y_te_sc = scaler_y.transform(y_te.reshape(-1,1)).ravel()

    # Feature scaler (입력 특징 전체)
    scaler_x = utils.get_scaler()
    X_tr_sc = scaler_x.fit_transform(X_tr)
    X_va_sc = scaler_x.transform(X_va)
    X_te_sc = scaler_x.transform(X_te)

    F = X.shape[1]
    print(f"[Shapes] X_tr:{X_tr.shape}, X_va:{X_va.shape}, X_te:{X_te.shape} | F={F}")
    print(f"[Info] Model input feature order -> {feat_names}")

    ds_tr = model.PatchTSTDataset(X_tr_sc, y_tr_sc, config.SEQ_LEN, config.PRED_LEN, config.PATCH_LEN, config.STRIDE)
    ds_va = model.PatchTSTDataset(X_va_sc, y_va_sc, config.SEQ_LEN, config.PRED_LEN, config.PATCH_LEN, config.STRIDE)
    ds_te = model.PatchTSTDataset(X_te_sc, y_te_sc, config.SEQ_LEN, config.PRED_LEN, config.PATCH_LEN, config.STRIDE)

    # drop_last=False 로 변경(작은 데이터셋에서도 학습 배치 보장)
    dl_tr = DataLoader(ds_tr, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=config.BATCH_SIZE, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=config.BATCH_SIZE, shuffle=False)

    model_obj = model.PatchTSTModel(
        in_features=F, patch_len=config.PATCH_LEN, d_model=config.D_MODEL, n_heads=config.N_HEADS,
        n_layers=config.ENC_LAYERS, ff_dim=config.FF_DIM, dropout=config.DROPOUT,
        pred_len=config.PRED_LEN, head_hidden=config.HEAD_HIDDEN
    ).to(config.DEVICE)

    # Loss / Optim / Scheduler
    crit = nn.HuberLoss(delta=1.0)
    opt  = torch.optim.AdamW(model_obj.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.EPOCHS, eta_min=1e-5)

    # ---- history for curves ----
    hist = {"train_loss":[], "val_loss":[], "train_mae":[], "val_mae":[]}

    best_val = float("inf"); best_state=None; noimp=0
    printed_batch_info = False
    for ep in range(1, config.EPOCHS+1):
        # ---- Train ----
        model_obj.train(); tr_loss_sum=0; tr_mae_sum=0; n=0
        # warmup
        for g in opt.param_groups:
            g['lr'] = warmup_lr(ep, config.LR, config.WARMUP_EPOCHS)

        for Xb,yb,_ in dl_tr:
            if not printed_batch_info:
                # Xb: (B, P, L, F)  ← 최종 모델 입력 텐서 구조
                print(f"[Batch] Xb.shape={tuple(Xb.shape)} (B,P,L,F), yb.shape={tuple(yb.shape)}")
                print(f"[Batch] Feature order used -> {feat_names}")
                printed_batch_info = True
            Xb=Xb.to(config.DEVICE); yb=yb.to(config.DEVICE)
            opt.zero_grad()
            pred = model_obj(Xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model_obj.parameters(), 1.0)
            opt.step()
            bs=yb.size(0)
            tr_loss_sum += loss.item()*bs; n+=bs
            tr_mae_sum  += batch_mae_in_original_units(pred, yb, scaler_y)*bs

        tr_loss = tr_loss_sum / max(1,n)
        tr_mae  = tr_mae_sum  / max(1,n)

        # ---- Validation ----
        model_obj.eval(); va_loss_sum=0; va_mae_sum=0; va_corr_sum = 0; n=0
        with torch.no_grad():
            for Xb,yb,_ in dl_va:
                Xb=Xb.to(config.DEVICE); yb=yb.to(config.DEVICE)
                pred = model_obj(Xb); loss = crit(pred,yb)
                bs=yb.size(0)
                va_loss_sum += loss.item()*bs; n+=bs
                va_mae_sum  += batch_mae_in_original_units(pred, yb, scaler_y)*bs
                va_corr_sum += batch_corrcoef(pred, yb, scaler_y)*bs
        va_loss = va_loss_sum / max(1,n)
        va_mae  = va_mae_sum  / max(1,n)
        va_corr = va_corr_sum / max(1,n)

        scheduler.step()

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["train_mae"].append(tr_mae)
        hist["val_mae"].append(va_mae)

        print(f"[Epoch {ep:03d}/{config.EPOCHS}] "
              f"LR={opt.param_groups[0]['lr']:.6f} | "
              f"Loss T/V={tr_loss:.5f}/{va_loss:.5f} | "
              f"MAE  T/V={tr_mae:.5f}/{va_mae:.5f}"
              f"Corr V={va_corr:.3f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss; noimp=0
            best_state = {k:v.detach().cpu().clone() for k,v in model_obj.state_dict().items()}
        else:
            noimp += 1
            if noimp >= config.PATIENCE:
                print(f"Early stopping after {ep} epochs (no improvement {config.PATIENCE}).")
                break

    if best_state is not None:
        model_obj.load_state_dict({k:v.to(config.DEVICE) for k,v in best_state.items()})

    # ---- Test & Metrics ----
    model_obj.eval(); preds=[]; trues=[]; starts=[]
    with torch.no_grad():
        for Xb,yb,i0 in dl_te:
            Xb=Xb.to(config.DEVICE)
            preds.append(model_obj(Xb).detach().cpu().numpy())
            trues.append(yb.numpy())
            starts.append(i0.numpy())
    yhat_sc = np.concatenate(preds,axis=0)
    ytrue_sc= np.concatenate(trues,axis=0)
    starts  = np.concatenate(starts,axis=0)

    # inverse scale (target only)
    yhat  = scaler_y.inverse_transform(yhat_sc.reshape(-1,1)).reshape(-1,config.PRED_LEN)
    ytrue = scaler_y.inverse_transform(ytrue_sc.reshape(-1,1)).reshape(-1,config.PRED_LEN)

    mse  = float(np.mean((yhat-ytrue)**2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(yhat-ytrue)))
    print("\n=== Final Test Metrics ===")
    print(f"MSE : {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")

    # =========================
    # Save per-window predictions
    # =========================
    cols_true = [f"true_t+{i}" for i in range(1,config.PRED_LEN+1)]
    cols_pred = [f"pred_t+{i}" for i in range(1,config.PRED_LEN+1)]
    out = pd.DataFrame(np.hstack([ytrue, yhat]), columns=cols_true+cols_pred)
    out.to_csv(config.OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved predictions -> {config.OUT_CSV}")

    # =========================
    # Plot_1: last window (H-step ahead)
    # =========================
    last_true = ytrue[-1]; last_pred = yhat[-1]
    weeks = np.arange(1, config.PRED_LEN+1)
    plt.figure(figsize=(10,4))
    plt.plot(weeks, last_true, label="Truth (last window)", linewidth=2)
    plt.plot(weeks, last_pred, label="Prediction (last window)", linewidth=2)
    plt.title("Last Test Window: Truth vs Prediction")
    plt.xlabel("Horizon (weeks ahead)")
    plt.ylabel("ILI per 1,000 Population")
    plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(config.PLOT_LAST_WINDOW, dpi=150)
    print(f"Saved plot -> {config.PLOT_LAST_WINDOW}")

    # =========================
    # Plot_2: test reconstruction (val-context included)
    # =========================
    context = y_va_sc[-config.SEQ_LEN:]                       # 표준화 컨텍스트
    y_ct_sc = np.concatenate([context, y_te_sc])       # [SEQ_LEN + test_len]
    # 입력 특징도 컨텍스트 포함해 재구성 필요 → X도 동일하게 붙여서 예측
    X_ct_sc = np.concatenate([X_va_sc[-config.SEQ_LEN:], X_te_sc], axis=0)
    ds_ct = model.PatchTSTDataset(X_ct_sc, y_ct_sc, config.SEQ_LEN, config.PRED_LEN, config.PATCH_LEN, config.STRIDE)
    dl_ct = DataLoader(ds_ct, batch_size=config.BATCH_SIZE, shuffle=False)

    model_obj.eval(); preds_ct=[]; starts_ct=[]
    with torch.no_grad():
        for Xb, _, i0 in dl_ct:
            Xb = Xb.to(config.DEVICE)
            preds_ct.append(model_obj(Xb).detach().cpu().numpy())  # (B, H)
            starts_ct.append(i0.numpy())
    yhat_ct_sc = np.concatenate(preds_ct, axis=0)
    starts_ct  = np.concatenate(starts_ct, axis=0)
    yhat_ct = scaler_y.inverse_transform(yhat_ct_sc.reshape(-1,1)).reshape(-1, config.PRED_LEN)

    test_len = len(y_te)
    recon_sum   = np.zeros(test_len)
    recon_count = np.zeros(test_len)
    h_weights = np.linspace(config.RECON_W_START, config.RECON_W_END, config.PRED_LEN)

    for k, s in enumerate(starts_ct):
        pos0_ct = int(s) + config.SEQ_LEN   # [context+test] 축
        pos0_te = pos0_ct - config.SEQ_LEN  # test 축으로 변환
        for j in range(config.PRED_LEN):
            idx = pos0_te + j
            if 0 <= idx < test_len:
                w = h_weights[j]
                recon_sum[idx]   += yhat_ct[k, j] * w
                recon_count[idx] += w

    recon = np.where(recon_count > 0, recon_sum / np.maximum(1, recon_count), np.nan)

    truth_test = y_te
    x_labels = lab_te
    tick_step = max(1, test_len // 12)
    tick_idx  = list(range(0, test_len, tick_step))
    if tick_idx[-1] != test_len-1:
        tick_idx.append(test_len-1)
    tick_text = [x_labels[i] for i in tick_idx]

    plt.figure(figsize=(12,5))
    plt.plot(range(test_len), truth_test, linewidth=2, label="Truth (test segment)")
    plt.plot(range(test_len), recon,      linewidth=2, label="Prediction (overlap-avg, weighted)")
    plt.title("Test Range: Truth vs Overlap-averaged Prediction (with context)")
    plt.xlabel("Season - Week"); plt.ylabel("ILI per 1,000 Population")
    plt.xticks(tick_idx, tick_text, rotation=45, ha="right")
    plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(config.PLOT_TEST_RECON, dpi=150)
    print(f"Saved plot -> {config.PLOT_TEST_RECON}")

    # =========================
    # Plot_3: Train/Val MAE curves
    # =========================
    xs = np.arange(1, len(hist["train_mae"])+1)
    plt.figure(figsize=(10,4))
    plt.plot(xs, hist["train_mae"], linewidth=2, label="Train MAE (original units)")
    plt.plot(xs, hist["val_mae"],   linewidth=2, label="Val MAE (original units)")
    plt.title("Training Curves: MAE per epoch (lower is better)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (ILI per 1,000)")
    plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(config.PLOT_MA_CURVES, dpi=150)
    print(f"Saved plot -> {config.PLOT_MA_CURVES}")

    # =========================
    # Feature Importance
    # =========================
    if compute_fi:
        import feature_importance
        fi_df = feature_importance.compute_feature_importance(
            model_obj,
            X_va_sc, y_va_sc,
            X_te_sc, y_te_sc,
            scaler_y=scaler_y,
            feat_names=feat_names,
            random_state=config.SEED
        )
        if save_fi:
            feature_importance.save_feature_importance(
                fi_df,
                out_csv=str(config.BASE_DIR / "data/results/feature_importance.csv"),
                out_png=str(config.BASE_DIR / "feature_importance.png")
            )
        return model_obj, X_va_sc, y_va_sc, X_te_sc, y_te_sc, scaler_y, feat_names, fi_df
    
    return model_obj, X_va_sc, y_va_sc, X_te_sc, y_te_sc, scaler_y, feat_names, None
