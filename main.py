# -*- coding: utf-8 -*-
"""
Main entry point for PatchTST model training and evaluation
"""

import config
import data_loader
import train

if __name__ == "__main__":
    print(f"Using CSV: {config.CSV_PATH.name} | Device: {config.DEVICE}")
    print(f"USE_EXOG = '{config.USE_EXOG}'  (auto-detects vaccine/resp columns)")
    X, y, labels, feat_names = data_loader.load_and_prepare(config.CSV_PATH, config.USE_EXOG)
    print(f"Data points: {len(y)} | Features used ({len(feat_names)}): {feat_names}")
    model, X_va_sc, y_va_sc, X_te_sc, y_te_sc, scaler_y, feat_names, fi_df = train.train_and_eval(
        X, y, labels, feat_names, compute_fi=True, save_fi=True
    )
    
    if fi_df is not None:
        print("\n=== [결과 요약] ===")
        print(f"Feature 개수: {len(feat_names)}")
        print("\n[Top 10 Feature Importance]")
        print(fi_df.head(10).to_string(index=False))
