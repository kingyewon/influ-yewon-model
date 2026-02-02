# -*- coding: utf-8 -*-
"""
Data loading and preparation functions
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import utils
import config

def load_and_prepare(csv_path: Path, use_exog: str = "auto") -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    Returns:
        X: (N, F) features (first column should be 'ili' to align with univariate fallback)
        y: (N,) target (ili)
        labels: list[str] for plotting ticks
        used_feat_names: list[str] feature column names (len=F)
    """
    df = utils.read_csv_kor(csv_path).copy()
    df = utils.weekly_to_daily_interp(df, season_col="season_norm", week_col="week", target_col="ili")
    # 정렬: 주→일 변환 후에는 date 기준으로만 정렬
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
    else:
        # (극히 드문 fallback) date가 없을 때만 기존 로직
        if {"season_norm", "week"}.issubset(df.columns):
            df["season_norm"] = df["season_norm"].astype(str).map(utils._norm_season_text)
            df["week"] = pd.to_numeric(df["week"], errors="coerce")
            df = df.sort_values(["season_norm", "week"]).copy()
        elif "label" in df.columns:
            df = df.sort_values(["label"]).copy()

    # 타깃
    if "ili" not in df.columns:
        raise ValueError("CSV에 'ili' 컬럼이 없습니다.")
    df["ili"] = pd.to_numeric(df["ili"], errors="coerce")
    if df["ili"].isna().any():
        df["ili"] = df["ili"].interpolate(method="linear", limit_direction="both").fillna(df["ili"].median())
    
    # --- ✅ Seasonality feature 추가 ---
    if "week" in df.columns:
        df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52.0)
        df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52.0)
    else:
        df["week_sin"] = 0.0
        df["week_cos"] = 0.0

    # --- ✅ Alias 매핑 ---
    if "case_count" in df.columns and "respiratory_index" not in df.columns:
        df["respiratory_index"] = df["case_count"]

    # 기후 피처 후보
    climate_feats = []
    if "wx_week_avg_temp" in df.columns:     climate_feats.append("wx_week_avg_temp")
    if "wx_week_avg_rain" in df.columns:     climate_feats.append("wx_week_avg_rain")
    if "wx_week_avg_humidity" in df.columns: climate_feats.append("wx_week_avg_humidity")

    # 외생 후보 존재 여부
    has_vax  = "vaccine_rate" in df.columns
    has_resp = "respiratory_index" in df.columns

    # 어떤 특징을 쓸지 결정
    mode = use_exog.lower()
    if mode == "auto":
        chosen = ["ili"]
        if has_vax:  chosen.append("vaccine_rate")
        if has_resp: chosen.append("respiratory_index")
        chosen += climate_feats
    elif mode == "none":
        chosen = ["ili"]
    elif mode == "vax":
        chosen = ["ili"] + (["vaccine_rate"] if has_vax else [])
    elif mode == "resp":
        chosen = ["ili"] + (["respiratory_index"] if has_resp else [])
    elif mode == "both":
        chosen = ["ili"]
        if has_vax:  chosen.append("vaccine_rate")
        if has_resp: chosen.append("respiratory_index")
        chosen += climate_feats
    elif mode == "climate":
        chosen = ["ili"] + climate_feats
    elif mode == "all":
        chosen = ["ili"]
        if has_vax:  chosen.append("vaccine_rate")
        if has_resp: chosen.append("respiratory_index")
        chosen += climate_feats
    else:
        raise ValueError(f"Unknown USE_EXOG mode: {use_exog}")

    # 숫자화 & 보간
    for c in chosen:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            df[c] = df[c].interpolate(method="linear", limit_direction="both").fillna(df[c].median())

    # 라벨
    if "label" in df.columns and df["label"].notna().any():
        labels = df["label"].astype(str).tolist()
    elif {"season_norm","week"}.issubset(df.columns):
        labels = (df["season_norm"].astype(str) + " season - W" + df["week"].astype(int).astype(str)).tolist()
    else:
        labels = [f"idx_{i}" for i in range(len(df))]

    # X, y 구성
    feat_names = chosen[:]
    if config.INCLUDE_SEASONAL_FEATS and {"week_sin", "week_cos"}.issubset(df.columns):
        feat_names += ["week_sin", "week_cos"]

    # 선택된 입력 피처 로그 찍기
    print("[Data] Exogenous detected -> vaccine_rate:", has_vax, "| respiratory_index:", has_resp, "| climate_feats:", climate_feats)
    print("[Data] Selected feature columns (order) ->", feat_names)

    X = df[feat_names].to_numpy(dtype=float)
    y = df["ili"].to_numpy(dtype=float)
    return X, y, labels, feat_names
