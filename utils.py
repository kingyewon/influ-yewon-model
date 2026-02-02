# -*- coding: utf-8 -*-
"""
Utility functions for PatchTST model
"""

import random
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import config

def _iso_weeks_in_year(y: int) -> int:
    # ISO 달력의 마지막 주 번호(52 또는 53)
    return date(y, 12, 28).isocalendar().week

def weekly_to_daily_interp(
    df: pd.DataFrame,
    season_col: str = "season_norm",
    week_col: str = "week",
    target_col: str = "ili",
) -> pd.DataFrame:
    """
    주 단위 데이터를 일 단위로 확장(선형보간). season/week 없으면 label에서 추출하거나,
    최후에는 연속 주차를 생성해 보간합니다.
    반환: date 컬럼 포함한 일 단위 DF
    """
    df = df.copy()
    df.columns = df.columns.str.replace("\ufeff", "", regex=True).str.strip()

    # --- 시즌/주차 확보 ---
    has_season = season_col in df.columns
    has_week   = week_col in df.columns

    if not (has_season and has_week):
        # label에서 시즌/주차 추출 시도: "2024-2025 season - W29"
        if "label" in df.columns:
            import re
            def _parse_label(lbl):
                m = re.search(r"(\d{4}-\d{4}).*W\s*([0-9]+)", str(lbl))
                if m:
                    return m.group(1), int(m.group(2))
                return None
            parsed = df["label"].map(_parse_label)
            if not has_season:
                df[season_col] = [p[0] if p else np.nan for p in parsed]
                has_season = True
            if not has_week:
                df[week_col] = [p[1] if p else np.nan for p in parsed]
                has_week = True

    # 최후의 수단: season_norm이 없으면 단일 시즌으로, week 없으면 1..N
    if not has_season:
        # 첫 행의 연도를 찾아 대체 시즌명 만들기
        # 없으면 "0000-0001"
        first_year = None
        if "date" in df.columns:
            try:
                first_year = pd.to_datetime(df["date"]).dt.year.min()
            except Exception:
                pass
        if first_year is None:
            first_year = pd.Timestamp.today().year
        df[season_col] = f"{first_year}-{first_year+1}"
        has_season = True

    if not has_week:
        df[week_col] = np.arange(1, len(df) + 1, dtype=int)
        has_week = True

    # 숫자화
    df[week_col] = pd.to_numeric(df[week_col], errors="coerce")
    # 시즌 문자열 정규화
    def _norm_season_text_local(s: str) -> str:
        ss = str(s).replace("절기", "")
        import re
        m = re.search(r"(\d{4})\s*-\s*(\d{4})", ss)
        return f"{m.group(1)}-{m.group(2)}" if m else ss.strip()
    df[season_col] = df[season_col].astype(str).map(_norm_season_text_local)

    # --- ISO 주 시작일 산출 (시즌 규칙 반영) ---
    week_starts = []
    for _, row in df.iterrows():
        season = str(row[season_col])
        try:
            y0 = int(season.split("-")[0])
        except Exception:
            y0 = pd.Timestamp.today().year
        wk = int(row[week_col]) if not pd.isna(row[week_col]) else 1
        iso_year = y0 if wk >= 36 else (y0 + 1)
        # 해당 ISO년의 실제 마지막 주 넘지 않도록 보정
        wk = min(max(1, wk), _iso_weeks_in_year(iso_year))
        # 월요일(1) 기준 주 시작일
        week_starts.append(pd.Timestamp.fromisocalendar(iso_year, wk, 1))
    df["week_start"] = week_starts

    # --- 중복 week_start 처리: 수치=mean, 비수치=first ---
    if df["week_start"].duplicated().any():
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        agg = {c: "mean" for c in num_cols}
        # 비수치 컬럼(라벨/시즌 등)은 첫 값 유지
        for c in df.columns:
            if c not in num_cols and c != "week_start":
                agg[c] = "first"
        df = df.groupby("week_start", as_index=False).agg(agg)

    # --- 일 단위 리샘플 ---
    df = df.set_index("week_start").sort_index()
    df_daily = df.resample("D").asfreq()

    # 수치형은 선형보간
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        df_daily[c] = df_daily[c].interpolate(method="linear", limit_direction="both")

    # 범주형은 앞뒤 채움
    cat_cols = [c for c in df.columns if c not in num_cols]
    for c in cat_cols:
        df_daily[c] = df_daily[c].ffill().bfill()

    # 결과
    out = df_daily.reset_index().rename(columns={"week_start": "date"})
    # date는 datetime으로 강제
    out["date"] = pd.to_datetime(out["date"])
    return out
    
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_csv_kor(path: Path) -> pd.DataFrame:
    for enc in ["euc-kr", "cp949", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, encoding="utf-8", errors="replace")

def make_splits(n: int, train_ratio=0.7, val_ratio=0.15):
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return (0, n_train), (n_train, n_train+n_val), (n_train+n_val, n)

def get_scaler(name=None):
    s = (name or config.SCALER_TYPE).lower()
    if s == "robust":  return RobustScaler()
    if s == "minmax":  return MinMaxScaler()
    return StandardScaler()

def _norm_season_text(s: str) -> str:
    ss = str(s).replace("절기", "")
    import re
    m = re.search(r"(\d{4})\s*-\s*(\d{4})", ss)
    return f"{m.group(1)}-{m.group(2)}" if m else ss.strip()
