from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import HIST_DATASET, LIVE_DATASET, MAX_ROWS_TEXT
from src.load_data import enforce_schema


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _thin_df(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if df is None or df.empty or len(df) <= limit:
        return df
    idx = np.linspace(0, len(df) - 1, num=limit, dtype=int)
    return df.iloc[idx].copy()


def load_historico_dataset(thin: bool = False) -> Tuple[pd.DataFrame, str, Path]:
    path = Path(HIST_DATASET)
    df = _safe_read_parquet(path)
    if df.empty:
        return df, f"Histórico: no disponible ({path})", path

    df = enforce_schema(df, min_text_len=20, drop_duplicates=True)
    if thin:
        df = _thin_df(df, int(MAX_ROWS_TEXT))

    label = f"Histórico: {path.name} · filas={len(df):,}"
    return df, label, path


def load_live_dataset(thin: bool = False) -> Tuple[pd.DataFrame, str, Path]:
    path = Path(LIVE_DATASET)
    df = _safe_read_parquet(path)
    if df.empty:
        return df, f"Live: no disponible ({path})", path

    df = enforce_schema(df, min_text_len=20, drop_duplicates=True)
    if thin:
        df = _thin_df(df, int(MAX_ROWS_TEXT))

    label = f"Live: {path.name} · filas={len(df):,}"
    return df, label, path
