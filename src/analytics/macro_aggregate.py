# src/analytics/macro_aggregate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Data structures
# =============================================================================
@dataclass(frozen=True)
class MacroAgg:
    # Base long table
    df_macro: pd.DataFrame          # columns: period, macro_area, count, total_docs, rel_freq

    # Wide matrices
    wide_rel_freq: pd.DataFrame     # index=period, cols=macro_area, values=rel_freq
    wide_count: pd.DataFrame        # index=period, cols=macro_area, values=count

    # Diagnostics / metadata (useful for UI + debugging)
    meta: Dict[str, object]


# =============================================================================
# Internal helpers
# =============================================================================
_ALLOWED_FREQ = {"D", "W", "M"}

_DEFAULT_TEXT_COLS = ("text_clean", "text", "abstract", "summary", "title")


def _coerce_freq(freq: str) -> str:
    f = (freq or "").strip().upper()
    if f not in _ALLOWED_FREQ:
        # Default safe choice for trend display
        return "W"
    return f


def _safe_str(x) -> str:
    try:
        s = str(x)
    except Exception:
        return ""
    return s.strip()


def _normalize_macro_area(s: object) -> str:
    v = _safe_str(s)
    if not v:
        return "Sin clasificar"
    # Normalización leve (evita duplicados por espacios)
    v = " ".join(v.split())
    return v if v else "Sin clasificar"


def _pick_text_col(df: pd.DataFrame) -> Optional[str]:
    for c in _DEFAULT_TEXT_COLS:
        if c in df.columns:
            return c
    return None


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    if "macro_area" not in dfx.columns:
        dfx["macro_area"] = "Sin clasificar"
    return dfx


def _parse_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    dfx = df.copy()
    dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce", utc=False)
    # Drop invalid
    dfx = dfx.dropna(subset=[date_col]).copy()
    return dfx


def _make_period_str(dates: pd.Series, freq: str) -> pd.Series:
    """
    Generate stable string periods:
      - M: 'YYYY-MM'
      - W: 'YYYY-Www' (ISO week) like '2026-W07'
      - D: 'YYYY-MM-DD'
    """
    freq = _coerce_freq(freq)
    d = pd.to_datetime(dates, errors="coerce")
    if freq == "M":
        return d.dt.to_period("M").astype(str)  # 'YYYY-MM'
    if freq == "D":
        return d.dt.to_period("D").astype(str)  # 'YYYY-MM-DD'
    # W: use Period('W') -> 'YYYY-MM-DD/...' is ugly; use ISO week label
    iso = d.dt.isocalendar()
    return (iso["year"].astype(str) + "-W" + iso["week"].astype(int).astype(str).str.zfill(2))


def _sort_period_index(periods: pd.Index, freq: str) -> pd.Index:
    """
    Ensure correct chronological sorting even for ISO week strings.
    """
    freq = _coerce_freq(freq)
    p = periods.astype(str)

    if freq == "W":
        # Parse 'YYYY-Www' into sortable tuple
        s = p.to_series()
        parts = s.str.extract(r"^(?P<y>\d{4})-W(?P<w>\d{2})$")
        # If parse fails, fallback to lexical
        if parts.isna().any(axis=None):
            return p.sort_values()
        key = parts["y"].astype(int) * 100 + parts["w"].astype(int)
        return p[np.argsort(key.values)]
    else:
        # For 'YYYY-MM' / 'YYYY-MM-DD', lexical is chronological
        return p.sort_values()


def _limit_top_k_columns(wide: pd.DataFrame, top_k: Optional[int]) -> pd.DataFrame:
    """
    Keep only top_k macro_areas by total sum (descending) to reduce UI cost.
    If top_k is None or <=0 => keep all.
    """
    if wide is None or wide.empty:
        return wide
    if top_k is None:
        return wide
    try:
        k = int(top_k)
    except Exception:
        return wide
    if k <= 0 or wide.shape[1] <= k:
        return wide

    sums = wide.sum(axis=0).sort_values(ascending=False)
    keep = sums.index[:k].tolist()
    return wide[keep].copy()


def _reindex_fill_periods(wide: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Make the period index dense for plotting (fills missing periods with 0).
    For W uses ISO-week labels; we approximate densification by converting to a
    canonical PeriodIndex when possible.
    """
    if wide is None or wide.empty:
        return wide

    freq = _coerce_freq(freq)

    if freq in ("M", "D"):
        # These parse well to PeriodIndex
        try:
            pi = pd.PeriodIndex(wide.index.astype(str), freq=freq)
            full = pd.period_range(pi.min(), pi.max(), freq=freq).astype(str)
            return wide.reindex(full).fillna(0.0)
        except Exception:
            return wide

    # W: densify by converting ISO week labels to a Monday date, then to weekly periods
    idx = wide.index.astype(str)
    s = pd.Series(idx, index=idx)

    parts = s.str.extract(r"^(?P<y>\d{4})-W(?P<w>\d{2})$")
    if parts.isna().any(axis=None):
        return wide

    y = parts["y"].astype(int).values
    w = parts["w"].astype(int).values

    # ISO week to date: year-week-1 (Monday)
    # pandas can parse with format %G-W%V-%u but not always; do manual fallback
    try:
        monday = pd.to_datetime(
            [f"{yy}-W{ww:02d}-1" for yy, ww in zip(y, w)],
            format="%G-W%V-%u",
            errors="coerce",
        )
        if monday.isna().any():
            return wide
        pi = monday.to_period("W-MON")  # weekly anchored Monday
        full = pd.period_range(pi.min(), pi.max(), freq="W-MON")
        # Map back to ISO label
        d = full.to_timestamp()
        iso = d.isocalendar()
        full_lbl = (iso["year"].astype(str) + "-W" + iso["week"].astype(int).astype(str).str.zfill(2)).tolist()
        return wide.reindex(full_lbl).fillna(0.0)
    except Exception:
        return wide


# =============================================================================
# Public API
# =============================================================================
def aggregate_macro(
    df: pd.DataFrame,
    freq: str,
    *,
    date_col: str = "date",
    macro_col: str = "macro_area",
    # optional constraints
    min_docs_per_period: int = 1,
    min_total_docs: int = 1,
    drop_empty_macro: bool = True,
    dense_period_index: bool = True,
    top_k: Optional[int] = None,
    # quality diagnostics
    compute_entropy: bool = True,
) -> MacroAgg:
    """
    Robust macro aggregation for both Histórico and Live.

    Produces:
      - df_macro (long): period, macro_area, count, total_docs, rel_freq
      - wide_rel_freq (wide): periods x macro_area (rel_freq)
      - wide_count (wide): periods x macro_area (count)
      - meta diagnostics: counts, period range, sparsity, entropy, etc.

    Notes:
      - freq supports: 'D','W','M'
      - Weekly uses ISO labels 'YYYY-Www' to avoid ugly ranges.
      - Can densify missing periods for stable plotting.
      - Can keep only top_k macro_areas (by volume) for UI.
    """
    meta: Dict[str, object] = {
        "freq": _coerce_freq(freq),
        "date_col": date_col,
        "macro_col": macro_col,
    }

    # Basic validation
    if df is None or df.empty:
        empty_long = pd.DataFrame(columns=["period", "macro_area", "count", "total_docs", "rel_freq"])
        meta.update(
            {
                "status": "empty_input",
                "n_rows_in": 0,
                "n_rows_used": 0,
                "n_periods": 0,
                "n_macro_areas": 0,
            }
        )
        return MacroAgg(empty_long, pd.DataFrame(), pd.DataFrame(), meta)

    dfx = _ensure_required_columns(df)

    if date_col not in dfx.columns:
        empty_long = pd.DataFrame(columns=["period", "macro_area", "count", "total_docs", "rel_freq"])
        meta.update(
            {
                "status": "missing_date_col",
                "n_rows_in": int(len(dfx)),
                "n_rows_used": 0,
                "n_periods": 0,
                "n_macro_areas": 0,
            }
        )
        return MacroAgg(empty_long, pd.DataFrame(), pd.DataFrame(), meta)

    # Normalize macro_area
    if macro_col not in dfx.columns:
        dfx[macro_col] = "Sin clasificar"
    dfx[macro_col] = dfx[macro_col].map(_normalize_macro_area)

    # Parse dates
    n_in = int(len(dfx))
    dfx = _parse_dates(dfx, date_col=date_col)
    n_used = int(len(dfx))

    if dfx.empty:
        empty_long = pd.DataFrame(columns=["period", "macro_area", "count", "total_docs", "rel_freq"])
        meta.update(
            {
                "status": "all_dates_invalid",
                "n_rows_in": n_in,
                "n_rows_used": 0,
                "n_periods": 0,
                "n_macro_areas": 0,
            }
        )
        return MacroAgg(empty_long, pd.DataFrame(), pd.DataFrame(), meta)

    # Build period labels
    freq_n = _coerce_freq(freq)
    dfx["period"] = _make_period_str(dfx[date_col], freq=freq_n)

    # Drop empties if requested
    if drop_empty_macro:
        dfx = dfx[dfx[macro_col].astype(str).str.strip().ne("")].copy()
        if dfx.empty:
            empty_long = pd.DataFrame(columns=["period", "macro_area", "count", "total_docs", "rel_freq"])
            meta.update(
                {
                    "status": "empty_after_macro_filter",
                    "n_rows_in": n_in,
                    "n_rows_used": 0,
                    "n_periods": 0,
                    "n_macro_areas": 0,
                }
            )
            return MacroAgg(empty_long, pd.DataFrame(), pd.DataFrame(), meta)

    # Total docs per period
    total_by_period = dfx.groupby("period").size().rename("total_docs").reset_index()

    # Filter periods by min docs
    if min_docs_per_period and int(min_docs_per_period) > 1:
        keep_p = total_by_period.loc[total_by_period["total_docs"] >= int(min_docs_per_period), "period"]
        keep_set = set(keep_p.astype(str).tolist())
        dfx = dfx[dfx["period"].astype(str).isin(keep_set)].copy()
        total_by_period = total_by_period[total_by_period["period"].astype(str).isin(keep_set)].copy()

    if dfx.empty or total_by_period.empty:
        empty_long = pd.DataFrame(columns=["period", "macro_area", "count", "total_docs", "rel_freq"])
        meta.update(
            {
                "status": "empty_after_period_filter",
                "n_rows_in": n_in,
                "n_rows_used": 0,
                "n_periods": 0,
                "n_macro_areas": 0,
            }
        )
        return MacroAgg(empty_long, pd.DataFrame(), pd.DataFrame(), meta)

    # Global min docs constraint
    if min_total_docs and int(min_total_docs) > 1 and int(dfx.shape[0]) < int(min_total_docs):
        empty_long = pd.DataFrame(columns=["period", "macro_area", "count", "total_docs", "rel_freq"])
        meta.update(
            {
                "status": "below_min_total_docs",
                "n_rows_in": n_in,
                "n_rows_used": int(dfx.shape[0]),
                "n_periods": int(total_by_period.shape[0]),
                "n_macro_areas": int(dfx[macro_col].nunique(dropna=True)),
            }
        )
        return MacroAgg(empty_long, pd.DataFrame(), pd.DataFrame(), meta)

    # Counts per macro+period
    counts = (
        dfx.groupby(["period", macro_col])
           .size()
           .rename("count")
           .reset_index()
           .rename(columns={macro_col: "macro_area"})
    )

    out = counts.merge(total_by_period, on="period", how="left")
    out["total_docs"] = out["total_docs"].fillna(0).astype(int)

    denom = out["total_docs"].replace(0, np.nan)
    out["rel_freq"] = (out["count"] / denom).fillna(0.0).astype(float)
    out["count"] = out["count"].fillna(0).astype(int)

    # Sort periods chronologically (robust for W)
    # We build a stable ordering from unique periods
    unique_p = out["period"].astype(str).unique()
    ordered_p = _sort_period_index(pd.Index(unique_p), freq=freq_n).tolist()
    out["period"] = pd.Categorical(out["period"].astype(str), categories=ordered_p, ordered=True)
    out = out.sort_values(["period", "macro_area"]).reset_index(drop=True)
    out["period"] = out["period"].astype(str)

    # Wide matrices
    wide_rel = (
        out.pivot_table(index="period", columns="macro_area", values="rel_freq", aggfunc="sum")
           .fillna(0.0)
    )
    wide_cnt = (
        out.pivot_table(index="period", columns="macro_area", values="count", aggfunc="sum")
           .fillna(0.0)
    )

    # Sort wide index chronologically
    wide_rel = wide_rel.reindex(_sort_period_index(wide_rel.index, freq=freq_n))
    wide_cnt = wide_cnt.reindex(_sort_period_index(wide_cnt.index, freq=freq_n))

    # Densify missing periods
    if dense_period_index:
        wide_rel = _reindex_fill_periods(wide_rel, freq=freq_n)
        wide_cnt = _reindex_fill_periods(wide_cnt, freq=freq_n)

        # Keep long table consistent with densified index? (optional)
        # We keep df_macro as observed-only (better for audit), wide is for UI.

    # Keep only top_k macro areas if requested (UI performance)
    wide_rel = _limit_top_k_columns(wide_rel, top_k)
    wide_cnt = _limit_top_k_columns(wide_cnt, top_k)

    # Diagnostics
    n_periods = int(wide_rel.shape[0]) if wide_rel is not None else 0
    n_macros = int(wide_rel.shape[1]) if wide_rel is not None else 0

    meta.update(
        {
            "status": "ok",
            "n_rows_in": n_in,
            "n_rows_used": int(n_used),
            "n_rows_after_filters": int(len(dfx)),
            "n_periods": n_periods,
            "n_macro_areas": n_macros,
            "period_min": str(wide_rel.index.min()) if n_periods else None,
            "period_max": str(wide_rel.index.max()) if n_periods else None,
            "min_docs_per_period": int(min_docs_per_period),
            "min_total_docs": int(min_total_docs),
            "dense_period_index": bool(dense_period_index),
            "top_k": int(top_k) if top_k is not None else None,
        }
    )

    # Sparsity (how many zero cells)
    try:
        if n_periods and n_macros:
            z = float((wide_cnt.values == 0).sum())
            meta["sparsity"] = float(z / (n_periods * n_macros))
        else:
            meta["sparsity"] = None
    except Exception:
        meta["sparsity"] = None

    # Entropy of distribution per period (optional; helps UI explanations)
    if compute_entropy:
        try:
            # Use counts (more stable)
            ent = []
            for _, row in wide_cnt.iterrows():
                v = row.values.astype(float)
                tot = float(v.sum())
                if tot <= 0:
                    ent.append(np.nan)
                    continue
                p = v / tot
                p = p[p > 0]
                ent.append(float(-(p * np.log(p)).sum()))
            meta["entropy_mean"] = float(np.nanmean(ent)) if len(ent) else None
            meta["entropy_last"] = float(ent[-1]) if len(ent) else None
        except Exception:
            meta["entropy_mean"] = None
            meta["entropy_last"] = None

    # Recommended: provide a simple “coverage” label for UI
    try:
        min_dt = pd.to_datetime(dfx[date_col], errors="coerce").min()
        max_dt = pd.to_datetime(dfx[date_col], errors="coerce").max()
        meta["date_min"] = str(min_dt.date()) if pd.notna(min_dt) else None
        meta["date_max"] = str(max_dt.date()) if pd.notna(max_dt) else None
    except Exception:
        meta["date_min"] = None
        meta["date_max"] = None

    # Final long df columns guaranteed
    df_macro = out[["period", "macro_area", "count", "total_docs", "rel_freq"]].copy()

    return MacroAgg(df_macro=df_macro, wide_rel_freq=wide_rel, wide_count=wide_cnt, meta=meta)
