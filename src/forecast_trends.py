# src/forecast_trends.py
# Tendencias + Clasificación + Predicción (batch) — ROBUSTO Y CANÓNICO (macro_area-first)
# -------------------------------------------------------------------------------
# Salidas:
#   - data/processed/macro_trends_full.parquet      (period, macro_area, count, total_docs, rel_freq)
#   - data/processed/macro_trend_classes.parquet    (macro_area, class, slope, growth, stability, total_count, n_periods)
#   - data/processed/macro_trends_forecast.parquet  (macro_area, period, rel_freq_pred, count_pred, model)
#
# Predicción:
#   - Holt-Winters (statsmodels ExponentialSmoothing) si hay datos suficientes
#   - Fallback lineal + estacional naive si no aplica o falla
#
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from src.load_data import enforce_schema

# -----------------------------
# Optional dependency (statsmodels)
# -----------------------------
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore

    STATSMODELS_OK = True
except Exception:
    ExponentialSmoothing = None  # type: ignore
    STATSMODELS_OK = False


@dataclass(frozen=True)
class ForecastCfg:
    h: int
    seasonal_periods: int
    prefer_hw: bool
    hw_min_n: int


# =============================================================================
# Helpers: period handling
# =============================================================================
def _future_periods(periods_hist: pd.Index, freq: str, h: int) -> List[str]:
    """
    periods_hist: index de strings (period) o PeriodIndex convertible.
    Devuelve labels futuros reales según freq. Si no parsea, usa "future+N".
    """
    if h <= 0:
        return []

    try:
        pi = pd.PeriodIndex(periods_hist.astype(str), freq=freq)
        last = pi.max()
        fut = [last + i for i in range(1, h + 1)]
        return [str(p) for p in fut]
    except Exception:
        return [f"future+{i}" for i in range(1, h + 1)]


def _safe_period_str_from_date(dt: pd.Series, freq: str) -> pd.Series:
    d = pd.to_datetime(dt, errors="coerce")
    return d.dt.to_period(freq).astype(str)


# =============================================================================
# Aggregate
# =============================================================================
def _aggregate_macro(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Devuelve:
      period(str), macro_area, count, total_docs, rel_freq
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["period", "macro_area", "count", "total_docs", "rel_freq"])

    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")
    dfx = dfx.dropna(subset=["date", "macro_area"]).copy()
    if dfx.empty:
        return pd.DataFrame(columns=["period", "macro_area", "count", "total_docs", "rel_freq"])

    dfx["period"] = _safe_period_str_from_date(dfx["date"], freq)

    total_by_period = dfx.groupby("period").size().rename("total_docs").reset_index()

    counts = (
        dfx.groupby(["period", "macro_area"])
        .size()
        .rename("count")
        .reset_index()
    )

    out = counts.merge(total_by_period, on="period", how="left")
    out["total_docs"] = out["total_docs"].fillna(0).astype(int)

    denom = out["total_docs"].replace(0, np.nan)
    out["rel_freq"] = (out["count"] / denom).fillna(0.0).astype(float)

    out = out.sort_values(["period", "macro_area"]).reset_index(drop=True)
    return out


def _pivot_metric(df_macro: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    period x macro_area => value_col (wide)
    """
    if df_macro is None or df_macro.empty:
        return pd.DataFrame()

    wide = (
        df_macro.pivot_table(
            index="period",
            columns="macro_area",
            values=value_col,
            aggfunc="sum",
        )
        .fillna(0.0)
        .sort_index()
    )
    return wide


def _compute_total_counts(df_macro: pd.DataFrame) -> pd.DataFrame:
    return (
        df_macro.groupby("macro_area")["count"]
        .sum()
        .rename("total_count")
        .reset_index()
    )


# =============================================================================
# Classification
# =============================================================================
def _classify_macro(df_wide_relfreq: pd.DataFrame) -> pd.DataFrame:
    if df_wide_relfreq is None or df_wide_relfreq.empty:
        return pd.DataFrame(columns=["macro_area", "class", "slope", "growth", "stability", "n_periods"])

    x = np.arange(len(df_wide_relfreq.index), dtype=float)

    rows = []
    for macro in df_wide_relfreq.columns:
        y = df_wide_relfreq[macro].astype(float).values
        y = np.nan_to_num(y, nan=0.0)

        slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else 0.0

        first3 = float(np.mean(y[:3])) if len(y) >= 3 else float(y[0])
        last3 = float(np.mean(y[-3:])) if len(y) >= 3 else float(y[-1])
        growth = float((last3 - first3) / (first3 + 1e-9))

        mean = float(np.mean(y))
        std = float(np.std(y))
        stability = float(max(0.0, 1.0 - (std / (mean + 1e-9))))

        label = "otros"
        if slope > 2e-4 and growth > 0.20:
            label = "emergente"
        elif slope < -2e-4 and growth < -0.15:
            label = "declive"
        else:
            if stability >= 0.65 and abs(slope) < 2e-4:
                label = "consolidada"

        rows.append(
            {
                "macro_area": str(macro),
                "class": label,
                "slope": slope,
                "growth": growth,
                "stability": stability,
                "n_periods": int(len(y)),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Forecast models
# =============================================================================
def _forecast_fallback(y: np.ndarray, h: int, sp: int) -> Tuple[np.ndarray, str]:
    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=0.0)
    n = len(y)

    if h <= 0:
        return np.zeros(0, dtype=float), "fallback-empty"

    if n <= 0:
        return np.zeros(h, dtype=float), "fallback-empty"

    x = np.arange(n, dtype=float)
    if n >= 2:
        a, b = np.polyfit(x, y, 1)
    else:
        a, b = 0.0, float(y[-1])

    base = a * np.arange(n, n + h, dtype=float) + b

    sp = int(sp or 0)
    if sp >= 2 and n >= sp:
        season = y[-sp:]
        season_rep = np.resize(season, h)
        pred = 0.75 * base + 0.25 * season_rep
        pred = np.clip(pred, 0.0, None)
        return pred, "linear+seasonal_naive"

    base = np.clip(base, 0.0, None)
    return base, "linear"


def _hw_fit_forecast(y: np.ndarray, h: int, sp: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Holt-Winters robusto:
    - prueba add/add y mul/mul (si datos >0)
    - si falla, regresa None
    """
    if not STATSMODELS_OK or ExponentialSmoothing is None:
        return None, None

    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=0.0)

    sp = int(sp or 0)
    if h <= 0 or sp < 2:
        return None, None

    # Regla mínima: al menos 2 ciclos y algunos puntos extra
    if len(y) < max(8, 2 * sp):
        return None, None

    y_pos = bool(np.all(y > 0))

    candidates: List[Tuple[str, dict]] = [("holt_winters_add", {"trend": "add", "seasonal": "add"})]
    if y_pos:
        candidates.append(("holt_winters_mul", {"trend": "mul", "seasonal": "mul"}))

    best_pred: Optional[np.ndarray] = None
    best_name: Optional[str] = None
    best_sse = float("inf")

    for name, params in candidates:
        try:
            model = ExponentialSmoothing(
                y,
                trend=params["trend"],
                seasonal=params["seasonal"],
                seasonal_periods=sp,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)

            resid = getattr(fit, "resid", None)
            sse = float(np.sum(np.square(resid))) if resid is not None else float("inf")

            pred = np.asarray(fit.forecast(h), dtype=float)
            pred = np.clip(pred, 0.0, None)

            if np.isfinite(sse) and sse < best_sse:
                best_sse = sse
                best_pred = pred
                best_name = name
        except Exception:
            continue

    return best_pred, best_name


def _should_try_hw(y: np.ndarray, cfg: ForecastCfg) -> bool:
    if not cfg.prefer_hw:
        return False
    if not STATSMODELS_OK:
        return False

    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=0.0)

    if len(y) < int(cfg.hw_min_n):
        return False
    if float(np.sum(y)) <= 0.0:
        return False
    if int(cfg.seasonal_periods or 0) < 2:
        return False

    return True


def _forecast_macro(
    df_wide_metric: pd.DataFrame,
    *,
    freq: str,
    value_col_name: str,
    cfg: ForecastCfg,
) -> pd.DataFrame:
    """
    Forecast por macro_area sobre una métrica (rel_freq o count).
    Produce period futuro REAL (string) + pred + model.
    """
    if df_wide_metric is None or df_wide_metric.empty:
        return pd.DataFrame(columns=["macro_area", "period", f"{value_col_name}_pred", "model"])

    macro_areas = list(df_wide_metric.columns)
    periods_hist = df_wide_metric.index.astype(str)
    future_periods = _future_periods(periods_hist, freq=freq, h=int(cfg.h))

    rows = []
    for macro in macro_areas:
        y = df_wide_metric[macro].astype(float).values
        y = np.nan_to_num(y, nan=0.0)

        pred: Optional[np.ndarray] = None
        model_name: str = "linear"

        if _should_try_hw(y, cfg):
            hw_pred, hw_name = _hw_fit_forecast(y, h=int(cfg.h), sp=int(cfg.seasonal_periods))
            if hw_pred is not None and hw_name is not None:
                pred = hw_pred
                model_name = hw_name

        if pred is None:
            pred, model_name = _forecast_fallback(y, h=int(cfg.h), sp=int(cfg.seasonal_periods))

        for p, v in zip(future_periods, pred):
            rows.append(
                {
                    "macro_area": str(macro),
                    "period": str(p),
                    f"{value_col_name}_pred": float(v),
                    "model": model_name,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Tendencias+Predicción (batch) por macro_area (robusto)")
    ap.add_argument("--input", default="data/processed/clean.parquet", help="Histórico limpio (recomendado)")
    ap.add_argument("--live", default="", help="Parquet live opcional (se concatena)")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--freq", default="M", choices=["D", "W", "M"])

    ap.add_argument("--min_text_len", type=int, default=20)
    ap.add_argument("--drop_duplicates", action="store_true")

    ap.add_argument("--forecast_h", type=int, default=6)
    ap.add_argument("--seasonal_periods", type=int, default=12)

    # HW controls
    ap.add_argument("--prefer_hw", action="store_true", help="Intentar Holt-Winters si hay datos suficientes")
    ap.add_argument("--hw_min_n", type=int, default=36, help="Mínimo de puntos para intentar Holt-Winters")

    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_trends = out_dir / "macro_trends_full.parquet"
    out_classes = out_dir / "macro_trend_classes.parquet"
    out_forecast = out_dir / "macro_trends_forecast.parquet"

    if not in_path.exists():
        print(f"[ERROR] No existe: {in_path}")
        return 2

    print(f"[INFO] Leyendo histórico: {in_path}")
    df_hist = pd.read_parquet(in_path)
    df_hist = enforce_schema(
        df_hist,
        min_text_len=int(args.min_text_len),
        drop_duplicates=bool(args.drop_duplicates),
    )

    frames = [df_hist]

    if args.live:
        live_path = Path(args.live)
        if not live_path.exists():
            print(f"[WARN] live no existe, se ignora: {live_path}")
        else:
            print(f"[INFO] Leyendo live: {live_path}")
            df_live = pd.read_parquet(live_path)
            df_live = enforce_schema(
                df_live,
                min_text_len=int(args.min_text_len),
                drop_duplicates=bool(args.drop_duplicates),
            )
            frames.append(df_live)

    df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df_all.empty:
        print("[ERROR] Dataset vacío tras normalización.")
        return 3

    # -----------------------------
    # Aggregate macro series
    # -----------------------------
    df_macro = _aggregate_macro(df_all, freq=str(args.freq))
    if df_macro.empty:
        print("[ERROR] No se pudieron construir agregaciones macro.")
        return 4

    # -----------------------------
    # Classification (sobre rel_freq)
    # -----------------------------
    df_wide_relfreq = _pivot_metric(df_macro, value_col="rel_freq")
    df_classes = _classify_macro(df_wide_relfreq)

    # total_count real
    df_totals = _compute_total_counts(df_macro)
    if not df_classes.empty:
        df_classes = df_classes.merge(df_totals, on="macro_area", how="left")
        df_classes["total_count"] = df_classes["total_count"].fillna(0).astype(int)
    else:
        df_classes = pd.DataFrame(columns=["macro_area", "class", "slope", "growth", "stability", "n_periods", "total_count"])

    # -----------------------------
    # Forecast config
    # -----------------------------
    cfg = ForecastCfg(
        h=int(args.forecast_h),
        seasonal_periods=int(args.seasonal_periods),
        prefer_hw=bool(args.prefer_hw),
        hw_min_n=int(args.hw_min_n),
    )

    # -----------------------------
    # Forecast rel_freq
    # -----------------------------
    df_fc = _forecast_macro(
        df_wide_relfreq,
        freq=str(args.freq),
        value_col_name="rel_freq",
        cfg=cfg,
    )

    # Normaliza nombre esperado
    if df_fc is None or df_fc.empty:
        df_fc = pd.DataFrame(columns=["macro_area", "period", "rel_freq_pred", "count_pred", "model"])
    else:
        # value_col_name="rel_freq" => "rel_freq_pred" ya existe
        pass

    # -----------------------------
    # count_pred (aprox): rel_freq_pred * promedio(total_docs por periodo)
    # -----------------------------
    avg_total_docs = (
        df_all.assign(period=_safe_period_str_from_date(df_all["date"], str(args.freq)))
        .groupby("period")
        .size()
        .mean()
    )
    avg_total_docs = float(avg_total_docs) if avg_total_docs and not np.isnan(avg_total_docs) else 0.0

    if "rel_freq_pred" in df_fc.columns and not df_fc.empty:
        df_fc["count_pred"] = (df_fc["rel_freq_pred"].astype(float) * avg_total_docs).round().clip(lower=0).astype(int)
    else:
        df_fc["count_pred"] = 0

    # Orden estable
    if not df_fc.empty:
        df_fc = df_fc.sort_values(["macro_area", "period"]).reset_index(drop=True)

    # -----------------------------
    # Save
    # -----------------------------
    print(f"[OK] Guardando: {out_trends}")
    df_macro.to_parquet(out_trends, index=False)

    print(f"[OK] Guardando: {out_classes}")
    df_classes.to_parquet(out_classes, index=False)

    print(f"[OK] Guardando: {out_forecast}")
    df_fc.to_parquet(out_forecast, index=False)

    print("[OK] Listo.")
    if not STATSMODELS_OK and bool(args.prefer_hw):
        print("[INFO] statsmodels no disponible -> se usó fallback (sin Holt-Winters).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
