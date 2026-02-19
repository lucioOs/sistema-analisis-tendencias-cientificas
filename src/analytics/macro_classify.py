# src/analytics/macro_classify.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Config
# =============================================================================
@dataclass(frozen=True)
class ClassifyCfg:
    # Ventana mínima para clasificar con sentido (n_periods)
    min_periods: int = 6

    # Umbrales base (se ajustan si la serie es muy ruidosa)
    slope_up: float = 2e-4
    slope_down: float = -2e-4
    growth_up: float = 0.20
    growth_down: float = -0.15

    # Consolidada: estable y pendiente pequeña
    stability_min: float = 0.65
    slope_near0: float = 2e-4

    # Suavizado de crecimiento (usa medias de bordes)
    edge_k: int = 3

    # Si todo es casi cero, no clasificar agresivo
    eps: float = 1e-12


# =============================================================================
# Helpers
# =============================================================================
def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _edge_mean(y: np.ndarray, k: int, side: str) -> float:
    if y.size == 0:
        return 0.0
    k = max(1, min(int(k), int(y.size)))
    if side == "first":
        return float(np.mean(y[:k]))
    return float(np.mean(y[-k:]))


def _linear_slope(y: np.ndarray) -> float:
    n = int(y.size)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0


def _stability(y: np.ndarray, eps: float) -> float:
    if y.size == 0:
        return 0.0
    mu = float(np.mean(y))
    sd = float(np.std(y))
    # estabilidad en [0,1] aprox
    return float(max(0.0, 1.0 - (sd / (mu + eps))))


def _dynamic_thresholds(y: np.ndarray, cfg: ClassifyCfg) -> Tuple[float, float]:
    """
    Ajusta tolerancia de pendiente según ruido:
    - series muy ruidosas => exige mayor pendiente
    """
    if y.size < 2:
        return cfg.slope_up, cfg.slope_down

    mu = float(np.mean(y))
    sd = float(np.std(y))
    cv = sd / (mu + cfg.eps)

    # Escala simple: si cv sube, sube el umbral
    scale = 1.0 + min(2.0, max(0.0, cv))  # entre 1 y 3
    return cfg.slope_up * scale, cfg.slope_down * scale


def _classify_one(y_raw: np.ndarray, cfg: ClassifyCfg) -> Dict[str, float | str | int]:
    y = np.asarray(y_raw, dtype=float)
    y = np.nan_to_num(y, nan=0.0)

    n = int(y.size)
    slope = _linear_slope(y)

    k = int(cfg.edge_k)
    first = _edge_mean(y, k, "first")
    last = _edge_mean(y, k, "last")
    growth = float((last - first) / (first + cfg.eps))

    stab = _stability(y, cfg.eps)

    # Si muy pocos periodos, no forzar etiquetas
    if n < int(cfg.min_periods):
        class_label = "insuficiente"
        return {
            "class": class_label,
            "slope": slope,
            "growth": growth,
            "stability": stab,
            "n_periods": n,
        }

    # Umbrales dinámicos por ruido
    slope_up, slope_down = _dynamic_thresholds(y, cfg)

    # Reglas
    class_label = "otros"

    # Emergente / Declive
    if slope > slope_up and growth > float(cfg.growth_up):
        class_label = "emergente"
    elif slope < slope_down and growth < float(cfg.growth_down):
        class_label = "declive"
    else:
        # Consolidada (estable y sin pendiente marcada)
        if stab >= float(cfg.stability_min) and abs(slope) < float(cfg.slope_near0):
            class_label = "consolidada"
        else:
            class_label = "neutro"

    return {
        "class": class_label,
        "slope": slope,
        "growth": growth,
        "stability": stab,
        "n_periods": n,
    }


# =============================================================================
# Public API
# =============================================================================
def classify_macro(
    df_wide_relfreq: pd.DataFrame,
    *,
    cfg: Optional[ClassifyCfg] = None,
    totals_count: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Clasifica macro-áreas para la ventana/rango actual (ideal para LIVE).

    Input:
      df_wide_relfreq:
        index=period (str o Period), columns=macro_area, values=rel_freq (0..1)
      totals_count (opcional):
        serie con total de documentos por macro_area (para mostrar volumen real)

    Output:
      DataFrame con:
        macro_area, class, slope, growth, stability, n_periods, total_count
    """
    cfg = cfg or ClassifyCfg()

    if df_wide_relfreq is None or df_wide_relfreq.empty:
        return pd.DataFrame(
            columns=["macro_area", "class", "slope", "growth", "stability", "n_periods", "total_count"]
        )

    wide = df_wide_relfreq.copy()
    wide = wide.fillna(0.0)

    rows = []
    for macro_area in wide.columns:
        y = wide[macro_area].astype(float).values
        metrics = _classify_one(y, cfg)

        total_count = 0
        if totals_count is not None:
            try:
                total_count = int(totals_count.get(macro_area, 0))
            except Exception:
                total_count = 0

        rows.append(
            {
                "macro_area": str(macro_area),
                "class": str(metrics["class"]),
                "slope": _safe_float(metrics["slope"]),
                "growth": _safe_float(metrics["growth"]),
                "stability": _safe_float(metrics["stability"]),
                "n_periods": int(metrics["n_periods"]),
                "total_count": int(total_count),
            }
        )

    out = pd.DataFrame(rows)

    # Orden útil: primero emergente/declive, luego por volumen
    class_order = {"emergente": 0, "declive": 1, "consolidada": 2, "neutro": 3, "otros": 4, "insuficiente": 5}
    out["class_rank"] = out["class"].map(class_order).fillna(99).astype(int)
    out = out.sort_values(["class_rank", "total_count"], ascending=[True, False]).drop(columns=["class_rank"])

    return out
