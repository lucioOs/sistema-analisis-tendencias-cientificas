# tests/test_holtwinters.py
"""
PRUEBA: Holt-Winters (Exponential Smoothing) sobre datos reales del proyecto.

Qué hace:
1) Carga data/processed/clean.parquet (columnas: date, text_clean, ...)
2) Construye una serie temporal mensual de "conteos" a partir de un filtro:
   - Por defecto: cuenta cuántos documentos contienen una palabra clave (keyword) en text_clean.
3) Toma una ventana de N meses (default 60), separa Train/Test (default 90/10).
4) Entrena Holt-Winters y predice los meses de prueba.
5) Calcula MAE y RMSE.
6) Exporta evidencias (CSV real vs pred, JSON con métricas y PNG con gráfica).

Ejecutar:
(.venv) python tests/test_holtwinters.py --keyword llm
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# statsmodels (Holt-Winters)
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# métricas (sin squared=... para compatibilidad)
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ----------------------------
# Config / Helpers
# ----------------------------
@dataclass
class HWReport:
    keyword: str
    parquet_path: str
    rows_total: int
    rows_matched: int
    months_used: int
    train_points: int
    test_points: int
    seasonal_periods: int
    trend: str
    seasonal: str
    damped_trend: bool
    mae: float
    rmse: float
    evidence_dir: str
    csv_path: str
    png_path: str
    json_path: str


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def normalize_series_monthly(
    s: pd.Series, months: int, fill_value: int = 0
) -> pd.Series:
    """
    - Convierte índice a datetime
    - Re-muestrea a fin de mes (ME)
    - Rellena meses faltantes
    - Toma últimos N meses
    """
    s = s.copy()
    s.index = pd.to_datetime(s.index)

    # resample a fin de mes (ME). Evita 'M' (deprecado).
    s = s.resample("ME").sum()

    # asegurar continuidad
    s = s.asfreq("ME")
    s = s.fillna(fill_value)

    # ventana
    if months and months > 0:
        s = s.tail(months)

    return s


def build_keyword_count_series(df: pd.DataFrame, keyword: str) -> tuple[pd.Series, int]:
    """
    Cuenta documentos por mes donde text_clean contiene keyword.
    Devuelve serie mensual (antes de normalizar) y rows_matched.
    """
    if "date" not in df.columns:
        raise ValueError("El parquet no tiene columna 'date'.")
    if "text_clean" not in df.columns:
        raise ValueError("El parquet no tiene columna 'text_clean'.")

    d = df[["date", "text_clean"]].dropna()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])

    # match keyword como substring simple (rápido y suficiente para evidencia)
    k = keyword.strip().lower()
    text = d["text_clean"].astype(str).str.lower()
    mask = text.str.contains(k, regex=False)
    matched = d.loc[mask, ["date"]]

    rows_matched = int(matched.shape[0])

    # conteo por mes
    series = matched.set_index("date").assign(cnt=1)["cnt"].resample("ME").sum()

    return series, rows_matched


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        default="data/processed/clean.parquet",
        help="Ruta a clean.parquet",
    )
    ap.add_argument(
        "--keyword",
        default="llm",
        help="Palabra clave para construir la serie (conteo mensual de docs que la contienen).",
    )
    ap.add_argument("--months", type=int, default=60, help="Ventana de meses a usar.")
    ap.add_argument(
        "--test_months",
        type=int,
        default=6,
        help="Meses al final para test (horizonte).",
    )
    ap.add_argument(
        "--seasonal_periods",
        type=int,
        default=12,
        help="Periodos estacionales (12 = anual en mensual).",
    )
    ap.add_argument(
        "--trend",
        default="add",
        choices=["add", "mul", "None"],
        help="Tipo de tendencia en Holt-Winters.",
    )
    ap.add_argument(
        "--seasonal",
        default="add",
        choices=["add", "mul", "None"],
        help="Tipo de estacionalidad en Holt-Winters.",
    )
    ap.add_argument(
        "--damped",
        action="store_true",
        help="Usar damped_trend=True.",
    )
    ap.add_argument(
        "--evidence_dir",
        default="evidence/P5_holtwinters",
        help="Carpeta base para evidencias.",
    )
    ap.add_argument(
        "--rows_limit",
        type=int,
        default=250000,
        help="Límite de filas a leer para acelerar (0 = leer todo).",
    )

    args = ap.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"[ERROR] No existe: {parquet_path}")
        return 2

    print("=== PRUEBA HOLT-WINTERS ===")
    print(f"Parquet: {parquet_path}")
    print(f"Keyword: {args.keyword}")
    print(f"Ventana (meses): {args.months} | Test (meses): {args.test_months}")
    print(f"HW: trend={args.trend} seasonal={args.seasonal} damped={args.damped}")
    print(f"seasonal_periods={args.seasonal_periods}")

    # Carga (con límite opcional)
    # Nota: leer por completo puede ser pesado; el limit reduce tiempo para evidencia.
    df = pd.read_parquet(parquet_path)
    rows_total = int(df.shape[0])
    if args.rows_limit and args.rows_limit > 0 and rows_total > args.rows_limit:
        df = df.sample(n=args.rows_limit, random_state=42).reset_index(drop=True)
        print(f"[INFO] Submuestreo aplicado: {args.rows_limit} filas (de {rows_total}).")
    else:
        print(f"[INFO] Filas leídas: {rows_total}")

    # Serie por keyword
    raw_series, rows_matched = build_keyword_count_series(df, args.keyword)

    if rows_matched == 0:
        print("[ERROR] No hubo coincidencias para esa keyword. Prueba otra (ej: 'graph', 'segmentation', 'transformer').")
        return 3

    series = normalize_series_monthly(raw_series, months=args.months, fill_value=0)
    months_used = int(series.shape[0])

    if months_used < max(args.seasonal_periods * 2, args.test_months + 12):
        print("[ERROR] Serie insuficiente para Holt-Winters con estacionalidad.")
        print(f"Meses disponibles: {months_used}. Recomendación: aumenta --months o usa keyword con más ocurrencias.")
        return 4

    if args.test_months <= 0 or args.test_months >= months_used:
        print("[ERROR] --test_months inválido para el tamaño de la serie.")
        return 5

    train = series.iloc[:-args.test_months]
    test = series.iloc[-args.test_months:]

    print(f"Puntos usados: {months_used}")
    print(f"Train: {len(train)} | Test: {len(test)}")
    print(f"Rango: {series.index.min().date()} -> {series.index.max().date()}")

    # Configurar HW (None -> None real)
    trend = None if args.trend == "None" else args.trend
    seasonal = None if args.seasonal == "None" else args.seasonal

    # Ajuste Holt-Winters
    try:
        model = ExponentialSmoothing(
            train.astype(float),
            trend=trend,
            damped_trend=bool(args.damped) if trend is not None else False,
            seasonal=seasonal,
            seasonal_periods=int(args.seasonal_periods) if seasonal is not None else None,
            initialization_method="estimated",
        ).fit(optimized=True)

        pred = model.forecast(steps=len(test))
        pred.index = test.index  # alinear

    except Exception as e:
        print("[ERROR] Falló el ajuste Holt-Winters.")
        print("Detalle:", repr(e))
        return 6

    # Métricas
    y_true = test.values.astype(float)
    y_pred = pred.values.astype(float)

    mae_val = float(mean_absolute_error(y_true, y_pred))
    rmse_val = rmse(y_true, y_pred)

    print("\n--- MÉTRICAS ---")
    print(f"MAE : {mae_val:.3f}")
    print(f"RMSE: {rmse_val:.3f}")

    # Evidencias (carpeta por corrida)
    base_dir = Path(args.evidence_dir)
    run_dir = base_dir / f"HW_{args.keyword}_{now_stamp()}"
    safe_mkdir(run_dir)

    # CSV real vs pred
    out_df = pd.DataFrame(
        {
            "date": test.index.astype(str),
            "real": y_true,
            "pred": y_pred,
            "abs_error": np.abs(y_true - y_pred),
            "sq_error": (y_true - y_pred) ** 2,
        }
    )
    csv_path = run_dir / "P5_hw_real_vs_pred.csv"
    out_df.to_csv(csv_path, index=False, encoding="utf-8")

    # PNG (gráfica)
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.plot(train.index, train.values, label="train")
        plt.plot(test.index, test.values, label="real (test)")
        plt.plot(pred.index, pred.values, label="pred")
        plt.title(f"Holt-Winters por keyword='{args.keyword}' | MAE={mae_val:.1f} RMSE={rmse_val:.1f}")
        plt.xlabel("Mes")
        plt.ylabel("Conteo de documentos")
        plt.legend()
        plt.tight_layout()
        png_path = run_dir / "P5_hw_plot.png"
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
    except Exception as e:
        print("[WARN] No se pudo generar PNG:", repr(e))
        png_path = run_dir / "P5_hw_plot.png"  # referencia aunque no exista

    # JSON resumen
    rep = HWReport(
        keyword=args.keyword,
        parquet_path=str(parquet_path),
        rows_total=int(rows_total),
        rows_matched=int(rows_matched),
        months_used=int(months_used),
        train_points=int(len(train)),
        test_points=int(len(test)),
        seasonal_periods=int(args.seasonal_periods),
        trend=str(args.trend),
        seasonal=str(args.seasonal),
        damped_trend=bool(args.damped),
        mae=float(mae_val),
        rmse=float(rmse_val),
        evidence_dir=str(run_dir),
        csv_path=str(csv_path),
        png_path=str(png_path),
        json_path=str(run_dir / "P5_hw_summary.json"),
    )
    json_path = run_dir / "P5_hw_summary.json"
    json_path.write_text(json.dumps(asdict(rep), indent=2, ensure_ascii=False), encoding="utf-8")

    # TXT corto para diapositiva
    txt_path = run_dir / "P5_hw_summary.txt"
    txt_path.write_text(
        "\n".join(
            [
                "PRUEBA HOLT-WINTERS (EVIDENCIA)",
                f"- Keyword: {args.keyword}",
                f"- Docs totales (parquet): {rows_total}",
                f"- Docs que contienen keyword: {rows_matched}",
                f"- Meses usados: {months_used} | Train: {len(train)} | Test: {len(test)}",
                f"- HW: trend={args.trend}, seasonal={args.seasonal}, seasonal_periods={args.seasonal_periods}, damped={args.damped}",
                f"- MAE: {mae_val:.3f}",
                f"- RMSE: {rmse_val:.3f}",
                f"- CSV: {csv_path}",
                f"- PNG: {png_path}",
                f"- JSON: {json_path}",
            ]
        ),
        encoding="utf-8",
    )

    print("\n--- EVIDENCIAS GENERADAS ---")
    print(f"Carpeta: {run_dir}")
    print(f"CSV    : {csv_path}")
    print(f"PNG    : {png_path}")
    print(f"JSON   : {json_path}")
    print(f"TXT    : {txt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
