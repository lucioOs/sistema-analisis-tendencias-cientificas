# src/preprocess.py
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from utils import Paths, log, die

# Núcleo de taxonomía (una sola regla para todo el sistema)
from src.taxonomy import assign_macro_area, taxonomy_report

# =============================================================================
# Limpieza de texto (robusta, pero ligera)
# - Mantiene letras latinas (incluye acentos) y números
# - Elimina URLs, menciones, hashtags (#X -> X)
# - Elimina símbolos no informativos
# =============================================================================
URL_RE = re.compile(r"http\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NONWORD_RE = re.compile(r"[^a-z0-9áéíóúñü\s]+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def clean_text(t: str) -> str:
    s = str(t or "").lower()
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = HASHTAG_RE.sub(r"\1", s)  # #GenerativeAI -> GenerativeAI
    s = NONWORD_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


# =============================================================================
# Schema / contrato mínimo (para HIST y LIVE)
# =============================================================================
REQUIRED_BASE_COLS = ("date", "text")


def _ensure_min_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Contrato mínimo del pipeline:
      - date: datetime (naive)
      - text: string (title+abstract si existe)
      - text_clean: string (derivable)
      - primary_category: (derivable si hay categories)
      - macro_area (+ trazabilidad) (derivable por taxonomy)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "text", "text_clean", "primary_category", "macro_area"])

    out = df.copy()

    # date
    if "date" not in out.columns:
        out["date"] = pd.NaT
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_convert(None)

    # construir text si no existe
    if "text" not in out.columns:
        if "title" in out.columns and "abstract" in out.columns:
            out["text"] = (out["title"].astype(str).fillna("") + ". " + out["abstract"].astype(str).fillna("")).str.strip()
        elif "title" in out.columns:
            out["text"] = out["title"].astype(str).fillna("").str.strip()
        elif "abstract" in out.columns:
            out["text"] = out["abstract"].astype(str).fillna("").str.strip()
        else:
            out["text"] = ""

    out["text"] = out["text"].astype(str).fillna("").str.replace(r"\s+", " ", regex=True).str.strip()

    # primary_category si no existe y hay categories
    if "primary_category" not in out.columns:
        # taxonomy.assign_macro_area también hace ensure_primary_category,
        # pero aquí dejamos columna lista para consistencia.
        out["primary_category"] = None

    # text_clean (si ya existe, lo respetamos; si no, lo derivamos)
    if "text_clean" not in out.columns:
        out["text_clean"] = out["text"].astype(str)

    out["text_clean"] = out["text_clean"].astype(str).fillna("").str.replace(r"\s+", " ", regex=True).str.strip()

    # limpieza básica de nulos críticos
    out = out.dropna(subset=["date"]).copy()

    return out


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_values("date").copy()

    # prioridad: id -> link -> text+date
    if "id" in out.columns:
        out = out.drop_duplicates(subset=["id"], keep="last")
    if "link" in out.columns:
        out = out.drop_duplicates(subset=["link"], keep="last")

    out = out.drop_duplicates(subset=["date", "text_clean"], keep="last")
    return out.reset_index(drop=True)


# =============================================================================
# QA / Reporte
# =============================================================================
@dataclass(frozen=True)
class PreprocessReport:
    rows_in: int
    rows_out: int
    min_text_len: int
    null_date_rate: float
    empty_text_rate: float
    sin_clasificar_rate: Optional[float]
    macro_source_breakdown: Optional[dict]
    min_date: Optional[str]
    max_date: Optional[str]
    taxonomy: Optional[dict]


def build_report(df_in: pd.DataFrame, df_out: pd.DataFrame, min_len: int) -> PreprocessReport:
    null_date_rate = float(df_out["date"].isna().mean()) if "date" in df_out.columns else 1.0
    empty_text_rate = float((df_out["text_clean"].astype(str).str.len() == 0).mean()) if "text_clean" in df_out.columns else 1.0

    sin_clas_rate = None
    if "macro_area" in df_out.columns and not df_out.empty:
        sin_clas_rate = float((df_out["macro_area"] == "Sin clasificar").mean())

    source_breakdown = None
    if "macro_area_source" in df_out.columns and not df_out.empty:
        source_breakdown = df_out["macro_area_source"].value_counts(dropna=False).to_dict()

    min_date = str(df_out["date"].min()) if "date" in df_out.columns and not df_out.empty else None
    max_date = str(df_out["date"].max()) if "date" in df_out.columns and not df_out.empty else None

    tax = taxonomy_report(df_out) if not df_out.empty else {}

    return PreprocessReport(
        rows_in=int(len(df_in)),
        rows_out=int(len(df_out)),
        min_text_len=int(min_len),
        null_date_rate=null_date_rate,
        empty_text_rate=empty_text_rate,
        sin_clasificar_rate=sin_clas_rate,
        macro_source_breakdown=source_breakdown,
        min_date=min_date,
        max_date=max_date,
        taxonomy=tax,
    )


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocesamiento robusto: text_clean + macro_area (pipeline)")
    ap.add_argument("--min_len", type=int, default=20, help="Longitud mínima del texto limpio (text_clean)")
    ap.add_argument("--drop_duplicates", action="store_true", help="Deduplicar (id/link/date+text_clean)")
    ap.add_argument("--in", dest="inp", default="data/processed/dataset.parquet", help="Ruta input parquet")
    ap.add_argument("--out", dest="out", default="data/processed/clean.parquet", help="Ruta output parquet")
    ap.add_argument("--write_report", action="store_true", help="Escribir reporte QA JSON junto al output")
    args = ap.parse_args()

    P = Paths()
    P.ensure()

    inp = Path(args.inp)
    if not inp.exists():
        die(f"No existe el input: {inp}. Ejecuta primero ingest/load_data.")

    df_in = pd.read_parquet(inp)
    if df_in.empty:
        die("Input parquet está vacío.")

    log(f"Leído {inp} shape={df_in.shape} cols={list(df_in.columns)}")

    # 1) Contrato mínimo
    df = _ensure_min_schema(df_in)

    if df.empty:
        die("Tras asegurar schema (date/text), el dataset quedó vacío (fechas inválidas).")

    # 2) Limpieza final (text_clean) a partir de text
    df["text_clean"] = df["text"].map(clean_text)

    # 3) Filtro por longitud mínima
    df = df[df["text_clean"].str.len() >= int(args.min_len)].copy()
    if df.empty:
        die("Tras limpieza + min_len, el dataset quedó vacío. Baja --min_len o revisa el texto.")

    # 4) Macro-área (núcleo único): se apoya en categories/primary_category si existen
    #    - Para histórico kaggle: viene 'categories'
    #    - Para live_runner: viene 'categories'
    df = assign_macro_area(
        df,
        cat_col="primary_category",          # se asegura/deriva internamente
        out_col="macro_area",
        source_col="macro_area_source",
        hits_col="macro_area_hits",
        score_col="macro_area_score",
        text_cols_priority=("text_clean", "text", "title", "abstract"),
    )

    # 5) Deduplicación opcional
    if bool(args.drop_duplicates):
        before = len(df)
        df = _dedup(df)
        log(f"Deduplicación: {before} -> {len(df)}")

    # 6) Orden / tipos finales (contrato estable)
    df = df.sort_values("date").reset_index(drop=True)
    df["macro_area"] = df["macro_area"].astype(str)

    # 7) Guardar
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # 8) QA report (opcional pero recomendado)
    rep = build_report(df_in=df_in, df_out=df, min_len=int(args.min_len))
    log(f"OK: creado {out_path} shape={df.shape}")
    log(f"QA: rows_in={rep.rows_in} rows_out={rep.rows_out} null_date_rate={rep.null_date_rate:.4f} empty_text_rate={rep.empty_text_rate:.4f}")
    if rep.sin_clasificar_rate is not None:
        log(f"QA: sin_clasificar_rate={rep.sin_clasificar_rate:.4f}")
    if rep.macro_source_breakdown:
        log(f"QA: macro_area_source={rep.macro_source_breakdown}")
    log(f"QA: date_range={rep.min_date} -> {rep.max_date}")

    if args.write_report:
        report_path = out_path.with_suffix(".report.json")
        _write_json(report_path, asdict(rep))
        log(f"QA report: {report_path}")

    # 9) Nota operativa: este output debe ser usado por forecast_trends
    log("Siguiente paso recomendado:")
    log(f"  python src/forecast_trends.py --input {out_path}  (y en UI cargar desde clean.parquet si aplica)")


if __name__ == "__main__":
    main()
