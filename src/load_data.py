# src/load_data.py
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

# IMPORTANTE: importar desde src.*
from src.taxonomy import ensure_primary_category, assign_macro_area, DEFAULT_MACRO_AREA


# =============================================================================
# Logging profesional (simple, consistente, sin dependencias)
# =============================================================================
def log(level: str, msg: str) -> None:
    print(f"[{level}] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    log("ERROR", msg)
    raise SystemExit(code)


# =============================================================================
# Paths del proyecto
# =============================================================================
@dataclass(frozen=True)
class ProjectPaths:
    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")

    def ensure(self) -> None:
        self.raw.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Heurísticas para detección de columnas
# =============================================================================
TEXT_KEYS = [
    "text",
    "content",
    "body",
    "article",
    "summary",
    "description",
    "abstract",
    "headline",
    "heading",
    "title",
]

DATE_KEYS = [
    "date",
    "published",
    "published_at",
    "publish_date",
    "time",
    "datetime",
    "created",
    "created_at",
    "update_date",
    "updated",
    "updated_at",
    "year",
]

TITLE_KEYS = ["title", "headline", "heading"]
CATEGORY_KEYS = ["primary_category", "categories", "category", "arxiv_cat"]
ID_KEYS = ["id", "paper_id", "doc_id", "identifier"]
SOURCE_KEYS = ["source", "origin", "dataset"]


def normalize_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s


def find_best_column(columns: Sequence[str], keys: Sequence[str]) -> Optional[str]:
    """
    Selecciona columna por:
      1) match exacto (normalizado)
      2) contains match (key dentro del nombre)
    """
    norm = {normalize_name(c): c for c in columns}
    cols_norm = list(norm.keys())

    for k in keys:
        kn = normalize_name(k)
        if kn in norm:
            return norm[kn]

    for c in cols_norm:
        for k in keys:
            if normalize_name(k) in c:
                return norm[c]

    return None


# =============================================================================
# Lectura robusta
# =============================================================================
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        die(f"No existe el archivo: {path}")

    encs = ["utf-8", "utf-8-sig", "latin-1"]
    last = None
    for enc in encs:
        try:
            df = pd.read_csv(path, encoding=enc)
            log("INFO", f"CSV leído: {path.name} | encoding={enc} | shape={df.shape}")
            return df
        except Exception as e:
            last = e

    die(f"No pude leer el CSV {path.name}. Último error: {last}")
    return pd.DataFrame()  # unreachable


def safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        die(f"No existe el archivo: {path}")
    try:
        df = pd.read_parquet(path)
        log("INFO", f"Parquet leído: {path.name} | shape={df.shape}")
        return df
    except Exception as e:
        die(f"No pude leer el Parquet {path.name}. Error: {e}")
        return pd.DataFrame()  # unreachable


# =============================================================================
# Normalización de fechas / texto
# =============================================================================
def parse_dates(series: pd.Series, col_name: str) -> pd.Series:
    """
    Convierte fechas de forma robusta:
    - si la columna parece 'year', construye YYYY-01-01
    - intenta parseo general
    - devuelve datetime naive (consistente para parquet y agregación por periodos)
    """
    s = series.copy()

    if "year" in normalize_name(col_name):
        s = s.astype(str).str.extract(r"(\d{4})")[0]
        s = pd.to_datetime(s + "-01-01", errors="coerce")
        return s

    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_convert(None)


def normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# =============================================================================
# Contrato y “enforcer”
# =============================================================================
REQUIRED_MIN_COLS = ["date", "text_clean", "primary_category", "macro_area"]


def enforce_schema(
    df: pd.DataFrame,
    *,
    date_col: Optional[str] = None,
    title_col: Optional[str] = None,
    text_col: Optional[str] = None,
    category_col: Optional[str] = None,
    id_col: Optional[str] = None,
    source_value: Optional[str] = None,
    min_text_len: int = 20,
    drop_duplicates: bool = False,
) -> pd.DataFrame:
    """
    Estandariza cualquier DF (hist o live o csv externo) al contrato del proyecto:
      - date (datetime naive)
      - text_clean (string)
      - primary_category (token arXiv cuando exista)
      - macro_area (canónico) + trazabilidad (source/hits/score)

    Si ya vienen columnas del contrato, las respeta y solo repara faltantes.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_MIN_COLS)

    df = df.copy()
    cols = list(df.columns)

    # 1) Detectar columnas si no vienen forzadas
    dcol = date_col or find_best_column(cols, DATE_KEYS)
    tcol = title_col or find_best_column(cols, TITLE_KEYS)
    xcol = text_col or find_best_column(cols, ["text_clean", "text"] + TEXT_KEYS)
    ccol = category_col or find_best_column(cols, CATEGORY_KEYS)
    icol = id_col or find_best_column(cols, ID_KEYS)

    # 2) date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif dcol and dcol in df.columns:
        df["date"] = parse_dates(df[dcol], dcol)
    else:
        df["date"] = pd.NaT

    # 3) text_clean (preferir text_clean; si no, usar text; si no, title+body)
    if "text_clean" in df.columns:
        df["text_clean"] = normalize_text(df["text_clean"])
    else:
        if "text" in df.columns:
            base = normalize_text(df["text"])
        else:
            title_part = normalize_text(df[tcol]) if tcol and tcol in df.columns else pd.Series([""] * len(df), index=df.index)
            body_part = normalize_text(df[xcol]) if xcol and xcol in df.columns else pd.Series([""] * len(df), index=df.index)

            if tcol and xcol and (tcol != xcol) and (tcol in df.columns) and (xcol in df.columns):
                base = (title_part + ". " + body_part).str.strip()
            else:
                base = body_part if (xcol and xcol in df.columns) else title_part

        df["text_clean"] = base

    # 4) primary_category
    if "primary_category" not in df.columns:
        if ccol and ccol in df.columns:
            df["primary_category"] = df[ccol]
        else:
            df["primary_category"] = None

    df = ensure_primary_category(
        df,
        possible_cols=("primary_category", "categories", "category", "arxiv_cat"),
        out_col="primary_category",
    )

    # 5) macro_area (API NUEVA: sin text_col, usa coalesce interno)
    #    Siempre recalculamos si no existe o si viene con NaNs.
    if ("macro_area" not in df.columns) or df["macro_area"].isna().any():
        df = assign_macro_area(
            df,
            cat_col="primary_category",
            out_col="macro_area",
            source_col="macro_area_source",
            hits_col="macro_area_hits",
            score_col="macro_area_score",
            text_cols_priority=("text_clean", "text", "title", "abstract"),
        )

    df["macro_area"] = df["macro_area"].fillna(DEFAULT_MACRO_AREA)

    # 6) id + source
    if "id" not in df.columns:
        if icol and icol in df.columns:
            df["id"] = df[icol].astype(str)
        else:
            df["id"] = ""
    else:
        df["id"] = df["id"].astype(str).fillna("")

    if "source" not in df.columns:
        if source_value:
            df["source"] = str(source_value)
        else:
            df["source"] = ""
    else:
        df["source"] = df["source"].astype(str).fillna("")

    # 7) Limpieza final + filtros
    df["text_clean"] = normalize_text(df["text_clean"])
    df = df.dropna(subset=["date"])
    df = df[df["text_clean"].str.len() >= int(min_text_len)].copy()

    if drop_duplicates:
        if df["id"].astype(str).str.len().gt(0).any():
            df = df.drop_duplicates(subset=["id"], keep="last")
        else:
            df = df.drop_duplicates(subset=["text_clean", "date"], keep="last")

    # 8) Orden + columnas mínimas primero (sin tirar extras)
    df = df.sort_values("date").reset_index(drop=True)
    cols_first = [
        "date",
        "id",
        "source",
        "primary_category",
        "macro_area",
        "macro_area_source",
        "macro_area_hits",
        "macro_area_score",
        "text_clean",
    ]
    cols_first = [c for c in cols_first if c in df.columns]
    remaining = [c for c in df.columns if c not in cols_first]
    df = df[cols_first + remaining]

    return df


# =============================================================================
# Metadata / utilidades
# =============================================================================
def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_meta(
    *,
    dataset_name: str,
    file_name: str,
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    selected: Dict[str, Any],
    min_text_len: int,
) -> dict:
    return {
        "dataset": dataset_name,
        "file": file_name,
        "rows_input": int(len(df_in)),
        "rows_output": int(len(df_out)),
        "columns_input": list(df_in.columns),
        "columns_output": list(df_out.columns),
        "selected": selected,
        "quality": {
            "null_date_rate_output": float(df_out["date"].isna().mean()) if "date" in df_out.columns else None,
            "min_text_len": int(min_text_len),
            "macro_area_null_rate": float(df_out["macro_area"].isna().mean()) if "macro_area" in df_out.columns else None,
            "macro_area_sin_clasificar_rate": float((df_out["macro_area"] == DEFAULT_MACRO_AREA).mean())
            if "macro_area" in df_out.columns else None,
        },
        "date_range": {
            "min": str(df_out["date"].min()) if "date" in df_out.columns else None,
            "max": str(df_out["date"].max()) if "date" in df_out.columns else None,
        },
    }


# =============================================================================
# Rebuild macro (para histórico ya existente)
# =============================================================================
def rebuild_macro_for_file(
    inp: Path,
    out: Optional[Path],
    *,
    min_text_len: int = 20,
    drop_duplicates: bool = False,
) -> Path:
    if not inp.exists():
        die(f"No existe input: {inp}")

    df_in = safe_read_parquet(inp)
    if df_in.empty:
        die("El parquet de entrada está vacío.")

    df_out = enforce_schema(
        df_in,
        min_text_len=int(min_text_len),
        drop_duplicates=bool(drop_duplicates),
    )

    if df_out.empty:
        die("Tras rebuild_macro, el dataset quedó vacío. Revisa columnas/texto/min_text_len.")

    out_path = out if out is not None else inp
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)

    meta = build_meta(
        dataset_name="rebuild_macro",
        file_name=inp.name,
        df_in=df_in,
        df_out=df_out,
        selected={"mode": "rebuild_macro", "input": str(inp), "output": str(out_path)},
        min_text_len=int(min_text_len),
    )
    write_json(Path("data/processed") / "rebuild_macro_meta.json", meta)

    log("INFO", f"✅ rebuild_macro OK: {out_path} rows={len(df_out)} cols={len(df_out.columns)}")
    return out_path


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Carga/Normalización robusta de dataset (CSV o Parquet) al contrato del proyecto"
    )

    # Modo 1: normalizar desde data/raw (CSV/Parquet externo)
    ap.add_argument("--file", default="mit_ai_news.csv", help="Nombre del archivo dentro de data/raw (CSV o Parquet)")
    ap.add_argument("--dataset_name", default="external_dataset", help="Nombre lógico del dataset (para metadata)")
    ap.add_argument("--out", default="data/processed/dataset.parquet", help="Ruta de salida parquet estandarizado")

    ap.add_argument("--text_col", default="", help="Forzar columna de texto/cuerpo (opcional)")
    ap.add_argument("--date_col", default="", help="Forzar columna de fecha (opcional)")
    ap.add_argument("--title_col", default="", help="Forzar columna de título (opcional)")
    ap.add_argument("--category_col", default="", help="Forzar columna de categoría (opcional)")
    ap.add_argument("--id_col", default="", help="Forzar columna id (opcional)")

    ap.add_argument("--source", default="", help="Valor fijo para columna source (opcional)")
    ap.add_argument("--drop_duplicates", action="store_true", help="Eliminar duplicados (id o text_clean+date)")
    ap.add_argument("--min_text_len", type=int, default=20, help="Longitud mínima del texto final")

    # Modo 2: rebuild macro para un parquet ya existente (ej. clean.parquet)
    ap.add_argument(
        "--rebuild-macro",
        action="store_true",
        help="Recalcula primary_category y macro_area en un parquet existente (por defecto sobre data/processed/clean.parquet).",
    )
    ap.add_argument(
        "--rebuild-in",
        default="data/processed/clean.parquet",
        help="Input parquet para --rebuild-macro",
    )
    ap.add_argument(
        "--rebuild-out",
        default="",
        help="Output parquet para --rebuild-macro (vacío = sobreescribir input)",
    )

    args = ap.parse_args()

    P = ProjectPaths()
    P.ensure()

    # -------------------------
    # MODO rebuild macro
    # -------------------------
    if bool(args.rebuild_macro):
        inp = Path(args.rebuild_in)
        out = Path(args.rebuild_out) if str(args.rebuild_out).strip() else None
        rebuild_macro_for_file(
            inp,
            out,
            min_text_len=int(args.min_text_len),
            drop_duplicates=bool(args.drop_duplicates),
        )
        return

    # -------------------------
    # MODO normalización desde data/raw
    # -------------------------
    src_path = P.raw / args.file
    if not src_path.exists():
        die(f"No existe: {src_path}")

    if src_path.suffix.lower() in (".parquet", ".pq"):
        df_in = safe_read_parquet(src_path)
    else:
        df_in = safe_read_csv(src_path)

    cols = list(df_in.columns)
    log("INFO", f"Columnas detectadas: {cols}")

    df_out = enforce_schema(
        df_in,
        date_col=args.date_col.strip() or None,
        title_col=args.title_col.strip() or None,
        text_col=args.text_col.strip() or None,
        category_col=args.category_col.strip() or None,
        id_col=args.id_col.strip() or None,
        source_value=args.source.strip() or None,
        min_text_len=int(args.min_text_len),
        drop_duplicates=bool(args.drop_duplicates),
    )

    if df_out.empty:
        die("Tras normalización, el dataset quedó vacío. Revisa columnas reales o baja --min_text_len.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)

    selected = {
        "date_col": args.date_col.strip() or find_best_column(cols, DATE_KEYS),
        "title_col": args.title_col.strip() or find_best_column(cols, TITLE_KEYS),
        "text_col": args.text_col.strip() or find_best_column(cols, ["text_clean", "text"] + TEXT_KEYS),
        "category_col": args.category_col.strip() or find_best_column(cols, CATEGORY_KEYS),
        "id_col": args.id_col.strip() or find_best_column(cols, ID_KEYS),
        "source_value": args.source.strip() or None,
    }
    meta = build_meta(
        dataset_name=str(args.dataset_name),
        file_name=src_path.name,
        df_in=df_in,
        df_out=df_out,
        selected=selected,
        min_text_len=int(args.min_text_len),
    )
    write_json(P.processed / "dataset_schema.json", meta)

    log("INFO", f"✅ parquet estandarizado: {out_path}")
    log("INFO", f"Filas salida: {len(df_out)}")
    if "date" in df_out.columns and not df_out.empty:
        log("INFO", f"Rango fechas: {df_out['date'].min()} -> {df_out['date'].max()}")
    log("INFO", f"Metadata: {P.processed / 'dataset_schema.json'}")


if __name__ == "__main__":
    main()
