# src/ingest_arxiv_kaggle.py
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
from email.utils import parsedate_to_datetime

from taxonomy import ensure_primary_category, assign_macro_area


# Categorías default (filtro rápido en ingest)
DEFAULT_CAT_PREFIXES = (
    "cs.AI",
    "cs.LG",
    "stat.ML",
    "cs.CL",
    "cs.CV",
    "cs.RO",
    "cs.IR",
    "cs.NE",
    "cs.CR",
    "cs.DC",
    "cs.SE",
)


def parse_any_date(rec: dict[str, Any]) -> Optional[pd.Timestamp]:
    """
    Intenta obtener una fecha estable (naive, sin tz) desde:
      1) versions[0].created (estilo RFC 2822 en arXiv)
      2) update_date
      3) created
    """
    versions = rec.get("versions") or []
    if isinstance(versions, list) and versions:
        v0 = versions[0] if isinstance(versions[0], dict) else {}
        created = v0.get("created")
        if created:
            try:
                dt = parsedate_to_datetime(created)
                return pd.Timestamp(dt).tz_convert(None) if dt.tzinfo else pd.Timestamp(dt)
            except Exception:
                pass

    upd = rec.get("update_date")
    if upd:
        try:
            return pd.to_datetime(upd, errors="coerce", utc=True).tz_convert(None)
        except Exception:
            pass

    cre = rec.get("created")
    if cre:
        try:
            return pd.to_datetime(cre, errors="coerce", utc=True).tz_convert(None)
        except Exception:
            pass

    return None


def open_text_maybe_gz(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def norm_categories(cat: Any) -> str:
    """
    Normaliza 'categories' a string (arXiv suele traer una lista/str con tokens tipo cs.LG).
    """
    if cat is None:
        return ""
    if isinstance(cat, (list, tuple, set)):
        return " ".join(str(x) for x in cat if x is not None)
    return str(cat)


def categories_match(cat_str: str, prefixes: Sequence[str]) -> bool:
    """
    Filtro simple por inclusión de prefijo.
    Ej: cat_str = "cs.LG cs.AI", prefixes = ["cs.LG"] -> True
    """
    if not prefixes:
        return True
    return any(p in cat_str for p in prefixes)


def build_text(title: Any, abstract: Any) -> str:
    t = str(title or "").strip()
    a = str(abstract or "").strip()
    if t and a:
        return f"{t}. {a}".strip()
    return (t or a).strip()


def enforce_minimal_schema_and_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deja el histórico listo para el pipeline:
    - date (datetime)
    - text (string) y text_clean (string provisional)
    - primary_category (token cs.XX)
    - macro_area + trazabilidad
    """
    # Tipos
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["text"] = df["text"].fillna("").astype(str)

    # text_clean provisional (tu preprocess puede hacer limpieza real)
    # Mantenerlo sencillo aquí para no duplicar responsabilidades
    df["text_clean"] = df["text"]

    # Asegurar primary_category (extrae token de categories)
    if "primary_category" not in df.columns:
        df["primary_category"] = df.get("categories")
    df = ensure_primary_category(
        df,
        possible_cols=("primary_category", "categories", "category", "arxiv_cat"),
        out_col="primary_category",
    )

    # Asignar macro_area (category-first + fallback keywords)
    df = assign_macro_area(
        df,
        cat_col="primary_category",
        text_col="text_clean",
        out_col="macro_area",
        source_col="macro_area_source",
        hits_col="macro_area_hits",
        score_col="macro_area_score",
    )
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/raw/arxiv-metadata-oai-snapshot.json",
        help="Ruta al .json o .json.gz (JSONL) de Kaggle",
    )
    ap.add_argument(
        "--out",
        default="data/processed/dataset.parquet",
        help="Salida parquet compatible con tu pipeline",
    )
    ap.add_argument(
        "--min_text_len",
        type=int,
        default=50,
        help="Longitud mínima del texto (title+abstract)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = sin límite; >0 para pruebas rápidas",
    )
    ap.add_argument("--since", default=None, help="Fecha mínima (YYYY-MM-DD)")
    ap.add_argument("--until", default=None, help="Fecha máxima (YYYY-MM-DD)")
    ap.add_argument(
        "--cat-prefix",
        action="append",
        default=None,
        help="Prefijo de categoría a incluir (repetible). Ej: --cat-prefix cs.AI",
    )
    ap.add_argument(
        "--use-default-cats",
        action="store_true",
        help="Usa lista default de categorías de computación/IA",
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help="Imprime un resumen de cobertura de macro_area al final",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] No existe el archivo: {input_path}")
        return 2

    since_ts = pd.to_datetime(args.since) if args.since else None
    until_ts = pd.to_datetime(args.until) if args.until else None

    prefixes: list[str] = []
    if args.use_default_cats:
        prefixes.extend(DEFAULT_CAT_PREFIXES)
    if args.cat_prefix:
        prefixes.extend(args.cat_prefix)

    rows: list[dict[str, Any]] = []
    bad_json = 0
    no_date = 0
    short_text = 0
    cat_skipped = 0
    date_skipped = 0

    with open_text_maybe_gz(input_path) as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception:
                bad_json += 1
                continue

            # Categorías (filtrar primero para ahorrar)
            cat_str = norm_categories(rec.get("categories"))
            if prefixes and not categories_match(cat_str, prefixes):
                cat_skipped += 1
                continue

            text = build_text(rec.get("title"), rec.get("abstract"))
            if len(text) < int(args.min_text_len):
                short_text += 1
                continue

            dt = parse_any_date(rec)
            if dt is None or pd.isna(dt):
                no_date += 1
                continue

            # Filtro por fechas
            if since_ts is not None and dt < since_ts:
                date_skipped += 1
                continue
            if until_ts is not None and dt > until_ts:
                date_skipped += 1
                continue

            rows.append(
                {
                    "date": dt,
                    "text": text,
                    "source": "arxiv_kaggle",
                    "id": rec.get("id"),
                    # conservar ambas: categories completo + primary_category (se extrae luego)
                    "categories": rec.get("categories"),
                    "primary_category": rec.get("categories"),
                    # opcionales útiles (si luego quieres)
                    "title": (rec.get("title") or "").strip() if isinstance(rec.get("title"), str) else rec.get("title"),
                }
            )

            if i % 100_000 == 0:
                print(
                    f"[INFO] Leídas {i:,} | válidas {len(rows):,} | "
                    f"cat_skip {cat_skipped:,} | date_skip {date_skipped:,} | "
                    f"sin_fecha {no_date:,} | json_malos {bad_json:,}"
                )

            if args.limit and len(rows) >= int(args.limit):
                break

    if not rows:
        print("[ERROR] No se generó ninguna fila válida.")
        print(f"  - json malos: {bad_json}")
        print(f"  - sin fecha: {no_date}")
        print(f"  - texto corto: {short_text}")
        print(f"  - filtradas por categoría: {cat_skipped}")
        print(f"  - filtradas por fecha: {date_skipped}")
        return 3

    df = pd.DataFrame(rows)

    # Deduplicación robusta
    if "id" in df.columns and df["id"].notna().any():
        df = df.drop_duplicates(subset=["id"], keep="last")
    else:
        df = df.drop_duplicates(subset=["date", "text"], keep="last")

    # Limpieza base y orden
    df = df.dropna(subset=["date", "text"]).sort_values("date").reset_index(drop=True)

    # Enforce schema + taxonomy (macro_area lista para todo el pipeline)
    df = enforce_minimal_schema_and_taxonomy(df)

    # Orden de columnas (contrato)
    cols_order = [
        "date",
        "id",
        "source",
        "categories",
        "primary_category",
        "title",
        "text",
        "text_clean",
        "macro_area",
        "macro_area_source",
        "macro_area_hits",
        "macro_area_score",
    ]
    cols_order = [c for c in cols_order if c in df.columns] + [c for c in df.columns if c not in cols_order]
    df = df[cols_order]

    # Export
    df.to_parquet(out_path, index=False)

    print(
        f"[OK] Exportado: {out_path} | rows={len(df):,} | "
        f"min={df['date'].min()} | max={df['date'].max()} | "
        f"cat_prefixes={prefixes if prefixes else 'NONE'}"
    )

    if args.report:
        # Resumen corto (sin depender de taxonomy_report si no quieres import adicional)
        macro_counts = df["macro_area"].value_counts(dropna=False).head(20)
        src_counts = df["macro_area_source"].value_counts(dropna=False)

        sin_clas = int(df["macro_area"].eq("Sin clasificar").sum())
        total = len(df)
        pct = (sin_clas / total * 100.0) if total else 0.0

        print("\n[REPORT] macro_area (top 20):")
        print(macro_counts.to_string())
        print("\n[REPORT] macro_area_source:")
        print(src_counts.to_string())
        print(f"\n[REPORT] Sin clasificar: {sin_clas:,}/{total:,} ({pct:.2f}%)")

        # Top categorías sin mapear (si quedaron en Sin clasificar)
        mask = df["macro_area"].eq("Sin clasificar")
        if mask.any():
            top_unmapped = (
                df.loc[mask, "primary_category"]
                .fillna("None")
                .astype(str)
                .value_counts()
                .head(15)
            )
            print("\n[REPORT] primary_category en Sin clasificar (top 15):")
            print(top_unmapped.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
