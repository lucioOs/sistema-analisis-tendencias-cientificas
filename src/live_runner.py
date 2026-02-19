# src/live_runner.py
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Tuple, List

import pandas as pd

# Fuente primaria (RSS). Debe aplicar rate limiting interno.
from src.ingest_rss import fetch_arxiv_rss

# Taxonomía única (macro_area) para TODO el sistema
from src.taxonomy import assign_macro_area, taxonomy_report

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


# =============================================================================
# Paths (alineados a src/config.py)
# - Dataset LIVE canónico: data/live/live_dataset.parquet
# - Meta LIVE: data/live/live_meta.json
#
# Nota:
#   En tu repo previo aparecía live_arxiv.parquet en processed. Para eliminar ambigüedad:
#   - LIVE must live in data/live/
#   - Histórico + artefactos en data/processed/
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LIVE_DIR = DATA_DIR / "live"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LIVE_DIR.mkdir(parents=True, exist_ok=True)

LIVE_DATASET = LIVE_DIR / "live_dataset.parquet"
LIVE_META = LIVE_DIR / "live_meta.json"


# =============================================================================
# Sources
# =============================================================================
ARXIV_RSS_URLS: list[str] = [
    "https://export.arxiv.org/rss/cs.AI",
    "https://export.arxiv.org/rss/cs.LG",
    "https://export.arxiv.org/rss/cs.CL",
    "https://export.arxiv.org/rss/cs.CV",
    "https://export.arxiv.org/rss/cs.SE",
    "https://export.arxiv.org/rss/cs.CR",
    "https://export.arxiv.org/rss/cs.DS",
    "https://export.arxiv.org/rss/stat.ML",
]
ARXIV_API_URL = "https://export.arxiv.org/api/query"


def _rss_url_to_cat(url: str) -> Optional[str]:
    parts = url.rstrip("/").split("/")
    cat = parts[-1].strip() if parts else ""
    return cat or None


# =============================================================================
# Schema (contrato LIVE)
# =============================================================================
REQUIRED_COLS: tuple[str, ...] = (
    "date",              # datetime (UTC)
    "title",             # str
    "abstract",          # str
    "text",              # str (title + abstract)
    "categories",        # str (csv o space separated)
    "primary_category",  # str (cs.LG)
    "link",              # str
    "id",                # str (arXiv id)
    "source",            # str (arxiv_rss/arxiv_api/...)
    "macro_area",        # str (taxonomía canónica)
    # columnas de trazabilidad opcionales (si taxonomy.py las genera):
    "macro_area_source",
    "macro_area_hits",
    "macro_area_score",
)

TEXT_COLS: tuple[str, ...] = tuple([c for c in REQUIRED_COLS if c != "date"])


# =============================================================================
# Config
# =============================================================================
@dataclass(frozen=True)
class RunConfig:
    # window + size
    days_back: int = 30
    max_keep: int = 20_000
    min_rows_warn: int = 20

    # behavior
    write_meta: bool = True
    strict: bool = False
    log_level: str = "INFO"

    # Fallback API
    api_fallback: bool = True
    api_page_size: int = 200
    api_max_total: int = 2000
    api_timeout_sec: int = 30
    api_retries: int = 3
    api_backoff_sec: float = 3.0
    api_polite_sleep_sec: float = 3.1

    # Demo mode (si queda vacío, fuerza algo razonable)
    force_non_empty: bool = True
    force_min_rows: int = 50

    # Normalización / calidad
    min_text_len: int = 20


# =============================================================================
# Logging
# =============================================================================
def _setup_logging(level: str) -> None:
    lvl = getattr(logging, str(level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# Helpers (I/O)
# =============================================================================
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding=encoding)
    os.replace(tmp, path)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logging.warning("No se pudo leer parquet '%s': %s", path, e)
        return pd.DataFrame()


def _write_meta(meta_path: Path, meta: dict) -> None:
    _atomic_write_text(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))


# =============================================================================
# DataFrame hygiene / contract enforcement
# =============================================================================
def _ensure_columns(df: pd.DataFrame, *, strict: bool) -> pd.DataFrame:
    """
    Garantiza el contrato LIVE:
    - crea columnas faltantes (o revienta si strict)
    - date -> datetime utc
    - strings -> str
    """
    out = df.copy()

    missing = [c for c in REQUIRED_COLS if c not in out.columns]
    if missing and strict:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    for c in missing:
        out[c] = pd.NaT if c == "date" else ""

    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True)
    out = out.dropna(subset=["date"]).reset_index(drop=True)

    for c in TEXT_COLS:
        out[c] = out[c].astype(str).fillna("")

    return out


def _normalize_text_fields(df: pd.DataFrame, *, min_text_len: int) -> pd.DataFrame:
    """
    Normaliza whitespace y rellena campos críticos:
    - text := title + abstract si está vacío / corto
    - primary_category desde categories si falta
    - id fallback
    """
    out = df.copy()

    # normalize whitespace
    for c in TEXT_COLS:
        out[c] = out[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # text fallback
    need_text = out["text"].str.len() < int(min_text_len)
    if need_text.any():
        out.loc[need_text, "text"] = (
            out.loc[need_text, "title"].fillna("")
            + ". "
            + out.loc[need_text, "abstract"].fillna("")
        ).astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # primary_category fallback: primer token de categories
    need_pc = out["primary_category"].str.len() == 0
    if need_pc.any():
        cats = out.loc[need_pc, "categories"].astype(str).str.replace(",", " ")
        out.loc[need_pc, "primary_category"] = cats.str.split().apply(lambda xs: xs[0] if xs else "")

    # id fallback
    need_id = out["id"].str.len() == 0
    if need_id.any():
        out.loc[need_id, "id"] = out.loc[need_id, "link"]

    still_need = out["id"].str.len() == 0
    if still_need.any():
        out.loc[still_need, "id"] = (
            out.loc[still_need, "title"]
            + "|"
            + out.loc[still_need, "date"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        )

    return out


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicación defensiva (id > link > title+date)."""
    if df.empty:
        return df

    out = df.sort_values("date").reset_index(drop=True)

    if "id" in out.columns:
        out = out.drop_duplicates(subset=["id"], keep="last")
    if "link" in out.columns:
        out = out.drop_duplicates(subset=["link"], keep="last")
    if {"title", "date"}.issubset(out.columns):
        out = out.drop_duplicates(subset=["title", "date"], keep="last")

    return out.sort_values("date").reset_index(drop=True)


def _trim(df: pd.DataFrame, max_keep: int) -> pd.DataFrame:
    if max_keep > 0 and len(df) > int(max_keep):
        return df.tail(int(max_keep)).reset_index(drop=True)
    return df


def _filter_days_back(df: pd.DataFrame, days_back: int) -> pd.DataFrame:
    if df.empty:
        return df
    now = pd.Timestamp.now(tz="UTC")
    cut = now - pd.Timedelta(days=int(days_back))
    return df[df["date"] >= cut].reset_index(drop=True)


# =============================================================================
# arXiv API (fallback)
# =============================================================================
def _api_http_get(url: str, params: dict, timeout_sec: int, user_agent: str) -> str:
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/atom+xml,application/xml,text/xml,*/*",
    }

    if requests is None:
        import urllib.parse
        import urllib.request

        q = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{url}?{q}", headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_sec) as r:
            return r.read().decode("utf-8", errors="replace")

    r = requests.get(url, params=params, headers=headers, timeout=timeout_sec)
    r.raise_for_status()
    return r.text


def _fetch_arxiv_api(
    *,
    cats: Sequence[str],
    days_back: int,
    page_size: int,
    max_total: int,
    timeout_sec: int,
    retries: int,
    backoff_sec: float,
    polite_sleep_sec: float,
    user_agent: str,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
) -> pd.DataFrame:
    """
    arXiv API oficial (Atom) con paginación y cutoff por fecha.
    """
    try:
        import feedparser  # type: ignore
    except ImportError:
        logging.error("Falta 'feedparser'. Instala: pip install feedparser")
        return pd.DataFrame()

    cats = [c for c in cats if c]
    if not cats:
        return pd.DataFrame()

    query = " OR ".join([f"cat:{c}" for c in cats])
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(days_back))

    all_rows: list[dict] = []
    start = 0
    page_size = int(max(1, page_size))
    max_total = int(max(0, max_total))

    while True:
        if max_total > 0 and len(all_rows) >= max_total:
            break

        max_results = page_size
        if max_total > 0:
            remaining = max_total - len(all_rows)
            max_results = min(max_results, remaining)

        params = {
            "search_query": query,
            "start": int(start),
            "max_results": int(max_results),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        entries = None
        last_err: Optional[Exception] = None

        for attempt in range(1, int(retries) + 1):
            try:
                if attempt > 1:
                    sleep_s = float(backoff_sec) * (2 ** (attempt - 2))
                    logging.warning("API retry %s/%s (sleep %.1fs)...", attempt, retries, sleep_s)
                    time.sleep(sleep_s)

                logging.info("arXiv API start=%s max_results=%s (intento %s)...", start, max_results, attempt)
                xml = _api_http_get(ARXIV_API_URL, params, timeout_sec=int(timeout_sec), user_agent=user_agent)
                feed = feedparser.parse(xml)
                entries = feed.get("entries", []) or []
                break
            except Exception as e:
                last_err = e
                entries = None

        if entries is None:
            logging.error("Error API en start=%s: %s", start, last_err)
            break

        if not entries:
            break

        oldest_in_page: Optional[pd.Timestamp] = None
        page_rows: list[dict] = []

        for entry in entries:
            dt = None
            if entry.get("published"):
                dt = pd.to_datetime(entry["published"], errors="coerce", utc=True)
            elif entry.get("updated"):
                dt = pd.to_datetime(entry["updated"], errors="coerce", utc=True)

            if dt is None or pd.isna(dt):
                continue

            dt_ts = pd.Timestamp(dt)
            if oldest_in_page is None or dt_ts < oldest_in_page:
                oldest_in_page = dt_ts

            title = (entry.get("title") or "").replace("\n", " ").strip()
            abstract = (entry.get("summary") or "").replace("\n", " ").strip()

            # link HTML
            link = entry.get("link") or ""
            for lk in (entry.get("links", []) or []):
                if lk.get("rel") == "alternate" and lk.get("type") == "text/html":
                    link = lk.get("href") or link
                    break

            tags = [t.get("term") for t in (entry.get("tags", []) or []) if t.get("term")]
            categories = ",".join(sorted(set(tags)))
            primary_category = str(tags[0]) if tags else ""

            raw_id = entry.get("id") or link
            clean_id = raw_id.split("/")[-1] if raw_id else f"{title[:20]}|{dt_ts.isoformat()}"

            page_rows.append(
                {
                    "date": dt_ts,
                    "title": title,
                    "abstract": abstract,
                    "text": f"{title}. {abstract}",
                    "categories": categories,
                    "primary_category": primary_category,
                    "link": link,
                    "id": clean_id,
                    "source": "arxiv_api",
                    "macro_area": "",
                }
            )

        if not page_rows:
            break

        all_rows.extend(page_rows)

        # arXiv policy: polite sleep
        time.sleep(float(polite_sleep_sec))

        if oldest_in_page is not None and oldest_in_page <= cutoff:
            break

        start += int(max_results)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


# =============================================================================
# Source fetch wrappers
# =============================================================================
def _fetch_from_rss(urls: Sequence[str], days_back: int) -> pd.DataFrame:
    """
    Wrapper defensivo para RSS.
    """
    try:
        df = fetch_arxiv_rss(list(urls), days_back=int(days_back))
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logging.exception("Fallo fetch_arxiv_rss: %s", e)
        return pd.DataFrame()


def _prepare_new_df(df_raw: pd.DataFrame, *, strict: bool, min_text_len: int, default_source: str) -> pd.DataFrame:
    """
    Pipeline único de normalización para “new_df”.
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=list(REQUIRED_COLS))

    df = df_raw.copy()
    if "source" not in df.columns:
        df["source"] = default_source
    else:
        df["source"] = df["source"].astype(str).replace({"": default_source})

    df = _ensure_columns(df, strict=strict)
    df = _normalize_text_fields(df, min_text_len=int(min_text_len))
    return df


# =============================================================================
# Main Logic
# =============================================================================
def run(cfg: RunConfig, urls: Optional[Sequence[str]] = None) -> int:
    urls = list(urls or ARXIV_RSS_URLS)

    # sanity
    if cfg.days_back <= 0:
        raise ValueError("days_back debe ser > 0")
    if cfg.max_keep < 0:
        raise ValueError("max_keep debe ser >= 0")
    if cfg.api_page_size <= 0:
        raise ValueError("api_page_size debe ser > 0")
    if cfg.api_max_total < 0:
        raise ValueError("api_max_total debe ser >= 0")
    if cfg.min_text_len <= 0:
        raise ValueError("min_text_len debe ser > 0")

    user_agent = os.getenv("ARXIV_USER_AGENT", "PredicTrends/1.0 (mailto:tu_email@dominio.com)")

    logging.info("Iniciando LIVE. RSS feeds=%s | days_back=%s", len(urls), cfg.days_back)

    # -------------------------------------------------------------------------
    # 1) RSS (primario)
    # -------------------------------------------------------------------------
    rss_raw = _fetch_from_rss(urls, days_back=int(cfg.days_back))
    rss_df = _prepare_new_df(
        rss_raw,
        strict=cfg.strict,
        min_text_len=int(cfg.min_text_len),
        default_source="arxiv_rss",
    )
    rss_df = _filter_days_back(rss_df, int(cfg.days_back))

    new_df = rss_df
    new_source = "rss"

    # -------------------------------------------------------------------------
    # 2) API fallback (si RSS queda vacío)
    # -------------------------------------------------------------------------
    if cfg.api_fallback and new_df.empty:
        cats = [c for c in (_rss_url_to_cat(u) for u in urls) if c]
        logging.warning("RSS vacío. Fallback API cats=%s", cats)

        api_raw = _fetch_arxiv_api(
            cats=cats,
            days_back=int(cfg.days_back),
            page_size=int(cfg.api_page_size),
            max_total=int(cfg.api_max_total),
            timeout_sec=int(cfg.api_timeout_sec),
            retries=int(cfg.api_retries),
            backoff_sec=float(cfg.api_backoff_sec),
            polite_sleep_sec=float(cfg.api_polite_sleep_sec),
            user_agent=user_agent,
        )
        api_df = _prepare_new_df(
            api_raw,
            strict=False,
            min_text_len=int(cfg.min_text_len),
            default_source="arxiv_api",
        )
        api_df = _filter_days_back(api_df, int(cfg.days_back))

        if not api_df.empty:
            new_df = api_df
            new_source = "api"

    # -------------------------------------------------------------------------
    # 3) Demo mode (no vacío)
    # -------------------------------------------------------------------------
    if cfg.force_non_empty and new_df.empty:
        cats = [c for c in (_rss_url_to_cat(u) for u in urls) if c]
        logging.warning("LIVE vacío. Forzando API (últimos %s registros)...", cfg.force_min_rows)

        force_raw = _fetch_arxiv_api(
            cats=cats,
            days_back=3650,
            page_size=min(int(cfg.api_page_size), max(int(cfg.force_min_rows), 10)),
            max_total=max(int(cfg.force_min_rows), 10),
            timeout_sec=int(cfg.api_timeout_sec),
            retries=int(cfg.api_retries),
            backoff_sec=float(cfg.api_backoff_sec),
            polite_sleep_sec=float(cfg.api_polite_sleep_sec),
            user_agent=user_agent,
        )
        force_df = _prepare_new_df(
            force_raw,
            strict=False,
            min_text_len=int(cfg.min_text_len),
            default_source="arxiv_api",
        )
        if not force_df.empty:
            new_df = force_df.tail(int(cfg.force_min_rows)).reset_index(drop=True)
            new_source = "api_force"

    # -------------------------------------------------------------------------
    # 4) Merge incremental con dataset previo
    # -------------------------------------------------------------------------
    old_raw = _safe_read_parquet(LIVE_DATASET)
    old_df = _prepare_new_df(
        old_raw,
        strict=False,
        min_text_len=int(cfg.min_text_len),
        default_source="old",
    ) if (old_raw is not None and not old_raw.empty) else pd.DataFrame(columns=list(REQUIRED_COLS))

    old_ids = set(old_df["id"].astype(str)) if not old_df.empty else set()
    new_ids = set(new_df["id"].astype(str)) if not new_df.empty else set()
    real_new_count = int(len(new_ids - old_ids))

    merged = pd.concat([d for d in (old_df, new_df) if d is not None and not d.empty], ignore_index=True) \
        if ((old_df is not None and not old_df.empty) or (new_df is not None and not new_df.empty)) \
        else pd.DataFrame(columns=list(REQUIRED_COLS))

    merged = _dedup(merged)
    merged = _trim(merged, int(cfg.max_keep))

    # -------------------------------------------------------------------------
    # 5) Taxonomía única: asignar macro_area SIEMPRE
    # -------------------------------------------------------------------------
    try:
        merged = assign_macro_area(merged, cat_col="primary_category", text_col="text", out_col="macro_area")
    except Exception as e:
        logging.exception("Fallo assign_macro_area (se deja macro_area='Sin clasificar'): %s", e)
        if "macro_area" not in merged.columns:
            merged["macro_area"] = "Sin clasificar"

    # Garantías finales
    merged = _ensure_columns(merged, strict=False)
    merged = merged.sort_values("date").reset_index(drop=True)

    # Persist
    _atomic_write_parquet(merged, LIVE_DATASET)

    # -------------------------------------------------------------------------
    # 6) Meta (trazabilidad)
    # -------------------------------------------------------------------------
    if cfg.write_meta:
        # Reporte de cobertura taxonómica (QA)
        try:
            tax_rep = taxonomy_report(merged, cat_col="primary_category", macro_col="macro_area", source_col="macro_area_source")
        except Exception:
            tax_rep = {}

        meta = {
            "updated_at_utc": _now_utc_iso(),
            "days_back": int(cfg.days_back),
            "source_used": str(new_source),
            "total_rows": int(len(merged)),
            "new_fetched": int(len(new_df)) if new_df is not None else 0,
            "new_unique": int(real_new_count),
            "min_date": str(merged["date"].min()) if not merged.empty else None,
            "max_date": str(merged["date"].max()) if not merged.empty else None,
            "parquet": str(LIVE_DATASET),
            "rss_feeds": list(urls),

            "config": asdict(cfg),

            "api": {
                "enabled": bool(cfg.api_fallback),
                "page_size": int(cfg.api_page_size),
                "max_total": int(cfg.api_max_total),
                "timeout_sec": int(cfg.api_timeout_sec),
                "retries": int(cfg.api_retries),
                "backoff_sec": float(cfg.api_backoff_sec),
                "polite_sleep_sec": float(cfg.api_polite_sleep_sec),
                "user_agent": user_agent,
            },

            "taxonomy_report": tax_rep,
        }
        _write_meta(LIVE_META, meta)

    if real_new_count < int(cfg.min_rows_warn):
        logging.warning(
            "Pocos registros nuevos (únicos): %s (min_rows_warn=%s). source=%s",
            real_new_count,
            cfg.min_rows_warn,
            new_source,
        )

    logging.info(
        "Fin LIVE. fuente=%s | nuevos_unicos=%s | total=%s | salida=%s",
        new_source,
        real_new_count,
        len(merged),
        LIVE_DATASET,
    )
    return 0


# =============================================================================
# CLI
# =============================================================================
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingesta LIVE arXiv RSS/API -> data/live/live_dataset.parquet")

    # ventana + size
    p.add_argument("--days-back", type=int, default=30, help="Ventana de días (default: 30)")
    p.add_argument("--max-keep", type=int, default=20_000, help="Máximo de filas a conservar (0=sin recorte)")
    p.add_argument("--min-rows-warn", type=int, default=20, help="Warn si entran pocos nuevos únicos")

    # comportamiento
    p.add_argument("--no-meta", action="store_true", help="No escribir live_meta.json")
    p.add_argument("--strict", action="store_true", help="Fallar si faltan columnas requeridas")
    p.add_argument("--log-level", default="INFO", help="DEBUG|INFO|WARNING|ERROR")

    # API
    p.add_argument("--no-api", action="store_true", help="Deshabilitar fallback arXiv API")
    p.add_argument("--api-page-size", type=int, default=200, help="Resultados por página del API (default: 200)")
    p.add_argument("--api-max-total", type=int, default=2000, help="Tope total acumulado (default: 2000)")
    p.add_argument("--api-timeout-sec", type=int, default=30, help="Timeout por request (default: 30)")
    p.add_argument("--api-retries", type=int, default=3, help="Reintentos (default: 3)")
    p.add_argument("--api-backoff-sec", type=float, default=3.0, help="Backoff base (default: 3.0)")
    p.add_argument("--api-polite-sleep-sec", type=float, default=3.1, help="Sleep entre requests (default: 3.1)")

    # demo mode
    p.add_argument("--no-force-non-empty", action="store_true", help="Deshabilitar modo demo (permitir vacío)")
    p.add_argument("--force-min-rows", type=int, default=50, help="Mínimo de filas en modo demo (default: 50)")

    # calidad
    p.add_argument("--min-text-len", type=int, default=20, help="Longitud mínima del texto final (default: 20)")

    # custom RSS urls
    p.add_argument(
        "--url",
        action="append",
        default=None,
        help="Agrega una URL RSS (se puede repetir). Si no se usa, toma el set por defecto.",
    )
    return p


def main(argv: list[str]) -> int:
    args = _build_argparser().parse_args(argv)

    cfg = RunConfig(
        days_back=int(args.days_back),
        max_keep=int(args.max_keep),
        min_rows_warn=int(args.min_rows_warn),
        write_meta=not bool(args.no_meta),
        strict=bool(args.strict),
        log_level=str(args.log_level),

        api_fallback=not bool(args.no_api),
        api_page_size=int(args.api_page_size),
        api_max_total=int(args.api_max_total),
        api_timeout_sec=int(args.api_timeout_sec),
        api_retries=int(args.api_retries),
        api_backoff_sec=float(args.api_backoff_sec),
        api_polite_sleep_sec=float(args.api_polite_sleep_sec),

        force_non_empty=not bool(args.no_force_non_empty),
        force_min_rows=int(args.force_min_rows),

        min_text_len=int(args.min_text_len),
    )

    _setup_logging(cfg.log_level)
    urls = args.url if args.url else ARXIV_RSS_URLS
    return run(cfg, urls=urls)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
