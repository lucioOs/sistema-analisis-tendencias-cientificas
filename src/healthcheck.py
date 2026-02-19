# src/healthcheck.py
# -----------------------------------------------------------------------------
# Healthcheck del pipeline (nivel ingeniería)
# - Valida existencia de artefactos (histórico + live)
# - Valida contrato mínimo de columnas / tipos
# - Valida que haya datos suficientes para series (>= 2 periodos)
# - Valida macro_area (si aplica) y distribución básica
# - Reporta KPIs de calidad (nulos, duplicados, rango fechas)
#
# Uso:
#   python -m src.healthcheck
#   python -m src.healthcheck --strict
#   python -m src.healthcheck --no-live
#   python -m src.healthcheck --freq M
#
# Return codes:
#   0 = OK
#   2 = WARN (no estricto)
#   3 = FAIL
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.config import HIST_DATASET, LIVE_DATASET, LIVE_META, HIST_YEARS_KEEP
from src.data.io import file_exists


# =============================================================================
# Output formatting
# =============================================================================
def _p(line: str = "") -> None:
    print(line, flush=True)


def _ok(msg: str) -> None:
    _p(f"[OK]   {msg}")


def _warn(msg: str) -> None:
    _p(f"[WARN] {msg}")


def _fail(msg: str) -> None:
    _p(f"[FAIL] {msg}")


# =============================================================================
# Configuration (contracts)
# =============================================================================
HIST_REQUIRED_COLS: Tuple[str, ...] = (
    "date",
    "text",
    "text_clean",
)
LIVE_REQUIRED_COLS: Tuple[str, ...] = (
    "date",
    "title",
    "abstract",
    "text",
    "categories",
    "primary_category",
    "link",
    "id",
    "source",
    "macro_area",
)

# columnas recomendadas (no obligatorias)
HIST_RECOMMENDED_COLS: Tuple[str, ...] = (
    "macro_area",
    "primary_category",
    "categories",
    "id",
    "title",
    "abstract",
    "link",
    "source",
)
LIVE_RECOMMENDED_COLS: Tuple[str, ...] = (
    "text_clean",  # si en algún punto lo aplicas al live también
)


# =============================================================================
# Data helpers
# =============================================================================
def _safe_read_parquet(path: Path, cols: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if cols:
            return pd.read_parquet(str(path), columns=cols)
        return pd.read_parquet(str(path))
    except Exception:
        try:
            return pd.read_parquet(str(path))
        except Exception:
            return pd.DataFrame()


def _coerce_date(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
    return df


def _period_count(df: pd.DataFrame, freq: str) -> int:
    if df.empty or "date" not in df.columns:
        return 0
    d = pd.to_datetime(df["date"], errors="coerce")
    d = d.dropna()
    if d.empty:
        return 0
    try:
        return int(d.dt.to_period(freq).nunique())
    except Exception:
        return 0


def _df_stats(df: pd.DataFrame, id_col: Optional[str] = None) -> Dict[str, object]:
    if df.empty:
        return {
            "rows": 0,
            "cols": 0,
            "min_date": None,
            "max_date": None,
            "null_date_rate": None,
            "dup_rate": None,
        }

    out: Dict[str, object] = {}
    out["rows"] = int(len(df))
    out["cols"] = int(df.shape[1])

    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        out["min_date"] = str(d.min()) if not d.dropna().empty else None
        out["max_date"] = str(d.max()) if not d.dropna().empty else None
        out["null_date_rate"] = float(d.isna().mean())
    else:
        out["min_date"] = None
        out["max_date"] = None
        out["null_date_rate"] = None

    if id_col and id_col in df.columns:
        dup_rate = float(df[id_col].astype(str).duplicated().mean()) if len(df) > 0 else 0.0
        out["dup_rate"] = dup_rate
    else:
        out["dup_rate"] = None

    return out


def _missing_cols(df: pd.DataFrame, required: Sequence[str]) -> List[str]:
    cols = set(df.columns) if df is not None else set()
    return [c for c in required if c not in cols]


def _warn_recommended(df: pd.DataFrame, recommended: Sequence[str], where: str) -> None:
    missing = [c for c in recommended if c not in df.columns]
    if missing:
        _warn(f"{where}: columnas recomendadas ausentes: {missing}")


def _macro_area_check(df: pd.DataFrame, where: str, strict: bool) -> Tuple[bool, int]:
    """
    Devuelve: (ok, warn_count)
    """
    warn_count = 0

    if df.empty:
        _warn(f"{where}: dataset vacío (no se evalúa macro_area).")
        return (not strict, 1)

    if "macro_area" not in df.columns:
        _warn(f"{where}: no existe 'macro_area'. (Recomendado para análisis por áreas).")
        return (not strict, 1)

    s = df["macro_area"].astype(str).fillna("")
    empty_rate = float((s.str.len() == 0).mean())
    if empty_rate > 0.02:
        warn_count += 1
        _warn(f"{where}: macro_area vacío en {empty_rate:.2%} de filas.")

    vc = s.value_counts().head(10)
    _p(f"      {where}: top macro_areas:")
    for k, v in vc.items():
        _p(f"        - {k}: {int(v)}")

    return (True, warn_count)


def _meta_check(meta_path: Path) -> Tuple[bool, Optional[dict], Optional[str]]:
    if not meta_path.exists():
        return False, None, f"Meta no existe: {meta_path}"
    try:
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return True, meta, None
    except Exception as e:
        return False, None, f"Meta corrupta/no legible: {e}"


# =============================================================================
# Result struct
# =============================================================================
@dataclass
class CheckResult:
    name: str
    ok: bool
    warns: int
    fails: int


# =============================================================================
# Checks
# =============================================================================
def check_hist(path: Path, *, strict: bool, freq: str) -> CheckResult:
    name = "Histórico"
    warns = 0
    fails = 0

    if not file_exists(path):
        _fail(f"{name}: no existe {path}")
        return CheckResult(name, ok=False, warns=0, fails=1)

    # lectura ligera
    cols = list(set(HIST_REQUIRED_COLS + HIST_RECOMMENDED_COLS))
    df = _safe_read_parquet(path, cols=cols)
    df = _coerce_date(df)

    if df.empty:
        msg = f"{name}: parquet vacío: {path}"
        if strict:
            _fail(msg)
            return CheckResult(name, ok=False, warns=0, fails=1)
        _warn(msg)
        return CheckResult(name, ok=True, warns=1, fails=0)

    missing = _missing_cols(df, HIST_REQUIRED_COLS)
    if missing:
        _fail(f"{name}: faltan columnas requeridas: {missing}")
        return CheckResult(name, ok=False, warns=0, fails=1)

    _warn_recommended(df, HIST_RECOMMENDED_COLS, where=name)

    stats = _df_stats(df, id_col="id" if "id" in df.columns else None)
    _ok(f"{name}: leído {path} rows={stats['rows']} cols={stats['cols']}")
    _p(f"      rango fechas: {stats['min_date']} -> {stats['max_date']}")
    _p(f"      null_date_rate: {stats['null_date_rate']}")
    if stats["dup_rate"] is not None:
        _p(f"      dup_rate(id): {stats['dup_rate']}")

    # periodos
    pc = _period_count(df, freq=freq)
    if pc < 2:
        msg = f"{name}: periodos insuficientes para tendencias (freq={freq}) → {pc}"
        if strict:
            _fail(msg)
            fails += 1
        else:
            _warn(msg)
            warns += 1
    else:
        _ok(f"{name}: periodos (freq={freq}) = {pc}")

    # macro_area si existe (hist puede tenerlo si preprocess/taxonomy se aplicó)
    ok_ma, w_ma = _macro_area_check(df, where=name, strict=False)  # hist macro_area es recomendado, no obligatorio
    warns += int(w_ma)
    _ = ok_ma

    return CheckResult(name, ok=(fails == 0), warns=warns, fails=fails)


def check_live(path: Path, *, strict: bool, freq: str, meta_path: Path) -> CheckResult:
    name = "Live"
    warns = 0
    fails = 0

    if not file_exists(path):
        _fail(f"{name}: no existe {path}")
        return CheckResult(name, ok=False, warns=0, fails=1)

    cols = list(set(LIVE_REQUIRED_COLS + LIVE_RECOMMENDED_COLS))
    df = _safe_read_parquet(path, cols=cols)
    df = _coerce_date(df)

    if df.empty:
        msg = f"{name}: parquet vacío: {path}"
        if strict:
            _fail(msg)
            return CheckResult(name, ok=False, warns=0, fails=1)
        _warn(msg)
        return CheckResult(name, ok=True, warns=1, fails=0)

    missing = _missing_cols(df, LIVE_REQUIRED_COLS)
    if missing:
        _fail(f"{name}: faltan columnas requeridas: {missing}")
        return CheckResult(name, ok=False, warns=0, fails=1)

    _warn_recommended(df, LIVE_RECOMMENDED_COLS, where=name)

    stats = _df_stats(df, id_col="id")
    _ok(f"{name}: leído {path} rows={stats['rows']} cols={stats['cols']}")
    _p(f"      rango fechas: {stats['min_date']} -> {stats['max_date']}")
    _p(f"      null_date_rate: {stats['null_date_rate']}")
    _p(f"      dup_rate(id): {stats['dup_rate']}")

    pc = _period_count(df, freq=freq)
    if pc < 2:
        msg = f"{name}: periodos insuficientes para tendencias (freq={freq}) → {pc}"
        if strict:
            _fail(msg)
            fails += 1
        else:
            _warn(msg)
            warns += 1
    else:
        _ok(f"{name}: periodos (freq={freq}) = {pc}")

    # macro_area obligatorio en live (porque el runner ya lo fuerza)
    ok_ma, w_ma = _macro_area_check(df, where=name, strict=strict)
    warns += int(w_ma)
    if strict and not ok_ma:
        fails += 1

    # meta (opcional, pero recomendado)
    meta_ok, meta, meta_err = _meta_check(meta_path)
    if meta_ok and meta:
        _ok(f"{name}: meta OK ({meta_path})")
        # mostrar llaves relevantes sin depender de schema exacto
        for k in ("updated_at_utc", "days_back", "source_used", "total_rows", "new_unique"):
            if k in meta:
                _p(f"      meta.{k}: {meta.get(k)}")
    else:
        msg = f"{name}: meta no disponible ({meta_err})"
        if strict:
            _warn(msg)  # meta no debe tumbar el sistema, pero sí avisar
            warns += 1
        else:
            _warn(msg)
            warns += 1

    return CheckResult(name, ok=(fails == 0), warns=warns, fails=fails)


# =============================================================================
# CLI
# =============================================================================
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Healthcheck del pipeline (histórico + live)")
    p.add_argument("--freq", default="M", choices=["D", "W", "M"], help="Frecuencia para validar periodos")
    p.add_argument("--strict", action="store_true", help="Convierte warnings clave en fallos")
    p.add_argument("--no-live", action="store_true", help="No validar LIVE")
    p.add_argument("--no-hist", action="store_true", help="No validar HISTÓRICO")
    p.add_argument("--hist", default=str(HIST_DATASET), help="Ruta parquet histórico (override)")
    p.add_argument("--live", default=str(LIVE_DATASET), help="Ruta parquet live (override)")
    p.add_argument("--live-meta", default=str(LIVE_META), help="Ruta meta live (override)")
    return p


def main(argv: List[str]) -> int:
    args = _build_argparser().parse_args(argv)

    strict = bool(args.strict)
    freq = str(args.freq)

    hist_path = Path(args.hist)
    live_path = Path(args.live)
    live_meta_path = Path(args.live_meta)

    _p("============================================================")
    _p("HEALTHCHECK · Pipeline de Tendencias (Histórico / Live)")
    _p("============================================================")
    _p(f"- freq:   {freq}")
    _p(f"- strict: {strict}")
    _p(f"- HIST:   {hist_path}")
    _p(f"- LIVE:   {live_path}")
    _p(f"- META:   {live_meta_path}")
    _p("------------------------------------------------------------")

    total_warns = 0
    total_fails = 0

    if not args.no_hist:
        r = check_hist(hist_path, strict=strict, freq=freq)
        total_warns += r.warns
        total_fails += r.fails
        _p("------------------------------------------------------------")

    if not args.no_live:
        r = check_live(live_path, strict=strict, freq=freq, meta_path=live_meta_path)
        total_warns += r.warns
        total_fails += r.fails
        _p("------------------------------------------------------------")

    # Resumen
    _p("RESUMEN")
    _p(f"- warnings: {total_warns}")
    _p(f"- fails:    {total_fails}")
    _p("============================================================")

    if total_fails > 0:
        return 3
    if total_warns > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
