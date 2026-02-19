# src/config.py
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, List, Optional


# =============================================================================
# Configuración central del proyecto (fuente única de verdad)
# -----------------------------------------------------------------------------
# Objetivos:
# 1) Rutas canónicas (HIST y LIVE) consistentes para TODO el sistema.
# 2) Parametrizable por variables de entorno (para Expo / HuggingFace / VPS).
# 3) Idempotente: crea directorios base sin romper si ya existen.
# 4) Trazabilidad: snapshot() y validaciones suaves (sin tumbar Streamlit).
# =============================================================================


# =============================================================================
# Base paths
# =============================================================================
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"  # histórico + artefactos batch/offline
LIVE_DIR: Final[Path] = DATA_DIR / "live"            # live_runner + meta (trazabilidad)


# =============================================================================
# Helpers: env parsing seguro
# =============================================================================
def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, default)
    return str(v).strip()


def _env_int(name: str, default: int, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    raw = os.getenv(name, "")
    try:
        v = int(str(raw).strip()) if raw != "" else int(default)
    except Exception:
        v = int(default)

    if min_value is not None:
        v = max(min_value, v)
    if max_value is not None:
        v = min(max_value, v)
    return v


def _env_float(name: str, default: float, *, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    raw = os.getenv(name, "")
    try:
        v = float(str(raw).strip()) if raw != "" else float(default)
    except Exception:
        v = float(default)

    if min_value is not None:
        v = max(min_value, v)
    if max_value is not None:
        v = min(max_value, v)
    return v


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    raw = str(raw).strip().lower()
    return raw in {"1", "true", "t", "yes", "y", "on"}


def _env_path(name: str, default: Path, *, must_be_relative_to_project: bool = False) -> Path:
    """
    Lee una ruta desde env y la normaliza.
    - Si env no existe: usa default.
    - Si env es relativo: lo interpreta relativo a PROJECT_ROOT.
    - Si must_be_relative_to_project=True: fuerza que sea relativa al proyecto.
    """
    v = os.getenv(name)
    if not v:
        return default

    p = Path(str(v).strip()).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    else:
        p = p.resolve()

    if must_be_relative_to_project:
        try:
            # asegura que cae dentro del proyecto
            p.relative_to(PROJECT_ROOT)
        except Exception:
            return default

    return p


# =============================================================================
# Directorios base
# =============================================================================
def ensure_dirs() -> None:
    """Asegura estructura mínima del proyecto (idempotente)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)


ensure_dirs()


# =============================================================================
# App UI
# =============================================================================
APP_TITLE: Final[str] = _env_str("APP_TITLE", "Panel de Tendencias en Computación")


# =============================================================================
# Cache / performance
# =============================================================================
# TTL para cache_data en Streamlit (segundos)
CACHE_TTL_SEC: Final[int] = _env_int("CACHE_TTL_SEC", 300, min_value=30, max_value=3600)

# Tope de filas para proteger la UI (muestra/adelgaza)
MAX_ROWS_TEXT: Final[int] = _env_int("MAX_ROWS_TEXT", 7000, min_value=500, max_value=500_000)

# Semilla global (si quieres muestreos determinísticos en UI/QA)
RANDOM_STATE: Final[int] = _env_int("RANDOM_STATE", 7, min_value=0, max_value=10_000)


# =============================================================================
# Histórico
# =============================================================================
HIST_YEARS_KEEP: Final[int] = _env_int("HIST_YEARS_KEEP", 10, min_value=1, max_value=50)


# =============================================================================
# Datasets canónicos (fuente única)
# -----------------------------------------------------------------------------
# RECOMENDACIÓN (pipeline macro-area-first):
# - HIST_DATASET  -> data/processed/clean.parquet
# - LIVE_DATASET  -> data/live/live_dataset.parquet
# - LIVE_META     -> data/live/live_meta.json
#
# Si tu live_runner escribe en otro lado, ajusta SOLO aquí y el resto del
# sistema debe consumirlo vía config.
# =============================================================================
HIST_DATASET: Final[Path] = _env_path("HIST_DATASET", PROCESSED_DIR / "clean.parquet")
LIVE_DATASET: Final[Path] = _env_path("LIVE_DATASET", LIVE_DIR / "live_dataset.parquet")
LIVE_META: Final[Path] = _env_path("LIVE_META", LIVE_DIR / "live_meta.json")


# =============================================================================
# Runner LIVE (usado por sidebar.py)
# =============================================================================
LIVE_RUNNER_MODULE: Final[List[str]] = [sys.executable, "-m", "src.live_runner"]
LIVE_RUNNER_SCRIPT: Final[List[str]] = [sys.executable, str(PROJECT_ROOT / "src" / "live_runner.py")]

# Elección por default: módulo (evita líos de imports/path)
LIVE_RUNNER: Final[List[str]] = LIVE_RUNNER_MODULE


# =============================================================================
# Artefactos de tendencias (batch / histórico)
# =============================================================================
# Si tu pipeline genera:
# - trends_full.parquet
# - trend_classes.parquet
# - trends_forecast.parquet
# deja estos como canónicos. (Tus screens actuales calculan on-the-fly,
# pero estos sirven para “modo rápido” si decides habilitarlo después.)
TRENDS_FULL: Final[Path] = _env_path("TRENDS_FULL", PROCESSED_DIR / "trends_full.parquet")
TREND_CLASSES: Final[Path] = _env_path("TREND_CLASSES", PROCESSED_DIR / "trend_classes.parquet")
TRENDS_FORECAST: Final[Path] = _env_path("TRENDS_FORECAST", PROCESSED_DIR / "trends_forecast.parquet")


# =============================================================================
# Agregados macro-áreas (opcional / UI rápida)
# =============================================================================
AREA_TS: Final[Path] = _env_path("AREA_TS", PROCESSED_DIR / "area_ts.parquet")
AREA_TOP: Final[Path] = _env_path("AREA_TOP", PROCESSED_DIR / "area_top.parquet")


# =============================================================================
# Parámetros recomendados (centralizados) para análisis
# =============================================================================
# Defaults “expo-safe” (rápidos, robustos)
DEFAULT_FREQ: Final[str] = _env_str("DEFAULT_FREQ", "W")  # D | W | M
DEFAULT_NGRAM_MAX: Final[int] = _env_int("DEFAULT_NGRAM_MAX", 2, min_value=1, max_value=3)
DEFAULT_MIN_DF: Final[int] = _env_int("DEFAULT_MIN_DF", 2, min_value=1, max_value=200)

# Live: ventana de UI (no el runner)
DEFAULT_LIVE_WINDOW_DAYS: Final[int] = _env_int("DEFAULT_LIVE_WINDOW_DAYS", 180, min_value=7, max_value=3650)

# Live runner: polite sleep / timeouts (si quieres exponerlos)
LIVE_API_TIMEOUT_SEC: Final[int] = _env_int("LIVE_API_TIMEOUT_SEC", 30, min_value=5, max_value=300)
LIVE_API_POLITE_SLEEP_SEC: Final[float] = _env_float("LIVE_API_POLITE_SLEEP_SEC", 3.1, min_value=0.0, max_value=30.0)


# =============================================================================
# Sanity / compat: no rompe la app
# =============================================================================
def _warn(msg: str) -> None:
    # No uses logging aquí para no depender del setup.
    print(f"[config] {msg}", file=sys.stderr)


def validate_paths_soft() -> None:
    """
    Validaciones suaves:
    - no lanzan excepción
    - ayudan a detectar desalineaciones (rutas inconsistentes)
    """
    try:
        if not PROCESSED_DIR.exists():
            _warn(f"PROCESSED_DIR no existe (se intentó crear): {PROCESSED_DIR}")
        if not LIVE_DIR.exists():
            _warn(f"LIVE_DIR no existe (se intentó crear): {LIVE_DIR}")

        # Solo warnings (no obligamos a existir)
        if HIST_DATASET.suffix.lower() != ".parquet":
            _warn(f"HIST_DATASET no parece parquet: {HIST_DATASET}")

        if LIVE_DATASET.suffix.lower() != ".parquet":
            _warn(f"LIVE_DATASET no parece parquet: {LIVE_DATASET}")

        if LIVE_META.suffix.lower() != ".json":
            _warn(f"LIVE_META no parece json: {LIVE_META}")
    except Exception as e:
        _warn(f"validate_paths_soft falló: {e}")


validate_paths_soft()


# =============================================================================
# Snapshot (trazabilidad)
# =============================================================================
@dataclass(frozen=True)
class ConfigSnapshot:
    project_root: str
    data_dir: str
    processed_dir: str
    live_dir: str

    hist_dataset: str
    live_dataset: str
    live_meta: str

    cache_ttl_sec: int
    max_rows_text: int
    hist_years_keep: int
    random_state: int

    defaults: dict


def snapshot() -> ConfigSnapshot:
    return ConfigSnapshot(
        project_root=str(PROJECT_ROOT),
        data_dir=str(DATA_DIR),
        processed_dir=str(PROCESSED_DIR),
        live_dir=str(LIVE_DIR),
        hist_dataset=str(HIST_DATASET),
        live_dataset=str(LIVE_DATASET),
        live_meta=str(LIVE_META),
        cache_ttl_sec=int(CACHE_TTL_SEC),
        max_rows_text=int(MAX_ROWS_TEXT),
        hist_years_keep=int(HIST_YEARS_KEEP),
        random_state=int(RANDOM_STATE),
        defaults={
            "freq": DEFAULT_FREQ,
            "ngram_max": int(DEFAULT_NGRAM_MAX),
            "min_df": int(DEFAULT_MIN_DF),
            "live_window_days": int(DEFAULT_LIVE_WINDOW_DAYS),
            "live_api_timeout_sec": int(LIVE_API_TIMEOUT_SEC),
            "live_api_polite_sleep_sec": float(LIVE_API_POLITE_SLEEP_SEC),
        },
    )
