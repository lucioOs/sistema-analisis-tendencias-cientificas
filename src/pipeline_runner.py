# src/pipeline_runner.py
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


# =============================================================================
# Pipeline Runner (robusto)
# - Ejecuta SIEMPRE como módulos: python -m src.<modulo>
# - Evita errores de imports tipo: ModuleNotFoundError: No module named 'src'
# - Soporta --all, --strict, --force, --only
# =============================================================================


@dataclass(frozen=True)
class PipelineFiles:
    # Entradas / outputs principales (ajusta si cambias tu contrato)
    dataset_parquet: Path = Path("data/processed/dataset.parquet")
    clean_parquet: Path = Path("data/processed/clean.parquet")
    trends_full_parquet: Path = Path("data/processed/trends_full.parquet")
    trend_classes_parquet: Path = Path("data/processed/trend_classes.parquet")
    model_pkl: Path = Path("models/model.pkl")

    raw_mit_csv: Path = Path("data/raw/mit_ai_news.csv")


REQ_FILES = PipelineFiles()


# =============================================================================
# Helpers
# =============================================================================
def _exists(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False


def _missing_outputs() -> List[Path]:
    outs = [
        REQ_FILES.dataset_parquet,
        REQ_FILES.clean_parquet,
        REQ_FILES.trends_full_parquet,
        REQ_FILES.trend_classes_parquet,
        REQ_FILES.model_pkl,
    ]
    return [p for p in outs if not _exists(p)]


def _run(cmd: List[str]) -> int:
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd)
    return int(p.returncode)


def _run_or_die(cmd: List[str]) -> None:
    rc = _run(cmd)
    if rc != 0:
        raise SystemExit(rc)


def _ensure_dirs() -> None:
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/live").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)


# =============================================================================
# Steps (módulos)
# =============================================================================
def step_load_data() -> None:
    # Carga CSV -> dataset.parquet (tu load_data escribe data/processed/dataset.parquet por default)
    if not _exists(REQ_FILES.raw_mit_csv):
        raise SystemExit(f"[ERROR] No existe input: {REQ_FILES.raw_mit_csv}")
    _run_or_die([sys.executable, "-m", "src.load_data", "--file", REQ_FILES.raw_mit_csv.name, "--drop_duplicates"])


def step_preprocess(min_len: int) -> None:
    # preprocess: dataset.parquet -> clean.parquet (según tu preprocess actual)
    _run_or_die([sys.executable, "-m", "src.preprocess", "--min_len", str(int(min_len))])


def step_trends_full(freq: str, min_df: int, ngram_max: int) -> None:
    _run_or_die(
        [
            sys.executable,
            "-m",
            "src.trends_full",
            "--freq",
            str(freq),
            "--min_df",
            str(int(min_df)),
            "--ngram_max",
            str(int(ngram_max)),
        ]
    )


def step_classify(freq_window: int, min_periods: int) -> None:
    _run_or_die(
        [
            sys.executable,
            "-m",
            "src.classify_trends",
            "--freq_window",
            str(int(freq_window)),
            "--min_periods",
            str(int(min_periods)),
        ]
    )


def step_features(horizon: int) -> None:
    _run_or_die([sys.executable, "-m", "src.features", "--horizon", str(int(horizon))])


def step_train(n_estimators: int) -> None:
    _run_or_die([sys.executable, "-m", "src.train", "--n_estimators", str(int(n_estimators))])


# =============================================================================
# CLI
# =============================================================================
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pipeline runner (robusto)")

    p.add_argument("--all", action="store_true", help="Ejecuta pipeline completo si faltan outputs")
    p.add_argument("--strict", action="store_true", help="Falla si falta input o si algún paso no puede correr")
    p.add_argument("--force", action="store_true", help="Ejecuta todo aunque ya existan outputs")

    p.add_argument(
        "--only",
        choices=["load", "preprocess", "trends", "classify", "features", "train"],
        default="",
        help="Ejecuta solo un paso",
    )

    # parámetros de tus scripts
    p.add_argument("--min_len", type=int, default=30)
    p.add_argument("--freq", default="M")
    p.add_argument("--min_df", type=int, default=3)
    p.add_argument("--ngram_max", type=int, default=2)
    p.add_argument("--freq_window", type=int, default=6)
    p.add_argument("--min_periods", type=int, default=8)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--n_estimators", type=int, default=500)

    return p


def main(argv: List[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv or [])
    _ensure_dirs()

    # Modo: solo un paso
    if args.only:
        if args.only == "load":
            step_load_data()
        elif args.only == "preprocess":
            step_preprocess(args.min_len)
        elif args.only == "trends":
            step_trends_full(args.freq, args.min_df, args.ngram_max)
        elif args.only == "classify":
            step_classify(args.freq_window, args.min_periods)
        elif args.only == "features":
            step_features(args.horizon)
        elif args.only == "train":
            step_train(args.n_estimators)
        print("[INFO] ✅ Paso ejecutado.", flush=True)
        return

    # Si no se pide --all, se comporta como antes: corre si faltan outputs
    missing = _missing_outputs()
    if not args.force and not missing:
        print("[INFO] Outputs ya existen. No se ejecuta pipeline.", flush=True)
        return

    if missing:
        print("[INFO] Faltan outputs:", flush=True)
        for m in missing:
            print(f"  - {m}", flush=True)

    print("[INFO] Ejecutando pipeline completo...", flush=True)

    # Validación mínima en strict
    if args.strict and not _exists(REQ_FILES.raw_mit_csv) and not _exists(REQ_FILES.dataset_parquet):
        raise SystemExit(f"[ERROR] strict: falta input {REQ_FILES.raw_mit_csv} y no existe dataset.parquet")

    # 1) Loader
    # Si ya existe dataset.parquet y no es force, puedes saltarlo
    if args.force or not _exists(REQ_FILES.dataset_parquet):
        step_load_data()
    else:
        print(f"[OK] dataset ya existe: {REQ_FILES.dataset_parquet}", flush=True)

    # 2) Preprocess -> clean.parquet
    if args.force or not _exists(REQ_FILES.clean_parquet):
        step_preprocess(args.min_len)
    else:
        print(f"[OK] clean ya existe: {REQ_FILES.clean_parquet}", flush=True)

    # 3) Tendencias full
    if args.force or not _exists(REQ_FILES.trends_full_parquet):
        step_trends_full(args.freq, args.min_df, args.ngram_max)
    else:
        print(f"[OK] trends_full ya existe: {REQ_FILES.trends_full_parquet}", flush=True)

    # 4) Clasificación
    if args.force or not _exists(REQ_FILES.trend_classes_parquet):
        step_classify(args.freq_window, args.min_periods)
    else:
        print(f"[OK] trend_classes ya existe: {REQ_FILES.trend_classes_parquet}", flush=True)

    # 5) Features + Train
    # Nota: features.py puede generar archivos intermedios; aquí solo garantizamos el modelo final
    if args.force or not _exists(REQ_FILES.model_pkl):
        step_features(args.horizon)
        step_train(args.n_estimators)
    else:
        print(f"[OK] model ya existe: {REQ_FILES.model_pkl}", flush=True)

    print("[INFO] ✅ Pipeline completo.", flush=True)


if __name__ == "__main__":
    main()
