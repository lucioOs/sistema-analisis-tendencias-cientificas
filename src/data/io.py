from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


def file_exists(path: str | Path) -> bool:
    try:
        return Path(path).exists()
    except Exception:
        return False


def safe_last_update_label(path: str | Path, label: str = "Actualizado") -> str:
    p = Path(path)
    if not p.exists():
        return f"{label}: no disponible"
    try:
        ts = pd.to_datetime(p.stat().st_mtime, unit="s")
        return f"{label}: {ts.strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception:
        return f"{label}: disponible"


def run_script_capture(cmd: Iterable[str], timeout_sec: int = 1800) -> Tuple[int, str]:
    try:
        completed = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(timeout_sec),
            check=False,
        )
        return int(completed.returncode), str(completed.stdout or "")
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        return 124, f"[timeout] {timeout_sec}s\n{out}"
    except Exception as e:  # error operativo de ejecución
        return 1, f"[error] {type(e).__name__}: {e}"
