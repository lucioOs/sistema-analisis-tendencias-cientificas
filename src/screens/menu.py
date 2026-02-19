# src/screens/menu.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from src.config import HIST_DATASET, LIVE_DATASET, LIVE_META


# =============================================================================
# Paths / artefactos esperados (canónicos vía config)
# =============================================================================
P_HIST_CLEAN = Path(HIST_DATASET)
P_LIVE = Path(LIVE_DATASET)
P_LIVE_META = Path(LIVE_META)

PROCESSED_DIR = Path("data/processed")
P_MACRO_TRENDS = PROCESSED_DIR / "macro_trends_full.parquet"
P_MACRO_CLASSES = PROCESSED_DIR / "macro_trend_classes.parquet"
P_MACRO_FC = PROCESSED_DIR / "macro_trends_forecast.parquet"


# =============================================================================
# Helpers
# =============================================================================
@dataclass(frozen=True)
class FileStatus:
    path: Path
    exists: bool
    rows: Optional[int]
    cols: Optional[int]
    mtime_label: str


def _mtime_label(path: Path) -> str:
    if not path.exists():
        return "—"
    try:
        ts = path.stat().st_mtime
        return pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "—"


def _quick_parquet_shape(path: Path) -> Tuple[Optional[int], Optional[int]]:
    if not path.exists():
        return None, None
    try:
        df = pd.read_parquet(path)
        return int(df.shape[0]), int(df.shape[1])
    except Exception:
        return None, None


def _status(path: Path) -> FileStatus:
    ex = path.exists()
    r, c = _quick_parquet_shape(path) if ex else (None, None)
    return FileStatus(
        path=path,
        exists=ex,
        rows=r,
        cols=c,
        mtime_label=_mtime_label(path),
    )


def _badge(ok: bool) -> str:
    return "✅" if ok else "❌"


def _goto(screen: str, default_action: str = "prediccion") -> None:
    st.session_state.screen = screen
    st.session_state.action = default_action
    st.rerun()


# =============================================================================
# Screen
# =============================================================================
def screen_menu() -> None:
    st.subheader("Menú principal")

    st.write("Selecciona un modo de exploración y revisa el estado del pipeline.")

    # -------------------------------------------------------------------------
    # Estado del pipeline
    # -------------------------------------------------------------------------
    st.markdown("### Estado del sistema (artefactos)")

    s_hist = _status(P_HIST_CLEAN)
    s_live = _status(P_LIVE)
    s_live_meta = _status(P_LIVE_META)
    s_tr = _status(P_MACRO_TRENDS)
    s_cl = _status(P_MACRO_CLASSES)
    s_fc = _status(P_MACRO_FC)

    # Tabla bonita de estado
    rows = []
    for name, s in [
        ("Histórico limpio", s_hist),
        ("Live", s_live),
        ("Live meta", s_live_meta),
        ("Tendencias (macro)", s_tr),
        ("Clases (macro)", s_cl),
        ("Predicción (macro)", s_fc),
    ]:
        rows.append(
            {
                "Componente": name,
                "Estado": f"{_badge(s.exists)} {'OK' if s.exists else 'Falta'}",
                "Archivo": str(s.path),
                "Filas": s.rows if s.rows is not None else "—",
                "Columnas": s.cols if s.cols is not None else "—",
                "Actualizado": s.mtime_label,
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Advertencia si faltan piezas críticas
    missing_critical = []
    if not s_hist.exists:
        missing_critical.append(f"{P_HIST_CLEAN.name} (histórico)")
    if not s_tr.exists:
        missing_critical.append(f"{P_MACRO_TRENDS.name} (tendencias)")
    if not s_fc.exists:
        missing_critical.append(f"{P_MACRO_FC.name} (predicción)")

    if missing_critical:
        st.warning("Pipeline incompleto. Faltan:\n- " + "\n- ".join(missing_critical))
        with st.expander("Cómo regenerar el pipeline (comandos)", expanded=False):
            st.code(
                "\n".join(
                    [
                        "# 0) (Opcional) Regenerar macro_area en histórico limpio",
                        f"python -m src.load_data --rebuild-macro --rebuild-in {P_HIST_CLEAN.as_posix()}",
                        "",
                        "# 1) Live (API) -> escribe a LIVE_DATASET (canónico)",
                        "python -m src.live_runner --days-back 180 --api-max-total 10000 --max-keep 20000",
                        "",
                        "# 2) Forecast macro (histórico) -> genera macro_trends_* en data/processed",
                        "python -m src.forecast_trends --freq W --forecast_h 8",
                    ]
                ),
                language="bash",
            )

    st.divider()

    # -------------------------------------------------------------------------
    # Navegación principal
    # -------------------------------------------------------------------------
    st.markdown("### ¿Qué quieres explorar?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="bigbtn">', unsafe_allow_html=True)
        hist_disabled = not s_hist.exists
        if st.button("Histórico", key="menu_hist", disabled=hist_disabled):
            _goto("historico", default_action="prediccion")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="muted">Tendencias de largo plazo y predicción por macro-áreas.</div>',
            unsafe_allow_html=True,
        )
        if hist_disabled:
            st.caption(f"Requiere: {P_HIST_CLEAN}")

    with col2:
        st.markdown('<div class="bigbtn">', unsafe_allow_html=True)
        live_disabled = not s_live.exists
        if st.button("Live", key="menu_live", disabled=live_disabled):
            _goto("live", default_action="prediccion")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="muted">Lo más reciente (actualizable) + predicción por macro-áreas.</div>',
            unsafe_allow_html=True,
        )
        if live_disabled:
            st.caption(f"Requiere: {P_LIVE}")

    st.divider()

    # -------------------------------------------------------------------------
    # Guía rápida (para expo / defensa)
    # -------------------------------------------------------------------------
    with st.expander("Guía rápida de uso (expo)", expanded=False):
        st.write(
            "- El filtro global **Macro-área (filtro PLN)** controla Histórico, Live, Tendencias y Predicción.\n"
            "- **Histórico**: muestra evolución y clasificación de macro-áreas a lo largo del tiempo.\n"
            "- **Live**: muestra artículos recientes, y permite ver tendencia/predicción del área seleccionada.\n"
            "- La predicción es un **indicador de dirección** (sube/baja) basado en series agregadas; no es garantía."
        )

    # -------------------------------------------------------------------------
    # Diagnóstico mínimo
    # -------------------------------------------------------------------------
    with st.expander("Diagnóstico técnico", expanded=False):
        st.write("Variables de sesión relevantes:")
        st.json(
            {
                "screen": st.session_state.get("screen", "menu"),
                "action": st.session_state.get("action", "prediccion"),
                "macro_area_selected": st.session_state.get("macro_area_selected", "Todas"),
                "HIST_DATASET": str(P_HIST_CLEAN),
                "LIVE_DATASET": str(P_LIVE),
                "LIVE_META": str(P_LIVE_META),
            }
        )
