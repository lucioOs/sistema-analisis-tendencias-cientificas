# src/sidebar.py
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.config import (
    APP_TITLE,
    CACHE_TTL_SEC,
    MAX_ROWS_TEXT,
    LIVE_DATASET,
    AREA_TS,
    AREA_TOP,
)
from src.data.io import file_exists, safe_last_update_label, run_script_capture
from src.data.datasets import load_historico_dataset


# =============================================================================
# Helpers
# =============================================================================
def _clear_streamlit_cache() -> None:
    try:
        st.cache_data.clear()
    except Exception:
        pass


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    try:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _get_macro_areas_from_data() -> List[str]:
    processed = Path("data/processed")

    df = _safe_read_parquet(processed / "macro_trends_full.parquet")
    if not df.empty and "macro_area" in df.columns:
        return sorted(a for a in df["macro_area"].dropna().astype(str).unique() if a.strip())

    df = _safe_read_parquet(processed / "clean.parquet")
    if not df.empty and "macro_area" in df.columns:
        return sorted(a for a in df["macro_area"].dropna().astype(str).unique() if a.strip())

    df = _safe_read_parquet(Path(LIVE_DATASET))
    if not df.empty and "macro_area" in df.columns:
        return sorted(a for a in df["macro_area"].dropna().astype(str).unique() if a.strip())

    return []


def _run_live_update(
    days_back: int,
    api_page_size: int,
    api_max_total: int,
    timeout_sec: int = 1800,
) -> Tuple[int, str, List[str]]:
    cmd = [
        sys.executable,
        "-m",
        "src.live_runner",
        "--days-back",
        str(int(days_back)),
        "--log-level",
        "INFO",
        "--api-page-size",
        str(int(api_page_size)),
        "--api-max-total",
        str(int(api_max_total)),
    ]
    code, out = run_script_capture(cmd, timeout_sec=timeout_sec)
    return int(code), (out or ""), cmd


def _safe_date_range_from_hist() -> Tuple[Optional[date], Optional[date]]:
    try:
        df_h, _, _ = load_historico_dataset(thin=True)
        if df_h is None or df_h.empty or "date" not in df_h.columns:
            return None, None
        d = pd.to_datetime(df_h["date"], errors="coerce").dropna()
        if d.empty:
            return None, None
        return d.min().date(), d.max().date()
    except Exception:
        return None, None


# =============================================================================
# Sidebar
# =============================================================================
def render_sidebar() -> Dict[str, Any]:
    # -----------------------------
    # Navegación
    # -----------------------------
    st.header("Navegación")
    cols = st.columns(2)
    if cols[0].button("Menú", key="sb_menu"):
        st.session_state.screen = "menu"
        st.rerun()
    if cols[1].button("Recargar", key="sb_reload_top"):
        _clear_streamlit_cache()
        st.rerun()

    st.divider()

    st.markdown("""
    <div class="card">
      <div style="font-weight:700; margin-bottom:0.2rem;">🧭 Cómo usar esta barra</div>
      <div class="muted">1) Define tipo de análisis. 2) Ajusta periodo. 3) Usa opciones avanzadas solo si lo necesitas.</div>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # Acción (token consistente)
    # -----------------------------
    st.subheader("1) Tipo de análisis")
    action_ui = st.radio(
        "Selecciona análisis",
        ["Predicción", "Creciendo", "Bajando", "Se mantiene", "Comparar"],
        index=0,
        key="sb_action_ui",
        horizontal=False,
    )
    action_map = {
        "Predicción": "prediccion",
        "Creciendo": "creciendo",
        "Bajando": "bajando",
        "Se mantiene": "neutro",
        "Comparar": "comparar",
    }
    action = action_map.get(action_ui, "prediccion")
    st.session_state.action = action

    st.divider()

    # -----------------------------
    # Periodo (frecuencia + rango opcional)
    # -----------------------------
    st.subheader("2) Periodo")

    freq_label = st.selectbox(
        "Agrupar resultados por",
        ["Semanas", "Meses", "Días"],
        index=0,
        key="sb_freq_label",
    )
    freq = "W" if freq_label == "Semanas" else ("M" if freq_label == "Meses" else "D")

    min_d, max_d = _safe_date_range_from_hist()
    use_range = st.toggle("Usar rango de fechas personalizado", value=False, key="sb_use_range")

    start_date: Optional[date] = None
    end_date: Optional[date] = None

    if use_range:
        if min_d and max_d:
            default_range = (min_d, max_d)
        else:
            today = pd.Timestamp.utcnow().date()
            default_range = (today.replace(year=today.year - 1), today)

        picked = st.date_input(
            "Rango (inicio, fin)",
            value=default_range,
            key="sb_date_range",
        )
        if isinstance(picked, tuple) and len(picked) == 2:
            start_date, end_date = picked[0], picked[1]
        else:
            start_date, end_date = None, None

    st.divider()

    # -----------------------------
    # Opciones (nube + knobs)
    # -----------------------------
    st.subheader("3) Opciones visuales")

    cloud_ui = st.selectbox(
        "Nube de palabras",
        ["Destacados (TF-IDF)", "Frecuencia"],
        index=0,
        key="sb_cloud_mode",
        help="Solo afecta la nube de palabras.",
    )
    cloud_mode = "tfidf" if cloud_ui.startswith("Destacados") else "freq"

    with st.expander("Ajustes avanzados", expanded=False):
        ngram_max = st.selectbox("Detectar frases de", [1, 2, 3], index=1, key="sb_ngram")
        min_df = st.slider("Frecuencia mínima (min_df)", 1, 10, 2, key="sb_min_df")

        # Live: ventana de visualización (solo aplica si NO hay rango manual)
        if st.session_state.get("screen") == "live":
            window_days = st.slider("Días a revisar (Live)", 7, 365, 180, key="sb_live_days")
        else:
            window_days = 180

        live_update_days = st.slider(
            "Actualizar Live con ventana (días)",
            7, 365, 180,
            key="sb_live_update_days",
        )

        api_page_size = st.selectbox("API page size", [50, 100, 200], index=2, key="sb_api_page_size")
        api_max_total = st.selectbox("API max total", [1000, 2000, 4000, 6000, 10000], index=4, key="sb_api_max_total")

    st.divider()

    # -----------------------------
    # Estado del sistema (info)
    # -----------------------------
    with st.expander("Estado del sistema", expanded=False):
        st.write(f"App: {APP_TITLE}")
        st.write(f"Pantalla: {st.session_state.get('screen', 'menu')}")
        st.write(f"Acción(token): {action}")
        st.write(f"cloud_mode(token): {cloud_mode}")
        st.write(f"Cache TTL (s): {CACHE_TTL_SEC}")
        st.write(f"MAX_ROWS_TEXT: {MAX_ROWS_TEXT}")
        st.write("Agregados macro-áreas (Histórico): " f"{'Sí' if (file_exists(AREA_TS) and file_exists(AREA_TOP)) else 'No'}")
        if use_range:
            st.write(f"Rango: {start_date} → {end_date}")

    st.divider()

    # -----------------------------
    # Información de datos
    # -----------------------------
    st.subheader("Datos")
    try:
        _, label_h, p_h = load_historico_dataset(thin=True)
        st.caption(label_h)
        st.caption(safe_last_update_label(p_h, "Histórico"))
    except Exception:
        st.caption("Histórico: no disponible (revisa data/processed).")

    try:
        st.caption(safe_last_update_label(LIVE_DATASET, "Live"))
    except Exception:
        st.caption("Live: no disponible.")

    processed = Path("data/processed")
    st.caption(safe_last_update_label(processed / "macro_trends_full.parquet", "Tendencias (macro)"))
    st.caption(safe_last_update_label(processed / "macro_trends_forecast.parquet", "Predicción (macro)"))

    st.divider()

    # -----------------------------
    # Acciones LIVE
    # -----------------------------
    st.subheader("Live")
    if "live_last_run" not in st.session_state:
        st.session_state.live_last_run = {"code": None, "cmd": None, "out": ""}

    if st.button("Actualizar Live", key="sb_live_update"):
        with st.spinner("Actualizando Live…"):
            code, out, cmd = _run_live_update(
                days_back=int(live_update_days),
                api_page_size=int(api_page_size),
                api_max_total=int(api_max_total),
                timeout_sec=1800,
            )

        st.session_state.live_last_run = {"code": code, "cmd": cmd, "out": out}

        if code == 0:
            st.success("Live actualizado correctamente.")
            _clear_streamlit_cache()
            time.sleep(0.2)
            st.rerun()
        else:
            st.error("Falló la actualización de Live. Revisa los logs abajo.")

    with st.expander("Logs de actualización Live", expanded=False):
        lr = st.session_state.get("live_last_run", {}) or {}
        code = lr.get("code")
        cmd = lr.get("cmd")
        out = lr.get("out", "")
        st.write(f"Return code: {code}")
        if cmd:
            st.caption("Comando:")
            st.code(" ".join(cmd))
        if out:
            st.caption("Salida / error:")
            st.text(out[:12000])
        else:
            st.caption("Sin logs aún.")

    if st.button("Recargar datos", key="sb_reload"):
        _clear_streamlit_cache()
        st.rerun()

    effective_window_days = None if use_range else int(window_days)

    return dict(
        # global
        macro_selected="Todas",
        action=str(action),

        # periodo
        freq=str(freq),
        start_date=start_date,
        end_date=end_date,

        # nube
        cloud_mode=str(cloud_mode),

        # knobs nube
        ngram_max=int(ngram_max),
        min_df=int(min_df),

        # live
        window_days=effective_window_days,   # None si hay rango manual
        live_update_days=int(live_update_days),
    )
