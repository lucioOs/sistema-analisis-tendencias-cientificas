# src/state.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import streamlit as st


# =============================================================================
# Defaults (centralizados)
# =============================================================================
@dataclass(frozen=True)
class StateDefaults:
    # navegación
    screen: str = "menu"          # menu | historico | live
    action: str = "prediccion"    # creciendo | bajando | prediccion | comparar

    # filtro global macro-área (PLN)
    macro_area_selected: str = "Todas"

    # histórico: filtros de tiempo
    hist_preset: str = "Todo"
    hist_start: Optional[Any] = None
    hist_end: Optional[Any] = None
    hist_filter_sig: str = ""

    # live: parámetros UI
    live_window_preset: str = "Usar slider del sidebar"


DEFAULTS = StateDefaults()


# =============================================================================
# Helpers
# =============================================================================
def _set_if_missing(key: str, value: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def _coerce_screen(v: Any) -> str:
    v = str(v or "").strip().lower()
    allowed = {"menu", "historico", "live"}
    return v if v in allowed else DEFAULTS.screen


def _coerce_action(v: Any) -> str:
    v = str(v or "").strip().lower()
    allowed = {"creciendo", "bajando", "prediccion", "comparar"}
    return v if v in allowed else DEFAULTS.action


def _coerce_macro(v: Any) -> str:
    # No validamos contra lista dinámica aquí para evitar IO / dependencias.
    # Se valida en sidebar al renderizar.
    v = str(v or "").strip()
    return v if v else DEFAULTS.macro_area_selected


# =============================================================================
# Public API
# =============================================================================
def init_state() -> None:
    """
    Inicializa st.session_state de forma robusta.
    - Define defaults coherentes para todo el sistema
    - Normaliza valores por si quedaron corruptos
    - Mantiene compatibilidad con keys antiguas
    """
    # Defaults base
    _set_if_missing("screen", DEFAULTS.screen)
    _set_if_missing("action", DEFAULTS.action)

    # Filtro global (macro-área)
    _set_if_missing("macro_area_selected", DEFAULTS.macro_area_selected)

    # Histórico: filtros
    _set_if_missing("hist_preset", DEFAULTS.hist_preset)
    _set_if_missing("hist_start", DEFAULTS.hist_start)
    _set_if_missing("hist_end", DEFAULTS.hist_end)
    _set_if_missing("hist_filter_sig", DEFAULTS.hist_filter_sig)

    # Live: compatibilidad
    _set_if_missing("live_window_preset", DEFAULTS.live_window_preset)

    # Normalizaciones (si alguien asignó algo raro)
    st.session_state.screen = _coerce_screen(st.session_state.get("screen"))
    st.session_state.action = _coerce_action(st.session_state.get("action"))
    st.session_state.macro_area_selected = _coerce_macro(st.session_state.get("macro_area_selected"))

    # Compatibilidad con versiones anteriores (si existían)
    # Ej: algunos builds guardaban "hist_area_kw" o "live_area_kw"
    # Ahora el filtro global oficial es "macro_area_selected".
    if "hist_area_kw" in st.session_state and st.session_state.macro_area_selected == "Todas":
        v = st.session_state.get("hist_area_kw")
        if isinstance(v, str) and v.strip():
            st.session_state.macro_area_selected = v.strip()

    if "live_area_kw" in st.session_state and st.session_state.macro_area_selected == "Todas":
        v = st.session_state.get("live_area_kw")
        if isinstance(v, str) and v.strip():
            st.session_state.macro_area_selected = v.strip()


def reset_navigation(to_screen: str = "menu", to_action: str = "prediccion") -> None:
    """
    Reset rápido (útil si quieres usarlo desde UI/botones).
    """
    st.session_state.screen = _coerce_screen(to_screen)
    st.session_state.action = _coerce_action(to_action)


def snapshot_state() -> Dict[str, Any]:
    """
    Snapshot pequeño para debugging (opcional).
    """
    keys = [
        "screen",
        "action",
        "macro_area_selected",
        "hist_preset",
        "hist_start",
        "hist_end",
        "hist_filter_sig",
        "live_window_preset",
    ]
    return {k: st.session_state.get(k) for k in keys}
