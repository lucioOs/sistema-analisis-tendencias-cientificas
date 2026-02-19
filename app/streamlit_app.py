# app/streamlit_app.py
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

# -----------------------------------------------------------------------------
# BOOTSTRAP: permitir imports "src.*" aunque streamlit se ejecute desde /app
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

# -----------------------------------------------------------------------------
# Imports del proyecto
# -----------------------------------------------------------------------------
from src.config import APP_TITLE
from src.state import init_state
from src.ui.styles import apply_custom_css
from src.ui.widgets import top_guide

from src.screens.menu import screen_menu
from src.screens.historico import screen_historico
from src.screens.live import screen_live


# -----------------------------------------------------------------------------
# Configuración y utilidades
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AppInfo:
    title: str = APP_TITLE
    root: Path = PROJECT_ROOT
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"


def _safe_set_page_config(title: str) -> None:
    """
    Streamlit: set_page_config debe ejecutarse 1 vez y lo más arriba posible.
    Este wrapper evita excepciones si alguien lo llama tarde por error.
    """
    try:
        st.set_page_config(page_title=title, layout="wide")
    except Exception:
        # No reventar la app por esto
        pass


def _render_sidebar() -> dict:
    """
    Sidebar global (import diferido para evitar ciclos).
    Debe devolver un dict de configuración.
    """
    with st.sidebar:
        from src.sidebar import render_sidebar  # import diferido

        return render_sidebar()


def _render_global_header(title: str) -> None:
    st.title(title)
    top_guide()


def _route_screen(screen: str, sidebar_cfg: dict) -> None:
    """
    Router robusto: evita caerse por un estado inválido.
    Pasa parámetros del sidebar a cada screen sin casteos incorrectos.
    """
    # Normalización defensiva
    screen = str(screen or "menu")

    if screen == "menu":
        screen_menu()
        return

    macro_selected = str(sidebar_cfg.get("macro_selected", "Todas"))
    action = str(sidebar_cfg.get("action", "prediccion"))

    start_date = sidebar_cfg.get("start_date", None)  # date | None
    end_date = sidebar_cfg.get("end_date", None)      # date | None

    freq = str(sidebar_cfg.get("freq", "M"))
    cloud_mode = str(sidebar_cfg.get("cloud_mode", "tfidf"))  # "tfidf" | "freq"

    ngram_max = int(sidebar_cfg.get("ngram_max", 2))
    min_df = int(sidebar_cfg.get("min_df", 2))

    if screen == "historico":
        screen_historico(
            macro_selected=macro_selected,
            action=action,
            start_date=start_date,
            end_date=end_date,
            freq=freq,
            cloud_mode=cloud_mode,
            ngram_max=ngram_max,
            min_df=min_df,
        )
        return

    if screen == "live":
        # window_days puede ser None si el usuario seleccionó rango manual
        window_days = sidebar_cfg.get("window_days", None)
        if window_days is not None:
            try:
                window_days = int(window_days)
            except Exception:
                window_days = None

        screen_live(
            macro_selected=macro_selected,
            action=action,
            start_date=start_date,
            end_date=end_date,
            freq=freq,
            window_days=window_days,
            cloud_mode=cloud_mode,
            ngram_max=ngram_max,
            min_df=min_df,
            live_update_days=int(sidebar_cfg.get("live_update_days", 180)),
        )
        return

    # Fallback: estado inválido -> menú
    st.session_state.screen = "menu"
    screen_menu()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    app = AppInfo()

    _safe_set_page_config(app.title)

    # Estilos y estado
    apply_custom_css()
    init_state()

    # Header
    _render_global_header(app.title)

    # Sidebar config
    sidebar_cfg = _render_sidebar()

    # Router
    screen = getattr(st.session_state, "screen", "menu")
    _route_screen(screen, sidebar_cfg)


if __name__ == "__main__":
    main()
