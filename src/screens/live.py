# src/screens/live.py
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import MaxNLocator

from src.analytics.macro_aggregate import aggregate_macro
from src.analytics.macro_classify import classify_macro
from src.config import MAX_ROWS_TEXT
from src.data.datasets import load_live_dataset
from src.data.io import safe_last_update_label
from src.load_data import enforce_schema
from src.metrics import limit_df
from src.plotting.charts import tick_step
from src.ui.widgets import download_table, render_recent, show_kpis
from src.ui.wordcloud_ui import render_wordcloud

PROCESSED_DIR = Path("data/processed")
P_MACRO_TRENDS = PROCESSED_DIR / "macro_trends_full.parquet"          # (solo referencia)
P_MACRO_FORECAST = PROCESSED_DIR / "macro_trends_forecast.parquet"    # (solo referencia)


# =============================================================================
# Cache parquet helpers (por si quieres mostrar batch en otra sección)
# =============================================================================
@st.cache_data(show_spinner=False, ttl=300)
def _load_parquet_cached(path: str, mtime: float) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _read_cached(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return _load_parquet_cached(str(path), path.stat().st_mtime)


# =============================================================================
# Filters
# =============================================================================
def _filter_macro(df: pd.DataFrame, macro_area: str) -> pd.DataFrame:
    if df is None or df.empty or not macro_area or macro_area == "Todas":
        return df
    if "macro_area" not in df.columns:
        return df
    return df[df["macro_area"].astype(str) == str(macro_area)].copy()


def _apply_date_range(
    df: pd.DataFrame,
    start_date: Optional[date],
    end_date: Optional[date],
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns:
        return df

    d = pd.to_datetime(df[date_col], errors="coerce")
    df = df.loc[d.notna()].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if start_date and end_date:
        lo = pd.Timestamp(start_date)
        hi = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    return df


def _apply_live_window(df: pd.DataFrame, days: Optional[int]) -> pd.DataFrame:
    if df is None or df.empty or not days or int(days) <= 0 or "date" not in df.columns:
        return df

    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")
    dfx = dfx.dropna(subset=["date"])
    if dfx.empty:
        return dfx

    mx = dfx["date"].max()
    if pd.isna(mx):
        return dfx

    cut = mx - pd.Timedelta(days=int(days))
    return dfx[dfx["date"] >= cut].copy()


# =============================================================================
# Tables / TS
# =============================================================================
def _build_ts(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=["date", "n"])

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    if d.empty:
        return pd.DataFrame(columns=["date", "n"])

    ts = (
        d.set_index("date")
        .resample(freq)
        .size()
        .rename("n")
        .reset_index()
        .sort_values("date")
    )
    return ts


def _build_article_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    if "title" not in out.columns:
        out["title"] = ""

    if "abstract" in out.columns:
        abs_col = "abstract"
    elif "summary" in out.columns:
        abs_col = "summary"
    elif "text" in out.columns:
        abs_col = "text"
    else:
        abs_col = None

    out["abstract"] = out[abs_col].astype(str) if abs_col else ""

    if "link" not in out.columns:
        out["link"] = ""
    if "id" not in out.columns:
        out["id"] = out["link"].astype(str)

    if "categories" not in out.columns:
        out["categories"] = ""
    if "primary_category" not in out.columns:
        out["primary_category"] = ""
    if "macro_area" not in out.columns:
        out["macro_area"] = "Sin clasificar"

    out["abstract_short"] = (
        out["abstract"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.slice(0, 350)
    )

    cols = ["date", "macro_area", "primary_category", "title", "abstract_short", "categories", "link", "id"]
    cols = [c for c in cols if c in out.columns]
    return out[cols].sort_values("date", ascending=False)


# =============================================================================
# Plotting
# =============================================================================
def _plot_series(periods: list[str], y: np.ndarray, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    x = np.arange(len(periods))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel("Periodo")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    n = len(periods)
    step = tick_step(n)
    tick_pos = x[::step]
    tick_lab = [periods[i] for i in range(0, n, step)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=30, ha="right")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _infer_seasonal_periods(freq: str) -> int:
    # Reglas defensibles:
    # - M: anual
    # - W: anual aproximado
    # - D: semanal
    if freq == "M":
        return 12
    if freq == "W":
        return 52
    if freq == "D":
        return 7
    return 12


def _future_period_labels(last_period: str, freq: str, h: int) -> list[str]:
    if h <= 0:
        return []
    try:
        p_last = pd.Period(last_period, freq=freq)
        fut = [p_last + i for i in range(1, h + 1)]
        return [str(p) for p in fut]
    except Exception:
        return [f"future+{i}" for i in range(1, h + 1)]


def _forecast_linear_seasonal_naive(y: np.ndarray, h: int, sp: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=0.0)
    n = len(y)
    if n <= 0:
        return np.zeros(h, dtype=float)

    x = np.arange(n, dtype=float)
    if n >= 2:
        a, b = np.polyfit(x, y, 1)
    else:
        a, b = 0.0, float(y[-1])

    base = a * np.arange(n, n + h, dtype=float) + b

    sp = int(sp or 0)
    if sp >= 2 and n >= sp:
        season = y[-sp:]
        season_rep = np.resize(season, h)
        pred = 0.75 * base + 0.25 * season_rep
        return np.clip(pred, 0.0, None)

    return np.clip(base, 0.0, None)


def _period_count_from_macro(df_macro_live: pd.DataFrame) -> int:
    if df_macro_live is None or df_macro_live.empty or "period" not in df_macro_live.columns:
        return 0
    return int(df_macro_live["period"].astype(str).nunique())


# =============================================================================
# Screen
# =============================================================================
def screen_live(
    *,
    macro_selected: str,
    action: str,
    start_date: Optional[date],
    end_date: Optional[date],
    freq: str,
    window_days: Optional[int],
    cloud_mode: str,
    ngram_max: int,
    min_df: int,
    live_update_days: Optional[int] = 14,
) -> None:
    # =========================================================
    # 1) Carga LIVE (sin thinning para tabla real)
    # =========================================================
    df_raw_full, label, p = load_live_dataset(thin=False)
    st.caption(f"{label} · {safe_last_update_label(p, 'Última actualización')}")

    if df_raw_full is None or df_raw_full.empty:
        st.warning("No hay datos LIVE listos para mostrar.")
        return

    df_raw_full = enforce_schema(df_raw_full, min_text_len=20, drop_duplicates=True)

    # =========================================================
    # 2) Macro-área se controla desde el sidebar
    # =========================================================
    st.subheader("Live (dinámico)")
    # 2) Selector principal de macro-área (afecta nube y gráficas)
    # =========================================================
    macro_options = ["Todas"]
    if "macro_area" in df_raw_full.columns:
        vals = sorted(a for a in df_raw_full["macro_area"].dropna().astype(str).unique() if a.strip())
        macro_options.extend(vals)

    default_macro = st.session_state.get(
        "live_main_macro",
        macro_selected if macro_selected in macro_options else "Todas",
    )
    if default_macro not in macro_options:
        default_macro = "Todas"

    st.subheader("Live (dinámico)")
    st.markdown("**Macro-área (afecta nube y gráficas):**")
    macro_selected = st.selectbox(
        "Macro-área para filtrar visualización",
        options=macro_options,
        index=macro_options.index(default_macro),
        key="live_main_macro",
    )

    df_view_full = _filter_macro(df_raw_full, macro_selected)

    # =========================================================
    # 3) Rango manual (prioridad) o ventana dinámica
    # =========================================================
    effective_window: Optional[int] = None
    if start_date and end_date:
        df_view_full = _apply_date_range(df_view_full, start_date, end_date, date_col="date")
    else:
        if window_days is not None:
            try:
                effective_window = int(window_days)
            except Exception:
                effective_window = None
        if effective_window is None:
            effective_window = int(live_update_days or 14)

        df_view_full = _apply_live_window(df_view_full, days=effective_window)

    with st.expander("Filtros activos", expanded=False):
        st.write(f"Macro-área: **{macro_selected}**")
        st.write(f"Acción: **{action}**")
        st.write(f"Frecuencia: **{freq}**")
        st.write(f"Nube: **{cloud_mode}**")
        if start_date and end_date:
            st.write(f"Rango: **{start_date} → {end_date}**")
        else:
            st.write(f"Ventana (días): **{effective_window if effective_window is not None else 'N/A'}**")

    if df_view_full is None or df_view_full.empty:
        st.warning("No hay registros LIVE con los filtros actuales (macro-área / rango / ventana).")
        render_recent(df_raw_full.head(2000).copy() if not df_raw_full.empty else df_raw_full)
        return

    # =========================================================
    # 4) UI-lite (para KPIs/nube/serie), FULL para artículos y macro-agg
    # =========================================================
    df_view_ui = limit_df(df_view_full, MAX_ROWS_TEXT)

    show_kpis(df_view_ui, freq)

    render_wordcloud(
        df_view_ui,
        "Nube de palabras (Live)",
        mode=cloud_mode,
        ngram_max=ngram_max,
        min_df=min_df,
    )

    df_ts = _build_ts(df_view_ui, freq=freq)
    st.subheader("Evolución por periodo (volumen)")
    if df_ts.empty or df_ts["n"].sum() == 0:
        st.info("No hay suficientes datos para construir la serie temporal con el filtro actual.")
    else:
        st.line_chart(df_ts.set_index("date")["n"])
        st.dataframe(df_ts, use_container_width=True, hide_index=True)

    # =========================================================
    # 5) Artículos (últimos reales)
    # =========================================================
    with st.expander("Artículos LIVE (últimos)", expanded=False):
        c1, c2 = st.columns([1, 2])
        with c1:
            max_articles = st.slider("Máximo a mostrar", 20, 300, 80, key="live_max_articles")
        with c2:
            q = st.text_input("Buscar en título/abstract (opcional)", value="", key="live_search").strip()

        tbl_all = _build_article_table(df_view_full)
        if q:
            qq = q.lower()
            mask = (
                tbl_all["title"].astype(str).str.lower().str.contains(qq, na=False)
                | tbl_all["abstract_short"].astype(str).str.lower().str.contains(qq, na=False)
            )
            tbl = tbl_all.loc[mask].copy()
        else:
            tbl = tbl_all

        tbl = tbl.head(int(max_articles))
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        download_table(tbl, filename_prefix="live_articles")

    # =========================================================
    # 6) Live macro-agg REAL (ventana actual) + clasificación REAL
    # =========================================================
    agg = aggregate_macro(df_view_full, freq=freq)
    df_macro_live = agg.df_macro.copy()
    wide_rel = agg.wide_rel_freq.copy()
    wide_cnt = agg.wide_count.copy()

    if df_macro_live.empty or wide_rel.empty:
        st.info("No hay suficientes datos para construir tendencias por macro-área en la ventana actual.")
        return

    # total_count real por macro (en la ventana actual)
    total_count = wide_cnt.sum(axis=0).rename("total_count") if not wide_cnt.empty else pd.Series(dtype=float)

    df_classes_live = classify_macro(wide_rel)
    if not df_classes_live.empty:
        if "macro_area" not in df_classes_live.columns:
            df_classes_live["macro_area"] = df_classes_live.index.astype(str)
        df_classes_live["macro_area"] = df_classes_live["macro_area"].astype(str)

        tc = total_count.reset_index()
        tc.columns = ["macro_area", "total_count"]
        df_classes_live = df_classes_live.drop(columns=["total_count"], errors="ignore").merge(tc, on="macro_area", how="left")
        df_classes_live["total_count"] = df_classes_live["total_count"].fillna(0).astype(int)

    # Guardrail: si hay muy pocos periodos, no hay “tendencia” real
    pc = _period_count_from_macro(df_macro_live)
    if pc < 2:
        st.info("Hay muy pocos periodos para ver tendencias. Prueba 'Semanas' o 'Meses', o amplía la ventana.")
        render_recent(df_view_ui)
        return

    # =========================================================
    # 7) Acción: Creciendo / Bajando / Se mantiene (DINÁMICO)
    # =========================================================
    if action in ("creciendo", "bajando", "neutro"):
        st.subheader("Tendencia (dinámica · ventana actual)")

        if df_classes_live.empty or "class" not in df_classes_live.columns:
            st.info("No hay clasificación suficiente para mostrar tendencia.")
            return

        if action == "creciendo":
            need_class = "emergente"
            title_class = "Creciendo (emergente)"
        elif action == "bajando":
            need_class = "declive"
            title_class = "Bajando (declive)"
        else:
            need_class = "consolidada"
            title_class = "Se mantiene (consolidada)"

        sub = df_classes_live[df_classes_live["class"].astype(str) == need_class].copy()
        if sub.empty:
            st.info(f"No hay macro-áreas en clase: {need_class} para esta ventana.")
            return

        # normaliza columnas numéricas por robustez
        for c in ["growth", "slope", "stability", "total_count"]:
            if c not in sub.columns:
                sub[c] = 0.0

        # ranking por clase
        if need_class == "emergente":
            sub = sub.sort_values(["growth", "slope", "total_count"], ascending=[False, False, False])
        elif need_class == "declive":
            sub = sub.sort_values(["growth", "slope", "total_count"], ascending=[True, True, False])
        else:
            sub = sub.sort_values(["stability", "total_count", "growth"], ascending=[False, False, False])

        show = sub[[c for c in ["macro_area", "total_count", "growth", "stability", "slope", "n_periods"] if c in sub.columns]].copy()
        show = show.rename(
            columns={
                "macro_area": "macro-área",
                "total_count": "docs",
                "growth": "cambio",
                "stability": "estabilidad",
                "slope": "pendiente",
                "n_periods": "periodos",
            }
        )
        st.caption(title_class)
        st.dataframe(show, use_container_width=True, hide_index=True)
        download_table(show, filename_prefix=f"live_dynamic_{need_class}_macro")

        macro_list = sub["macro_area"].astype(str).tolist()
        default_macro = macro_list[0] if macro_list else None
        if macro_selected != "Todas" and macro_selected in macro_list:
            default_macro = macro_selected

        macro_pick = st.selectbox(
            "Macro-área para ver su tendencia",
            macro_list,
            index=macro_list.index(default_macro) if (default_macro in macro_list) else 0,
            key=f"live_dyn_macro_pick_{need_class}",
        )

        d = df_macro_live[df_macro_live["macro_area"].astype(str) == str(macro_pick)].sort_values("period")
        if d.empty:
            st.info("No hay serie para la macro-área seleccionada en la ventana actual.")
            return

        metric = st.selectbox(
            "Métrica",
            ["Proporción (rel_freq)", "Volumen (count)"],
            index=0,
            key=f"live_dyn_trend_metric_{need_class}",
        )
        if metric.startswith("Proporción"):
            y_col = "rel_freq"
            ylabel = "rel_freq"
        else:
            y_col = "count"
            ylabel = "count"

        periods = d["period"].astype(str).tolist()
        y = d[y_col].astype(float).values
        _plot_series(periods, y, title=f"Live dinámico · {title_class} · {macro_pick}", ylabel=ylabel)

        with st.expander("Top-N (gráficas rápidas)", expanded=False):
            top_n = st.slider("Top-N a graficar", 1, 8, 3, key=f"live_dyn_topn_{need_class}")
            top_macros = sub["macro_area"].astype(str).tolist()[: int(top_n)]
            for mm in top_macros:
                dd = df_macro_live[df_macro_live["macro_area"].astype(str) == str(mm)].sort_values("period")
                if dd.empty:
                    continue
                pp = dd["period"].astype(str).tolist()
                yy = dd[y_col].astype(float).values
                _plot_series(pp, yy, title=f"{title_class} · {mm}", ylabel=ylabel)

        return

    # =========================================================
    # 8) Acción: Comparar (DINÁMICO)
    # =========================================================
    if action == "comparar":
        st.subheader("Comparar dos macro-áreas (dinámico · ventana actual)")

        areas = sorted(df_macro_live["macro_area"].dropna().astype(str).unique().tolist())
        if len(areas) < 2:
            st.info("No hay suficientes macro-áreas para comparar en la ventana actual.")
            return

        if macro_selected != "Todas" and macro_selected in areas:
            default_a = macro_selected
            remaining = [x for x in areas if x != default_a]
            default_b = remaining[0] if remaining else areas[0]
        else:
            default_a, default_b = areas[0], areas[1]

        c1, c2 = st.columns(2)
        with c1:
            a = st.selectbox("Macro-área A", areas, index=areas.index(default_a), key="live_cmp_a_dyn")
        with c2:
            b = st.selectbox("Macro-área B", areas, index=areas.index(default_b), key="live_cmp_b_dyn")

        metric = st.selectbox("Métrica", ["Proporción (rel_freq)", "Volumen (count)"], index=0, key="live_cmp_metric_dyn")
        y_col = "rel_freq" if metric.startswith("Proporción") else "count"

        da = df_macro_live[df_macro_live["macro_area"].astype(str) == str(a)][["period", y_col]].copy()
        db = df_macro_live[df_macro_live["macro_area"].astype(str) == str(b)][["period", y_col]].copy()

        m = (
            da.merge(db, on="period", how="outer", suffixes=("_a", "_b"))
            .fillna(0.0)
            .sort_values("period")
        )

        periods = m["period"].astype(str).tolist()
        ya = m[f"{y_col}_a"].astype(float).values
        yb = m[f"{y_col}_b"].astype(float).values

        fig, ax = plt.subplots(figsize=(10.8, 4.2))
        x = np.arange(len(periods))
        ax.plot(x, ya, label=a)
        ax.plot(x, yb, label=b)
        ax.set_title("Live dinámico · Comparación")
        ax.set_xlabel("Periodo")
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.25)

        n = len(periods)
        step = tick_step(n)
        tick_pos = x[::step]
        tick_lab = [periods[i] for i in range(0, n, step)]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, rotation=30, ha="right")
        ax.legend()
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    # =========================================================
    # 9) Acción: Predicción (DINÁMICA sobre ventana LIVE)
    # =========================================================
    st.subheader("Predicción (dinámica · ventana actual)")

    macro_list = sorted(df_macro_live["macro_area"].dropna().astype(str).unique().tolist())
    if not macro_list:
        st.info("No hay macro-áreas disponibles para predicción en la ventana actual.")
        return

    if macro_selected != "Todas" and macro_selected in macro_list:
        macro_pick = macro_selected
        st.caption(f"Macro-área seleccionada: **{macro_pick}**")
    else:
        macro_pick = st.selectbox("Macro-área a predecir", macro_list, key="live_pred_macro_dyn")

    metric = st.selectbox(
        "Métrica",
        ["Proporción (rel_freq)", "Volumen (count)"],
        index=0,
        key="live_pred_metric_dyn",
    )
    y_col = "rel_freq" if metric.startswith("Proporción") else "count"

    d = df_macro_live[df_macro_live["macro_area"].astype(str) == str(macro_pick)].sort_values("period")
    if d.empty:
        st.info("No hay serie para esa macro-área en la ventana actual.")
        return

    periods_hist = d["period"].astype(str).tolist()
    y_hist = d[y_col].astype(float).values

    h = st.slider("Horizonte (periodos)", 2, 18, 6, key="live_pred_h_dyn")
    sp = _infer_seasonal_periods(freq)
    y_fc = _forecast_linear_seasonal_naive(y_hist, h=int(h), sp=int(sp))

    future_periods = _future_period_labels(periods_hist[-1], freq=freq, h=int(h))

    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    xh = np.arange(len(periods_hist))
    ax.plot(xh, y_hist, label="histórico (ventana live)")

    xf = np.arange(len(periods_hist), len(periods_hist) + len(future_periods))
    ax.plot(xf, y_fc, label="predicción (fallback)")

    ax.set_title(f"Live dinámico · Predicción: {macro_pick}")
    ax.set_xlabel("Periodo")
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)

    all_periods = periods_hist + future_periods
    n = len(all_periods)
    step = tick_step(n)
    tick_pos = np.arange(n)[::step]
    tick_lab = [all_periods[i] for i in range(0, n, step)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=30, ha="right")
    ax.legend()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption("Modelo: **linear+seasonal_naive (dinámico)**")
