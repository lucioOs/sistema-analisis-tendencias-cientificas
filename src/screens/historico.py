# src/screens/historico.py
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import MaxNLocator

from src.load_data import enforce_schema
from src.config import HIST_YEARS_KEEP, MAX_ROWS_TEXT
from src.data.datasets import load_historico_dataset
from src.data.io import safe_last_update_label
from src.metrics import limit_df, period_count
from src.plotting.charts import tick_step
from src.ui.widgets import download_table, render_recent, show_kpis  # ✅ sin render_actions_header
from src.ui.wordcloud_ui import render_wordcloud


PROCESSED_DIR = Path("data/processed")
P_MACRO_TRENDS = PROCESSED_DIR / "macro_trends_full.parquet"
P_MACRO_CLASSES = PROCESSED_DIR / "macro_trend_classes.parquet"
P_MACRO_FORECAST = PROCESSED_DIR / "macro_trends_forecast.parquet"


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


def _apply_year_cap(df: pd.DataFrame, years_keep: int) -> pd.DataFrame:
    if df.empty or years_keep <= 0 or "date" not in df.columns:
        return df
    mx = pd.to_datetime(df["date"], errors="coerce").max()
    if pd.isna(mx):
        return df
    cut = mx - pd.Timedelta(days=int(years_keep) * 365)
    return df[pd.to_datetime(df["date"], errors="coerce") >= cut].copy()


def _filter_macro(df: pd.DataFrame, macro_area: str) -> pd.DataFrame:
    if df.empty or not macro_area or macro_area == "Todas":
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


def _build_ts(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
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


def _filter_macro_period_df(
    df_macro: pd.DataFrame,
    start_date: Optional[date],
    end_date: Optional[date],
    freq: str,
) -> pd.DataFrame:
    if df_macro is None or df_macro.empty or "period" not in df_macro.columns:
        return df_macro
    if not (start_date and end_date):
        return df_macro

    try:
        pi = pd.PeriodIndex(df_macro["period"].astype(str), freq=freq)
        p0 = pd.Period(pd.Timestamp(start_date), freq=freq)
        p1 = pd.Period(pd.Timestamp(end_date), freq=freq)
        m = (pi >= p0) & (pi <= p1)
        return df_macro.loc[m].copy()
    except Exception:
        return df_macro


def _plot_macro_series(periods: list[str], y: np.ndarray, title: str, ylabel: str) -> None:
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


def _plot_compare(periods: list[str], a: np.ndarray, b: np.ndarray, label_a: str, label_b: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    x = np.arange(len(periods))
    ax.plot(x, a, label=label_a)
    ax.plot(x, b, label=label_b)
    ax.set_title("Comparación")
    ax.set_xlabel("Periodo")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    n = len(periods)
    step = tick_step(n)
    tick_pos = x[::step]
    tick_lab = [periods[i] for i in range(0, n, step)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _max_view_by_freq(freq: str) -> int:
    # Recorte VISUAL para que NO se aplaste la predicción (solo si NO hay rango manual)
    if freq == "M":
        return 120   # ~10 años
    if freq == "W":
        return 260   # ~5 años
    if freq == "D":
        return 365   # ~1 año
    return 120


def screen_historico(
    *,
    macro_selected: str,
    action: str,
    start_date: Optional[date],
    end_date: Optional[date],
    freq: str,
    cloud_mode: str,
    ngram_max: int,
    min_df: int,
) -> None:
    df_raw, label, p = load_historico_dataset()
    st.caption(f"{label} · {safe_last_update_label(p, 'Última actualización')}")

    if df_raw is None or df_raw.empty:
        st.warning("No hay datos históricos listos para mostrar.")
        return

    df_raw = enforce_schema(df_raw, min_text_len=20, drop_duplicates=True)
    df_raw = _apply_year_cap(df_raw, int(HIST_YEARS_KEEP))
    df_raw = limit_df(df_raw, MAX_ROWS_TEXT)

    st.subheader("Histórico")

    # Macro-área se controla desde el sidebar
    df_view = _filter_macro(df_raw, macro_selected)
    df_view = _apply_date_range(df_view, start_date, end_date, date_col="date")

    with st.expander("Filtros activos", expanded=False):
        st.write(f"Macro-área: **{macro_selected}**")
        st.write(f"Acción: **{action}**")
        st.write(f"Frecuencia: **{freq}**")
        st.write(f"Nube: **{cloud_mode}**")
        if start_date and end_date:
            st.write(f"Rango: **{start_date} → {end_date}**")
        else:
            st.write("Rango: **(sin rango manual)**")

    if df_view.empty:
        st.warning("No hay registros con el filtro actual (Macro-área / Rango).")
        render_recent(df_raw.head(2000).copy() if not df_raw.empty else df_raw)
        return

    show_kpis(df_view, freq)

    render_wordcloud(
    df_view,
    "Nube de palabras (Histórico)",
    mode=cloud_mode,
    ngram_max=ngram_max,
    min_df=min_df,
)

    df_ts = _build_ts(df_view, freq=freq)
    st.subheader("Evolución por periodo")
    if df_ts.empty or df_ts["n"].sum() == 0:
        st.info("No hay suficientes datos para construir la serie temporal con el filtro actual.")
    else:
        st.line_chart(df_ts.set_index("date")["n"])
        st.dataframe(df_ts, use_container_width=True, hide_index=True)

    if period_count(df_view, freq) < 2:
        st.info("Hay muy pocos periodos para ver tendencias. Prueba 'Semanas' o 'Meses'.")
        render_recent(df_view)
        return

    df_macro = _read_cached(P_MACRO_TRENDS)
    df_classes = _read_cached(P_MACRO_CLASSES)
    df_fc = _read_cached(P_MACRO_FORECAST)

    if df_macro.empty:
        st.error("No existe macro_trends_full.parquet. Ejecuta: python -m src.forecast_trends")
        return

    df_macro_f = df_macro if macro_selected == "Todas" else df_macro[df_macro["macro_area"] == macro_selected].copy()
    df_cls_f = df_classes if macro_selected == "Todas" else df_classes[df_classes["macro_area"] == macro_selected].copy()
    df_fc_f = df_fc if macro_selected == "Todas" else df_fc[df_fc["macro_area"] == macro_selected].copy()

    df_macro_f = _filter_macro_period_df(df_macro_f, start_date, end_date, freq=freq)

    # =========================
    # Resumen de clasificación
    # =========================
    with st.expander("Resumen de clasificación (macro-áreas)", expanded=False):
        if df_classes.empty:
            st.info("No hay clasificación disponible (macro_trend_classes.parquet).")
        else:
            show_cols = [
                c for c in ["macro_area", "class", "total_count", "growth", "stability", "slope", "n_periods"]
                if c in df_classes.columns
            ]
            st.dataframe(
                df_classes[show_cols].sort_values(["class", "total_count"], ascending=[True, False]),
                use_container_width=True,
                hide_index=True,
            )

    # =========================
    # Creciendo / Bajando
    # =========================
    if action in ("creciendo", "bajando"):
        st.subheader("Resultados")

        if df_classes.empty:
            st.info("No hay clasificación para mostrar.")
            return

        label_need = "emergente" if action == "creciendo" else "declive"
        sub = df_classes[df_classes["class"] == label_need].copy()

        if macro_selected != "Todas":
            sub = sub[sub["macro_area"] == macro_selected].copy()

        if sub.empty:
            st.info("No se encontraron macro-áreas con esa clasificación.")
            return

        show = sub[[c for c in ["macro_area", "total_count", "growth", "stability", "slope"] if c in sub.columns]].copy()
        show = show.rename(
            columns={
                "macro_area": "macro-área",
                "total_count": "docs",
                "growth": "cambio",
                "stability": "estabilidad",
                "slope": "pendiente",
            }
        )
        st.dataframe(show, use_container_width=True, hide_index=True)
        download_table(show, filename_prefix=f"historico_{label_need}_macro")

        if macro_selected == "Todas":
            st.info("Selecciona una macro-área específica en el sidebar para ver su tendencia.")
            return

        macro_pick = macro_selected
        d = df_macro[df_macro["macro_area"] == macro_pick].sort_values("period")
        d = _filter_macro_period_df(d, start_date, end_date, freq=freq)
        if d.empty:
            st.info("No hay serie para la macro-área seleccionada en el rango actual.")
            return

        y_col = "rel_freq" if "rel_freq" in d.columns else "count"
        periods = d["period"].astype(str).tolist()
        y = d[y_col].astype(float).values
        _plot_macro_series(periods, y, title=f"Tendencia (Histórico): {macro_pick}", ylabel=y_col)
        return

    # =========================
    # Comparar
    # =========================
    if action == "comparar":
        st.subheader("Comparar dos macro-áreas")

        areas = sorted(df_macro["macro_area"].dropna().astype(str).unique().tolist())
        if len(areas) < 2:
            st.info("No hay suficientes macro-áreas para comparar.")
            return

        if macro_selected != "Todas":
            default_a = macro_selected
            remaining = [x for x in areas if x != default_a]
            if not remaining:
                st.info("No hay otra macro-área para comparar.")
                return
            default_b = remaining[0]
        else:
            default_a, default_b = areas[0], areas[1]

        c1, c2 = st.columns(2)
        with c1:
            a = st.selectbox("Macro-área A", areas, index=areas.index(default_a), key="hist_cmp_a")
        with c2:
            b = st.selectbox("Macro-área B", areas, index=areas.index(default_b), key="hist_cmp_b")

        da = _filter_macro_period_df(df_macro[df_macro["macro_area"] == a].sort_values("period"), start_date, end_date, freq=freq)
        db = _filter_macro_period_df(df_macro[df_macro["macro_area"] == b].sort_values("period"), start_date, end_date, freq=freq)

        if da.empty or db.empty:
            st.info("No se pudieron obtener series para comparar en el rango actual.")
            return

        y_col = "rel_freq" if "rel_freq" in df_macro.columns else "count"
        m = (
            da[["period", y_col]]
            .merge(db[["period", y_col]], on="period", how="outer", suffixes=("_a", "_b"))
            .fillna(0.0)
            .sort_values("period")
        )

        periods = m["period"].astype(str).tolist()
        ya = m[f"{y_col}_a"].astype(float).values
        yb = m[f"{y_col}_b"].astype(float).values
        _plot_compare(periods, ya, yb, a, b, ylabel=y_col)
        return

    # =========================
    # Predicción
    # =========================
    st.subheader("Predicción (macro-áreas)")

    if df_fc.empty:
        st.error("No existe macro_trends_forecast.parquet. Ejecuta: python -m src.forecast_trends")
        return

    if macro_selected == "Todas":
        st.info("Selecciona una macro-área específica en el sidebar para ejecutar la predicción.")
        return

    macro_pick = macro_selected
    st.caption(f"Macro-área seleccionada: **{macro_pick}**")

    d_hist = df_macro[df_macro["macro_area"] == macro_pick].sort_values("period")
    d_fc = df_fc[df_fc["macro_area"] == macro_pick].sort_values("period")

    d_hist = _filter_macro_period_df(d_hist, start_date, end_date, freq=freq)

    if d_hist.empty:
        st.info("No hay histórico para esa macro-área en el rango actual.")
        return
    if d_fc.empty:
        st.info("No hay predicción para esa macro-área.")
        return

    metric = st.selectbox("Métrica", ["Proporción (rel_freq)", "Volumen (count)"], index=0, key="hist_pred_metric")

    if metric.startswith("Proporción"):
        y_hist = d_hist["rel_freq"].astype(float).values if "rel_freq" in d_hist.columns else d_hist["count"].astype(float).values
        y_fc = d_fc["rel_freq_pred"].astype(float).values
        ylabel = "rel_freq"
    else:
        y_hist = d_hist["count"].astype(float).values
        y_fc = d_fc["count_pred"].astype(float).values if "count_pred" in d_fc.columns else d_fc["rel_freq_pred"].astype(float).values
        ylabel = "count"

    periods_hist = d_hist["period"].astype(str).tolist()
    periods_fc = d_fc["period"].astype(str).tolist()

    # Recorte VISUAL para no aplastar la predicción (solo si NO hay rango manual)
    if not (start_date and end_date):
        max_view = _max_view_by_freq(freq)
        if len(periods_hist) > max_view:
            periods_hist = periods_hist[-max_view:]
            y_hist = y_hist[-max_view:]

    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    xh = np.arange(len(periods_hist))
    ax.plot(xh, y_hist, label="histórico")

    xf = np.arange(len(periods_hist), len(periods_hist) + len(periods_fc))
    ax.plot(xf, y_fc, label="predicción")

    ax.set_title(f"Predicción: {macro_pick}")
    ax.set_xlabel("Periodo")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    all_periods = periods_hist + periods_fc
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

    model_name = d_fc["model"].iloc[0] if "model" in d_fc.columns and not d_fc.empty else "desconocido"
    st.caption(f"Modelo: **{model_name}**")
