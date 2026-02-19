# src/ui/wordcloud_ui.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.analytics.trends_engine import STOPWORDS_ALL

CACHE_TTL_SEC = 300
MAX_WC_DOCS = 600
MAX_WC_STRATA = 24

try:
    from wordcloud import WordCloud  # type: ignore
    WORDCLOUD_OK = True
except Exception:
    WordCloud = None  # type: ignore
    WORDCLOUD_OK = False


# =============================================================================
# Sampling
# =============================================================================
def _pick_text_col(df: pd.DataFrame) -> str:
    """
    Elige una columna de texto disponible (prioridad):
      text_clean -> text -> abstract -> summary -> title
    """
    for c in ["text_clean", "text", "abstract", "summary", "title"]:
        if c in df.columns:
            return c
    return ""


def _stratified_time_sample(
    df: pd.DataFrame,
    max_docs: int,
    strata: int = MAX_WC_STRATA,
    seed: int = 7,
) -> List[str]:
    """
    Muestreo estratificado por tiempo:
    - conserva diversidad temporal (evita que todo sea "lo último")
    - controla costo de TF-IDF/CountVectorizer
    """
    if df is None or df.empty:
        return []

    text_col = _pick_text_col(df)
    if not text_col:
        return []

    # Orden temporal
    dfx = df.sort_values("date").copy() if "date" in df.columns else df.copy()
    n = len(dfx)
    if n <= max_docs:
        return dfx[text_col].astype(str).fillna("").tolist()

    strata = max(4, int(strata))
    per = max(1, int(max_docs // strata))
    out_texts: List[str] = []

    edges = np.linspace(0, n, strata + 1).astype(int)
    rng = np.random.default_rng(seed)
    for i in range(strata):
        a, b = int(edges[i]), int(edges[i + 1])
        if b <= a:
            continue
        part = dfx.iloc[a:b]
        if len(part) <= per:
            out_texts.extend(part[text_col].astype(str).fillna("").tolist())
        else:
            idx = rng.choice(len(part), size=per, replace=False)
            out_texts.extend(part.iloc[idx][text_col].astype(str).fillna("").tolist())

    # Limpieza mínima
    out_texts = [t.strip() for t in out_texts if isinstance(t, str) and t.strip()]
    return out_texts[:max_docs]


# =============================================================================
# Vectorization (TF-IDF / Frequency)
# =============================================================================
def _normalize_ngram_max(x: int) -> int:
    try:
        v = int(x)
    except Exception:
        v = 2
    return max(1, min(3, v))


def _normalize_min_df(x: int) -> int:
    try:
        v = int(x)
    except Exception:
        v = 2
    return max(1, min(50, v))


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def _wc_freq_from_tfidf(
    texts: Tuple[str, ...],
    ngram_max: int,
    min_df: int,
) -> Dict[str, float]:
    if not texts:
        return {}

    ngram_max = _normalize_ngram_max(ngram_max)
    min_df = _normalize_min_df(min_df)

    vec = TfidfVectorizer(
        stop_words=list(STOPWORDS_ALL),
        lowercase=True,
        ngram_range=(1, ngram_max),
        max_features=8000,
        min_df=min_df,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]+\b",
    )
    X = vec.fit_transform(list(texts))
    terms = vec.get_feature_names_out()
    weights = np.asarray(X.sum(axis=0)).ravel().astype(float)

    freq: Dict[str, float] = {}
    for t, w in zip(terms, weights):
        if w > 0 and len(t) >= 3:
            freq[t] = float(w)
    return freq


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def _wc_freq_from_counts(
    texts: Tuple[str, ...],
    ngram_max: int,
    min_df: int,
) -> Dict[str, float]:
    if not texts:
        return {}

    ngram_max = _normalize_ngram_max(ngram_max)
    min_df = _normalize_min_df(min_df)

    vec = CountVectorizer(
        stop_words=list(STOPWORDS_ALL),
        lowercase=True,
        ngram_range=(1, ngram_max),
        max_features=8000,
        min_df=min_df,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]+\b",
    )
    X = vec.fit_transform(list(texts))
    terms = vec.get_feature_names_out()
    counts = np.asarray(X.sum(axis=0)).ravel().astype(float)

    freq: Dict[str, float] = {}
    for t, c in zip(terms, counts):
        if c > 0 and len(t) >= 3:
            freq[t] = float(c)
    return freq


# =============================================================================
# Plotting
# =============================================================================
def _plot_from_freq(freq: Dict[str, float]) -> Optional[plt.Figure]:
    if not WORDCLOUD_OK or not freq:
        return None
    try:
        wc = WordCloud(
            width=1000,
            height=420,
            background_color=None,
            mode="RGBA",
            collocations=False,
            max_words=250,
        ).generate_from_frequencies(freq)

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.patch.set_alpha(0)
        return fig
    except Exception:
        return None


# =============================================================================
# Public UI
# =============================================================================
def render_wordcloud(
    df: pd.DataFrame,
    title: str,
    *,
    mode: str = "tfidf",   # "tfidf" | "freq"
    ngram_max: int = 2,
    min_df: int = 2,
) -> None:
    """
    Nube de palabras (robusta):
      - mode="tfidf": destacados por peso TF-IDF agregado
      - mode="freq": frecuencia bruta (CountVectorizer)
    Respeta:
      - ngram_max (1..3)
      - min_df (>=1)
    """
    with st.expander(title, expanded=True):
        if df is None or df.empty:
            st.info("No hay datos para generar la nube.")
            return
        if not WORDCLOUD_OK:
            st.info("WordCloud no está disponible en este entorno.")
            return

        texts = _stratified_time_sample(df, max_docs=MAX_WC_DOCS, strata=MAX_WC_STRATA)
        if not texts:
            st.info("No hay texto suficiente para generar la nube con los filtros actuales.")
            return

        # Normaliza tokens esperados desde sidebar
        mode = str(mode or "tfidf").strip().lower()
        mode = "freq" if mode in ("freq", "frecuencia", "count") else "tfidf"

        if mode == "tfidf":
            freq_map = _wc_freq_from_tfidf(tuple(texts), ngram_max=ngram_max, min_df=min_df)
        else:
            freq_map = _wc_freq_from_counts(tuple(texts), ngram_max=ngram_max, min_df=min_df)

        fig = _plot_from_freq(freq_map)

        if fig is None:
            st.info("No se pudo generar la nube (poca información o formato irregular).")
            return

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
