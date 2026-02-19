# tests/run_evidence.py
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_text(p: Path, lines: List[str]) -> None:
    p.write_text("\n".join(lines), encoding="utf-8")


def write_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def now_ms() -> int:
    return int(time.time() * 1000)


def safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    return pd.read_parquet(path)


def to_year_series(df: pd.DataFrame, date_col: str) -> pd.Series:
    # robusto a formatos raros
    dt = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    return dt.dt.year


def classify_slope(m: float, eps: float) -> str:
    if m > eps:
        return "Creciente"
    if m < -eps:
        return "Decreciente"
    return "Estable"


def top_terms_from_kmeans(km: KMeans, terms: np.ndarray, topn: int = 10) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    centers = km.cluster_centers_
    for i in range(km.n_clusters):
        idx = np.argsort(centers[i])[-topn:][::-1]
        out[i] = terms[idx].tolist()
    return out


def run_safely(name: str, fn, results: Dict[str, Any], errors: Dict[str, Any]) -> None:
    t0 = now_ms()
    try:
        results[name] = fn()
        results[name]["_status"] = "ok"
    except Exception as e:
        errors[name] = {"error": str(e), "traceback": traceback.format_exc()}
        results[name] = {"_status": "failed"}
    finally:
        results[name]["_elapsed_ms"] = now_ms() - t0


# -----------------------------
# Sampling (fast + representative)
# -----------------------------
def sample_stratified_by_year(
    df: pd.DataFrame,
    *,
    date_col: str,
    max_n: int,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Selecciona un subconjunto grande y representativo estratificado por año.
    Esto mantiene la distribución temporal para P3/P4 y acelera muchísimo.
    """
    meta: Dict[str, Any] = {"sampling": "none"}

    if max_n <= 0 or len(df) <= max_n:
        meta.update({"sampling": "none", "rows_in": int(len(df)), "rows_out": int(len(df))})
        return df, meta

    if date_col not in df.columns:
        # si no hay fecha, muestreamos simple (P1/P2 aún sirven)
        df2 = df.sample(max_n, random_state=random_state).reset_index(drop=True)
        meta.update({"sampling": "simple", "rows_in": int(len(df)), "rows_out": int(len(df2)), "max_n": max_n})
        return df2, meta

    tmp = df.copy()
    tmp["_year_tmp"] = to_year_series(tmp, date_col=date_col)
    tmp = tmp.dropna(subset=["_year_tmp"]).copy()
    tmp["_year_tmp"] = tmp["_year_tmp"].astype(int)

    if len(tmp) <= max_n:
        meta.update({"sampling": "none_after_dropna_year", "rows_in": int(len(df)), "rows_out": int(len(tmp))})
        tmp = tmp.drop(columns=["_year_tmp"])
        return tmp.reset_index(drop=True), meta

    frac = max_n / len(tmp)

    def _pick(g: pd.DataFrame) -> pd.DataFrame:
        n = max(1, int(round(len(g) * frac)))
        return g.sample(n=n, random_state=random_state)

    sampled = (
        tmp.groupby("_year_tmp", group_keys=False)
        .apply(_pick)
        .reset_index(drop=True)
        .drop(columns=["_year_tmp"])
    )

    meta.update(
        {
            "sampling": "stratified_by_year",
            "rows_in": int(len(df)),
            "rows_in_with_year": int(len(tmp)),
            "rows_out": int(len(sampled)),
            "max_n": max_n,
            "frac": float(frac),
        }
    )
    return sampled, meta


# -----------------------------
# Model build (computed once)
# -----------------------------
def build_topic_model(
    df: pd.DataFrame,
    *,
    text_col: str,
    max_df: float,
    min_df: int,
    ngram_max: int,
    stop_words: Optional[str],
    max_features: Optional[int],
    k: int,
    random_state: int,
    n_init: int,
) -> Tuple[TfidfVectorizer, Any, np.ndarray, KMeans]:
    tfidf = TfidfVectorizer(
        max_df=max_df,
        min_df=min_df,
        ngram_range=(1, ngram_max),
        stop_words=stop_words,
        max_features=max_features,
    )
    X = tfidf.fit_transform(df[text_col].astype(str))

    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X)
    return tfidf, X, labels, km


# -----------------------------
# Tests (P1..P4)
# -----------------------------
def P1_text(df_clean: pd.DataFrame, df_live: pd.DataFrame, out_dir: Path, *, clean_text_col: str, live_title_col: str) -> Dict[str, Any]:
    ensure_dir(out_dir)

    if clean_text_col not in df_clean.columns:
        raise KeyError(f"clean.parquet no tiene columna '{clean_text_col}'")
    if live_title_col not in df_live.columns:
        raise KeyError(f"live_dataset.parquet no tiene columna '{live_title_col}'")

    mean_len = float(df_clean[clean_text_col].astype(str).str.len().mean())

    cv = CountVectorizer()
    cv.fit(df_clean[clean_text_col].astype(str))
    vocab_size = int(len(cv.vocabulary_))

    sample = pd.concat(
        [
            df_live[[live_title_col]].head(5).reset_index(drop=True),
            df_clean[[clean_text_col]].head(5).reset_index(drop=True),
        ],
        axis=1,
    )
    sample_path = out_dir / "P1_before_after.csv"
    sample.to_csv(sample_path, index=False, encoding="utf-8")

    summary = {
        "rows_clean_used": int(len(df_clean)),
        "rows_live": int(len(df_live)),
        "mean_text_len": mean_len,
        "vocab_size": vocab_size,
        "before_after_csv": str(sample_path),
    }

    write_text(
        out_dir / "P1_summary.txt",
        [
            "PRUEBA 1 - Procesamiento y preparación del texto",
            f"- Registros (clean usado): {summary['rows_clean_used']}",
            f"- Longitud promedio text_clean: {mean_len:.2f}",
            f"- Tamaño vocabulario (CountVectorizer): {vocab_size}",
            f"- Evidencia antes/después: {sample_path.name}",
        ],
    )
    write_json(out_dir / "P1_summary.json", summary)
    return summary


def P2_topics(tfidf: TfidfVectorizer, X, labels: np.ndarray, km: KMeans, out_dir: Path, *, topn: int) -> Dict[str, Any]:
    ensure_dir(out_dir)
    docs, feats = X.shape
    sparsity = float(1.0 - (X.nnz / (docs * feats)))
    clusters = int(len(set(labels)))

    terms = tfidf.get_feature_names_out()
    top_terms = top_terms_from_kmeans(km, terms, topn=topn)

    top_terms_df = pd.DataFrame(
        [{"topic": t, "top_terms": ", ".join(words)} for t, words in sorted(top_terms.items())]
    )
    top_terms_path = out_dir / "P2_top_terms.csv"
    top_terms_df.to_csv(top_terms_path, index=False, encoding="utf-8")

    summary = {
        "docs": int(docs),
        "features": int(feats),
        "sparsity": sparsity,
        "clusters": clusters,
        "top_terms_csv": str(top_terms_path),
    }

    write_text(
        out_dir / "P2_summary.txt",
        [
            "PRUEBA 2 - Identificación de patrones temáticos",
            f"- Docs: {docs}",
            f"- Features (TF-IDF): {feats}",
            f"- Sparsity: {sparsity:.4f}",
            f"- Clusters detectados: {clusters}",
            f"- Top términos por tema: {top_terms_path.name}",
        ],
    )
    write_json(out_dir / "P2_summary.json", summary)
    return summary


def P3_temporal(
    df_clean: pd.DataFrame,
    labels: np.ndarray,
    out_dir: Path,
    *,
    date_col: str,
    slope_eps: float,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    if date_col not in df_clean.columns:
        raise KeyError(f"clean.parquet no tiene columna '{date_col}'")

    years = to_year_series(df_clean, date_col=date_col)
    mask = pd.notna(years).to_numpy()
    if mask.sum() < 10:
        raise ValueError("Muy pocos registros con fecha válida para análisis temporal.")

    df = df_clean.loc[mask].copy()
    df["year"] = years.loc[mask].astype(int).to_numpy()
    df["topic"] = labels[mask]

    table = pd.crosstab(df["year"], df["topic"]).sort_index()
    counts_path = out_dir / "P3_year_topic_counts.csv"
    table.to_csv(counts_path, encoding="utf-8")

    share = table.div(table.sum(axis=1), axis=0)

    slopes_rows = []
    for t in share.columns:
        y = share[t].values.reshape(-1, 1)
        x = np.arange(len(y)).reshape(-1, 1)
        m = float(LinearRegression().fit(x, y).coef_[0][0])
        slopes_rows.append({"topic": int(t), "slope_share": m, "clasificacion": classify_slope(m, slope_eps)})

    slopes_df = pd.DataFrame(slopes_rows).sort_values("topic")
    slopes_path = out_dir / "P3_slopes.csv"
    slopes_df.to_csv(slopes_path, index=False, encoding="utf-8")

    summary = {
        "years_min": int(table.index.min()),
        "years_max": int(table.index.max()),
        "counts_csv": str(counts_path),
        "slopes_csv": str(slopes_path),
    }

    write_text(
        out_dir / "P3_summary.txt",
        [
            "PRUEBA 3 - Evolución temporal de los temas",
            f"- Rango años: {summary['years_min']} - {summary['years_max']}",
            f"- Evidencia tabla año×tema: {counts_path.name}",
            f"- Evidencia pendientes (share) + clasificación: {slopes_path.name}",
        ],
    )
    write_json(out_dir / "P3_summary.json", summary)
    return summary


def P4_backtesting(
    df_clean: pd.DataFrame,
    labels: np.ndarray,
    out_dir: Path,
    *,
    date_col: str,
    horizon_years: int,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    if date_col not in df_clean.columns:
        raise KeyError(f"clean.parquet no tiene columna '{date_col}'")

    years = to_year_series(df_clean, date_col=date_col)
    mask = pd.notna(years).to_numpy()
    if mask.sum() < 10:
        raise ValueError("Muy pocos registros con fecha válida para backtesting.")

    df = df_clean.loc[mask].copy()
    df["year"] = years.loc[mask].astype(int).to_numpy()
    df["topic"] = labels[mask]

    table = pd.crosstab(df["year"], df["topic"]).sort_index()
    years_idx = table.index.values

    if len(years_idx) <= horizon_years:
        raise ValueError(f"No hay suficientes años: {len(years_idx)} para horizon={horizon_years}")

    split = len(years_idx) - horizon_years
    years_test = years_idx[split:]

    rows = []
    for t in table.columns:
        y = table[t].values.astype(float)
        y_train = y[:split]
        y_test = y[split:]

        x_train = np.arange(len(y_train)).reshape(-1, 1)
        x_test = np.arange(len(y_train), len(y)).reshape(-1, 1)

        lr = LinearRegression().fit(x_train, y_train)
        pred = lr.predict(x_test)

        mae = float(mean_absolute_error(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))

        rows.append(
            {
                "topic": int(t),
                "years_test": json.dumps(years_test.tolist()),
                "mae": mae,
                "rmse": rmse,
                "real": json.dumps([float(v) for v in y_test]),
                "pred": json.dumps([float(v) for v in pred]),
            }
        )

    df_metrics = pd.DataFrame(rows).sort_values("topic")
    metrics_path = out_dir / "P4_backtesting_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False, encoding="utf-8")

    summary = {
        "years_test": years_test.tolist(),
        "horizon_years": horizon_years,
        "metrics_csv": str(metrics_path),
    }

    write_text(
        out_dir / "P4_summary.txt",
        [
            "PRUEBA 4 - Predicción (backtesting)",
            f"- Años test: {years_test.tolist()}",
            f"- Evidencia métricas MAE/RMSE + real vs pred: {metrics_path.name}",
        ],
    )
    write_json(out_dir / "P4_summary.json", summary)
    return summary


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Genera evidencias (P1..P4) rápido y reproducible.")
    ap.add_argument("--clean", default="data/processed/clean.parquet")
    ap.add_argument("--live", default="data/live/live_dataset.parquet")
    ap.add_argument("--out", default="evidence")

    # Datos / columnas
    ap.add_argument("--text_col", default="text_clean")
    ap.add_argument("--date_col", default="date")
    ap.add_argument("--live_title_col", default="title")

    # Modelo (consistentes)
    ap.add_argument("--max_df", type=float, default=0.9)
    ap.add_argument("--min_df", type=int, default=5)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_init", type=int, default=5)  # más rápido para pruebas

    # Stopwords: por defecto NONE (sin stopwords), como pediste
    ap.add_argument("--stop_words", default="none")  # "none" o "english"

    # Performance control
    ap.add_argument("--max_n", type=int, default=120000, help="Subconjunto grande (estratificado por año). 0=sin muestreo")
    ap.add_argument("--max_features", type=int, default=100000, help="Límite de features TF-IDF (0=sin límite)")

    # P3/P4
    ap.add_argument("--slope_eps", type=float, default=0.0015, help="Umbral para clasificar pendiente share")
    ap.add_argument("--horizon_years", type=int, default=5, help="Años finales para backtesting")
    ap.add_argument("--topn_terms", type=int, default=10)

    args = ap.parse_args()

    stop_words = None if str(args.stop_words).lower() == "none" else args.stop_words
    max_features = None if int(args.max_features) <= 0 else int(args.max_features)

    root = Path(args.out)
    ensure_dir(root)

    # Load once
    df_clean_full = safe_read_parquet(Path(args.clean))
    df_live = safe_read_parquet(Path(args.live))

    # Sample once (fast + representative)
    df_clean, sample_meta = sample_stratified_by_year(
        df_clean_full,
        date_col=args.date_col,
        max_n=int(args.max_n),
        random_state=int(args.random_state),
    )

    # Global report
    results: Dict[str, Any] = {
        "params": vars(args),
        "sampling": sample_meta,
        "tests": {},
    }
    errors: Dict[str, Any] = {}

    # P1 (sobre df_clean muestreado; es válido para evidencia y acelera)
    run_safely(
        "P1_texto",
        lambda: P1_text(
            df_clean, df_live, root / "P1_texto", clean_text_col=args.text_col, live_title_col=args.live_title_col
        ),
        results["tests"],
        errors,
    )

    # Build model once for P2..P4
    tfidf = X = labels = km = None
    try:
        if args.text_col not in df_clean.columns:
            raise KeyError(f"clean.parquet no tiene columna '{args.text_col}'")

        tfidf, X, labels, km = build_topic_model(
            df_clean,
            text_col=args.text_col,
            max_df=float(args.max_df),
            min_df=int(args.min_df),
            ngram_max=int(args.ngram_max),
            stop_words=stop_words,
            max_features=max_features,
            k=int(args.k),
            random_state=int(args.random_state),
            n_init=int(args.n_init),
        )

        # Guardar metadatos del modelo para defensa
        model_meta = {
            "tfidf": {"max_df": args.max_df, "min_df": args.min_df, "ngram_range": [1, args.ngram_max], "stop_words": args.stop_words, "max_features": max_features},
            "kmeans": {"k": args.k, "random_state": args.random_state, "n_init": args.n_init},
            "matrix_shape": {"docs": int(X.shape[0]), "features": int(X.shape[1])},
        }
        write_json(root / "MODEL_meta.json", model_meta)

    except Exception as e:
        errors["MODEL_BUILD"] = {"error": str(e), "traceback": traceback.format_exc()}

    # If model built, run remaining tests. If not, mark as failed but keep artifacts.
    if tfidf is None:
        results["tests"]["P2_temas"] = {"_status": "failed", "_reason": "model_build_failed"}
        results["tests"]["P3_evolucion"] = {"_status": "failed", "_reason": "model_build_failed"}
        results["tests"]["P4_prediccion"] = {"_status": "failed", "_reason": "model_build_failed"}
    else:
        run_safely(
            "P2_temas",
            lambda: P2_topics(tfidf, X, labels, km, root / "P2_temas", topn=int(args.topn_terms)),
            results["tests"],
            errors,
        )
        run_safely(
            "P3_evolucion",
            lambda: P3_temporal(df_clean, labels, root / "P3_evolucion", date_col=args.date_col, slope_eps=float(args.slope_eps)),
            results["tests"],
            errors,
        )
        run_safely(
            "P4_prediccion",
            lambda: P4_backtesting(df_clean, labels, root / "P4_prediccion", date_col=args.date_col, horizon_years=int(args.horizon_years)),
            results["tests"],
            errors,
        )

    # Final reports
    write_json(root / "report.json", results)
    if errors:
        write_json(root / "errors.json", errors)
        write_text(root / "RUN_STATUS.txt", ["EVIDENCIAS GENERADAS CON ERRORES", "Revisa evidence/errors.json"])
        print("Listo con errores. Revisa:", (root / "errors.json").resolve())
    else:
        write_text(root / "RUN_STATUS.txt", ["EVIDENCIAS GENERADAS OK"])
        print("OK. Evidencias en:", root.resolve())


if __name__ == "__main__":
    main()
