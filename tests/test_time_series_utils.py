from __future__ import annotations

import pandas as pd

from tests.test_holtwinters import (
    build_keyword_count_series,
    build_synthetic_dataset,
    normalize_series_monthly,
)


def test_build_keyword_count_series_counts_matches() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-15", "2024-01-20", "2024-02-01"],
            "text_clean": ["llm trends", "no keyword", "another llm paper"],
        }
    )
    series, rows_matched = build_keyword_count_series(df, "llm")

    assert rows_matched == 2
    assert float(series.sum()) == 2.0


def test_normalize_series_monthly_fills_missing_months() -> None:
    raw = pd.Series([2, 3], index=pd.to_datetime(["2024-01-31", "2024-03-31"]))
    out = normalize_series_monthly(raw, months=12, fill_value=0)

    assert len(out) == 3
    assert out.loc[pd.Timestamp("2024-02-29")] == 0


def test_build_synthetic_dataset_has_required_columns() -> None:
    df = build_synthetic_dataset(months=24, docs_per_month=10, keyword="llm")
    assert {"date", "text_clean"}.issubset(df.columns)
    assert len(df) >= 24 * 5


def test_build_keyword_count_series_raises_if_missing_columns() -> None:
    df = pd.DataFrame({"date": ["2024-01-01"]})
    try:
        build_keyword_count_series(df, "llm")
        assert False, "Debe lanzar ValueError por ausencia de text_clean"
    except ValueError as exc:
        assert "text_clean" in str(exc)
