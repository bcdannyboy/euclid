from __future__ import annotations

from typing import Any, Mapping, Sequence


def sanitized_data_window(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]] = (),
    test_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    all_rows = tuple(train_rows) + tuple(validation_rows) + tuple(test_rows)
    return {
        "train_row_count": len(train_rows),
        "validation_row_count": len(validation_rows),
        "test_row_count": len(test_rows),
        "first_event_time": None if not all_rows else str(all_rows[0]["event_time"]),
        "last_event_time": None if not all_rows else str(all_rows[-1]["event_time"]),
    }


def statistical_baseline_diagnostics(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    if not train_rows or not feature_names:
        return {"status": "not_applicable", "reason": "no_features_or_rows"}
    try:
        import numpy as np
        import statsmodels.api as sm
        from sklearn.linear_model import LinearRegression
    except Exception as exc:  # pragma: no cover - dependency shape differs by env
        return {
            "status": "unavailable",
            "reason": type(exc).__name__,
        }

    x_matrix = np.asarray(
        [[float(row[name]) for name in feature_names] for row in train_rows],
        dtype=float,
    )
    y_vector = np.asarray([float(row["target"]) for row in train_rows], dtype=float)
    sklearn_model = LinearRegression().fit(x_matrix, y_vector)
    statsmodels_model = sm.OLS(y_vector, sm.add_constant(x_matrix)).fit()
    return {
        "status": "available",
        "sklearn_backend": "sklearn.linear_model.LinearRegression",
        "statsmodels_backend": "statsmodels.OLS",
        "sklearn_r2": float(sklearn_model.score(x_matrix, y_vector)),
        "statsmodels_rsquared": float(statsmodels_model.rsquared),
    }


__all__ = ["sanitized_data_window", "statistical_baseline_diagnostics"]
