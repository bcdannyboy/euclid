from __future__ import annotations

from euclid.runtime.cache import EvaluationCache, cache_key_for


def test_evaluation_cache_keys_are_canonical_and_category_scoped() -> None:
    feature_payload = {
        "rows": [{"event_time": "2026-01-01", "x": 1.0}],
        "columns": ["x"],
    }
    reordered_payload = {
        "columns": ["x"],
        "rows": [{"x": 1.0, "event_time": "2026-01-01"}],
    }

    feature_key = cache_key_for("feature_matrix", feature_payload)
    reordered_key = cache_key_for("feature_matrix", reordered_payload)
    expression_key = cache_key_for("expression_evaluation", reordered_payload)

    assert feature_key == reordered_key
    assert feature_key != expression_key
    assert feature_key.startswith("cache:feature_matrix:sha256:")


def test_evaluation_cache_reuses_values_without_changing_results() -> None:
    cache = EvaluationCache()
    calls = 0

    def compute_matrix() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"columns": ["x"], "values": [[1.0], [2.0]]}

    first = cache.get_or_compute(
        category="feature_matrix",
        payload={"dataset": "fixture", "columns": ["x"]},
        compute=compute_matrix,
    )
    second = cache.get_or_compute(
        category="feature_matrix",
        payload={"columns": ["x"], "dataset": "fixture"},
        compute=compute_matrix,
    )

    assert first == second
    assert calls == 1
    assert cache.stats().hit_count == 1
    assert cache.stats().miss_count == 1


def test_evaluation_cache_covers_expression_subtree_constants_and_simplification_keys(
) -> None:
    cache = EvaluationCache()
    categories = (
        "expression_evaluation",
        "subtree_evaluation",
        "fitted_constants",
        "simplification",
    )

    for category in categories:
        cache.get_or_compute(
            category=category,
            payload={"expr": "x + 0", "assumptions": {"x": "real"}},
            compute=lambda category=category: {"category": category},
        )

    diagnostics = cache.replay_diagnostics()

    assert diagnostics["cache_status"] == "captured"
    assert diagnostics["stats"] == {
        "hit_count": 0,
        "miss_count": 4,
        "entry_count": 4,
    }
    assert [entry["category"] for entry in diagnostics["entries"]] == sorted(
        categories
    )
    assert all(entry["cache_key"].startswith(f"cache:{entry['category']}:") for entry in diagnostics["entries"])
