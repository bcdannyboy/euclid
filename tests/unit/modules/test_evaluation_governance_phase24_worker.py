from __future__ import annotations

from pathlib import Path
from typing import Any

from euclid.contracts.loader import load_contract_catalog
from euclid.modules.evaluation_governance import (
    build_predictive_gate_policy,
    predictive_governance_reason_codes,
    resolve_confirmatory_promotion_allowed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _predictive_gate_policy_manifest(
    *,
    allowed_forecast_object_types: tuple[str, ...] = ("point",),
):
    catalog = load_contract_catalog(PROJECT_ROOT)
    return build_predictive_gate_policy(
        allowed_forecast_object_types=allowed_forecast_object_types,
    ).to_manifest(catalog)


def _passed_predictive_test_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_name": "paired_predictive_test_result@1.0.0",
        "declared_test_id": "diebold_mariano_hln_v1",
        "actual_test_id": "diebold_mariano_hln_v1",
        "status": "passed",
        "promotion_allowed": True,
        "reason_codes": [],
        "mean_loss_differential": 0.5,
        "confidence_interval": [0.3, 0.8],
        "confidence_interval_method": "dm_hln_hac_t_interval",
        "p_value": 0.01,
        "practical_margin": 0.1,
        "raw_metric_comparison_role": "diagnostic_only",
        "statistical_test_backend": "diebold_mariano_hln_v1",
        "raw_pair_count": 80,
        "effective_sample_size": 80,
        "effective_block_count": 80,
        "minimum_pair_policy": {
            "minimum_effective_sample_size": 25,
            "minimum_effective_block_count": 8,
            "human_review_effective_sample_size": 50,
        },
    }
    body.update(overrides)
    return body


def test_resolver_rejects_thin_sample_even_when_legacy_boolean_allows() -> None:
    thin_sample_result = _passed_predictive_test_body(
        raw_pair_count=1,
        effective_sample_size=1,
        effective_block_count=1,
    )

    assert (
        resolve_confirmatory_promotion_allowed(
            candidate_beats_baseline=True,
            predictive_gate_policy_manifest=_predictive_gate_policy_manifest(),
            predictive_test_result_manifest=thin_sample_result,
        )
        is False
    )


def test_predictive_governance_reason_codes_preserve_paired_test_reasons() -> None:
    comparison_universe_body = {
        "baseline_id": "declared_baseline",
        "paired_comparison_records": [
            {
                "comparator_id": "declared_baseline",
                "paired_predictive_test_result": {
                    "reason_codes": [
                        "insufficient_paired_count",
                        "missing_practical_effect_margin",
                    ]
                },
            }
        ],
    }

    assert predictive_governance_reason_codes(
        evaluation_governance_body={"confirmatory_promotion_allowed": False},
        comparison_universe_body=comparison_universe_body,
    ) == (
        "insufficient_paired_count",
        "missing_practical_effect_margin",
    )
