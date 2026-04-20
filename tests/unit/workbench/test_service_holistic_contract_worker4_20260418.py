from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from euclid.workbench.service import normalize_analysis_payload

FIXTURE_DIR = (
    Path(__file__).resolve().parents[2] / "frontend" / "workbench" / "fixtures"
)

SPY_FIXTURE = "analysis-holistic-contract-worker4-spy-daily-return-20260418.json"
SPY_CSV = "analysis-holistic-contract-worker4-spy-daily-return-20260418.csv"
GLD_FIXTURE = "analysis-holistic-contract-worker4-gld-price-close-20260418.json"
GLD_CSV = "analysis-holistic-contract-worker4-gld-price-close-20260418.csv"
ABSTENTION_REASONS = [
    "robustness_failed",
    "perturbation_protocol_failed",
    "descriptive_only",
]
GLD_GAP_REPORT = [
    "operator_not_publishable",
    "no_backend_joint_claim",
    "probabilistic_evidence_thin",
    "requires_posthoc_symbolic_synthesis",
    "descriptive_only",
]
DESCRIPTIVE_SELECTION_SCOPE = "shared_planning_cir_only"
DESCRIPTIVE_SELECTION_RULE = (
    "min_total_code_bits_then_max_description_gain_then_min_structure_code_bits_"
    "then_min_canonical_byte_length_then_candidate_id"
)


def _load_fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _bind_dataset_csv(
    payload: dict[str, object],
    *,
    csv_name: str,
) -> dict[str, object]:
    analysis = deepcopy(payload)
    dataset = dict(analysis["dataset"])
    dataset["dataset_csv"] = str(FIXTURE_DIR / csv_name)
    analysis["dataset"] = dataset
    return analysis


def _assert_descriptive_fit_contract(
    descriptive_fit: dict[str, object],
    *,
    expected_source: str,
    expected_submitter_id: str,
    expected_candidate_id: str | None,
    expected_family_id: str,
    expected_metrics: dict[str, object],
    expected_law_eligible: bool,
    expected_law_rejection_reason_codes: list[str],
    expected_submitter_class: str | None = None,
) -> None:
    assert descriptive_fit["status"] == "completed"
    assert descriptive_fit["source"] == expected_source
    assert descriptive_fit["submitter_id"] == expected_submitter_id
    if expected_submitter_class is None:
        assert descriptive_fit.get("submitter_class") in (None, "")
    else:
        assert descriptive_fit["submitter_class"] == expected_submitter_class
    assert descriptive_fit["candidate_id"] == expected_candidate_id
    assert descriptive_fit["family_id"] == expected_family_id
    assert descriptive_fit["metrics"] == expected_metrics
    assert descriptive_fit["honesty_note"]
    assert descriptive_fit["selection_scope"] == DESCRIPTIVE_SELECTION_SCOPE
    assert descriptive_fit["selection_rule"] == DESCRIPTIVE_SELECTION_RULE
    assert descriptive_fit["law_eligible"] is expected_law_eligible
    assert (
        descriptive_fit["law_rejection_reason_codes"]
        == expected_law_rejection_reason_codes
    )
    assert descriptive_fit["claim_class"] == "descriptive_fit"
    assert descriptive_fit["is_law_claim"] is False
    assert descriptive_fit["equation"]["label"]
    assert descriptive_fit["equation"].get("family_id") == expected_family_id
    assert descriptive_fit["equation"].get("candidate_id") == expected_candidate_id
    assert descriptive_fit["chart"]["actual_series"]
    assert descriptive_fit["chart"]["equation_curve"]
    audit = descriptive_fit["semantic_audit"]
    assert audit["headline"]
    assert audit["summary"]
    assert "recommended_target_id" in audit
    assert "recommended_target_label" in audit
    assert "recommended_target_reason" in audit


def test_spy_fixture_rejects_exact_closure_as_top_line_claim_worker4_20260418() -> None:
    payload = _bind_dataset_csv(_load_fixture(SPY_FIXTURE), csv_name=SPY_CSV)
    payload["holistic_equation"] = {
        "status": "completed",
        "source": "operator_point",
        "exactness": "sample_exact_closure",
        "equation": {
            "label": (
                "y(t) = (0.000072 + 0.000001*t) + exact_closure(sample)"
            )
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["operator_point"]["publication"]["status"] == "abstained"
    assert normalized["operator_point"]["result_mode"] == "abstention_only_publication"
    assert normalized.get("claim_class") is None
    assert normalized.get("holistic_equation") is None
    assert normalized.get("predictive_law") is None
    assert normalized.get("would_have_abstained_because") == ABSTENTION_REASONS
    assert "requires_exact_sample_closure" in (normalized.get("gap_report") or [])
    assert normalized.get("not_holistic_because") == normalized.get("gap_report")


def test_spy_fixture_rejects_legacy_synthetic_holistic_metadata_worker4_20260418() -> (
    None
):
    payload = _bind_dataset_csv(_load_fixture(SPY_FIXTURE), csv_name=SPY_CSV)
    payload["claim_class"] = "holistic_equation"
    payload["publishable"] = True
    payload["gap_report"] = []
    payload["not_holistic_because"] = []
    payload["would_have_abstained_because"] = []
    payload["operator_point"] = {
        **payload["operator_point"],
        "result_mode": "candidate_publication",
        "publication": {
            "status": "publishable",
            "headline": "Saved analysis still carried a completed holistic payload.",
            "reason_codes": [],
        },
        "claim_card_ref": {
            "schema_name": "claim_card",
            "object_id": "spy_joint_claim",
        },
        "scorecard_ref": {
            "schema_name": "scorecard",
            "object_id": "spy_joint_claim",
        },
        "validation_scope_ref": {
            "schema_name": "validation_scope",
            "object_id": "spy_joint_claim",
        },
        "publication_record_ref": {
            "schema_name": "publication_record",
            "object_id": "spy_joint_claim",
        },
        "scorecard": {
            "descriptive_status": "passed",
            "predictive_status": "passed",
        },
        "claim_card": {
            "claim_type": "point_forecast",
            "claim_ceiling": "predictively_supported",
            "predictive_support_status": "confirmatory_supported",
            "allowed_interpretation_codes": [
                "point_forecast_within_declared_validation_scope"
            ],
        },
    }
    payload["operator_point"].pop("abstention", None)
    payload["probabilistic"] = {
        "distribution": {
            "status": "completed",
            "selected_family": "analytic",
            "replay_verification": "verified",
            "validation_scope_ref": {
                "schema_name": "validation_scope",
                "object_id": "spy_joint_claim",
            },
            "publication_record_ref": {
                "schema_name": "publication_record",
                "object_id": "spy_joint_claim",
            },
            "evidence": {
                "strength": "strong",
                "headline": "Strong calibration support within the declared scope.",
                "sample_size": 12,
                "origin_count": 12,
                "horizon_count": 1,
            },
            "calibration": {
                "status": "passed",
                "passed": True,
                "gate_effect": "publishable",
            },
            "equation": {
                "label": "y(t) = 0.0002 + 0.0005*t",
                "family_id": "analytic",
                "curve": payload["operator_point"]["chart"]["equation_curve"],
            },
            "latest_row": {
                "origin_time": "2026-04-15T00:00:00Z",
                "available_at": "2026-04-16T21:00:00Z",
                "horizon": 1,
                "location": 0.0025,
                "scale": 0.0035,
                "realized_observation": 0.0045,
            },
            "rows": [
                {
                    "origin_time": f"2026-04-{day:02d}T00:00:00Z",
                    "available_at": f"2026-04-{day + 1:02d}T21:00:00Z",
                    "horizon": 1,
                    "location": 0.001 + 0.0001 * index,
                    "scale": 0.0034,
                    "realized_observation": 0.0015 if index % 2 == 0 else -0.0008,
                }
                for index, day in enumerate(range(4, 16))
            ],
            "chart": {
                "forecast_bands": [
                    {
                        "origin_time": "2026-04-15T00:00:00Z",
                        "available_at": "2026-04-16T21:00:00Z",
                        "center": 0.0025,
                        "lower": -0.001,
                        "upper": 0.006,
                        "realized_observation": 0.0045,
                    }
                ]
            },
        }
    }
    payload["holistic_equation"] = {
        "status": "completed",
        "claim_class": "holistic_equation",
        "exactness": "joint_validation_scope_claim",
        "deterministic_source": "operator_point",
        "probabilistic_source": "distribution",
        "validation_scope_ref": "validation_scope:spy_joint_claim",
        "publication_record_ref": "publication_record:spy_joint_claim",
        "mode": "composition_stochastic",
        "composition_operator": "additive_residual",
        "selected_probabilistic_lane": "distribution",
        "equation": {
            "label": "y(t) = compact_drift + epsilon_bridge",
            "curve": payload["operator_point"]["chart"]["equation_curve"],
        },
        "honesty_note": (
            "Stale saved analysis carried a fully populated holistic payload with "
            "legacy synthetic composition metadata."
        ),
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["operator_point"]["publication"]["status"] == "publishable"
    assert normalized.get("would_have_abstained_because") == []
    assert normalized.get("claim_class") == "predictive_law"
    assert normalized.get("publishable") is True
    assert normalized.get("holistic_equation") is None
    predictive_law = normalized.get("predictive_law")
    assert isinstance(predictive_law, dict)
    assert predictive_law.get("claim_class") == "predictive_law"
    assert predictive_law.get("validation_scope_ref") == "validation_scope:spy_joint_claim"
    assert predictive_law.get("publication_record_ref") == (
        "publication_record:spy_joint_claim"
    )
    assert normalized.get("gap_report") == [
        "no_backend_joint_claim",
        "requires_posthoc_symbolic_synthesis",
    ]
    assert normalized.get("not_holistic_because") == normalized.get("gap_report")


def test_gld_fixture_keeps_abstained_operator_out_of_holistic_claims_worker4_20260418() -> None:
    normalized = normalize_analysis_payload(
        _bind_dataset_csv(_load_fixture(GLD_FIXTURE), csv_name=GLD_CSV)
    )

    descriptive_fit = normalized["descriptive_fit"]
    probabilistic = normalized["probabilistic"]

    assert normalized["operator_point"]["publication"]["status"] == "abstained"
    assert normalized["operator_point"]["abstention"]["blocked_ceiling"] == (
        "descriptive_only"
    )
    assert set(probabilistic) >= {
        "distribution",
        "quantile",
        "interval",
        "event_probability",
    }
    assert all(
        probabilistic[lane_kind]["status"] == "completed"
        for lane_kind in ("distribution", "quantile", "interval", "event_probability")
    )
    _assert_descriptive_fit_contract(
        descriptive_fit,
        expected_source="benchmark_local_selection",
        expected_submitter_id="portfolio_orchestrator",
        expected_candidate_id="analytic_lag1_affine",
        expected_family_id="analytic",
        expected_metrics={"total_code_bits": 6970.0},
        expected_law_eligible=True,
        expected_law_rejection_reason_codes=[],
    )
    assert descriptive_fit["semantic_audit"]["classification"] == (
        "near_persistence"
    )
    distribution_row = probabilistic["distribution"]["latest_row"]
    assert distribution_row["realized_observation"] == descriptive_fit["chart"][
        "actual_series"
    ][-1]["observed_value"]
    assert abs(
        float(distribution_row["location"])
        - float(descriptive_fit["chart"]["equation_curve"][-1]["fitted_value"])
    ) < 2.0
    assert normalized.get("stochastic_label") is None
    assert normalized.get("claim_class") == "descriptive_fit"
    assert normalized.get("holistic_equation") is None
    assert not isinstance(normalized.get("holistic_equation"), dict)
    assert normalized.get("predictive_law") is None
    assert normalized.get("would_have_abstained_because") == ABSTENTION_REASONS
    assert normalized.get("gap_report") == GLD_GAP_REPORT
    assert normalized.get("not_holistic_because")


def test_legacy_gld_fixture_projects_descriptive_fallback_worker4_20260418() -> None:
    payload = _bind_dataset_csv(_load_fixture(GLD_FIXTURE), csv_name=GLD_CSV)
    payload.pop("descriptive_fit", None)
    payload.pop("holistic_equation", None)
    payload["benchmark"]["descriptive_fit_status"] = {
        "status": "candidate_available_but_not_loaded",
        "headline": (
            "Saved analysis predates descriptive-bank payloads; project a "
            "descriptive fallback instead of leaving the lane empty."
        ),
        "reason_codes": [],
    }
    current_payload = deepcopy(payload)
    current_payload["analysis_version"] = "1.0.0"
    current_normalized = normalize_analysis_payload(current_payload)

    assert current_normalized.get("descriptive_fit") is None
    assert current_normalized.get("claim_class") is None

    payload["analysis_version"] = "0.9.0"

    normalized = normalize_analysis_payload(payload)

    descriptive_fit = normalized["descriptive_fit"]

    _assert_descriptive_fit_contract(
        descriptive_fit,
        expected_source="legacy_operator_point_fallback",
        expected_submitter_id="legacy_operator_point_fallback",
        expected_submitter_class="legacy_compatibility_projection",
        expected_candidate_id=None,
        expected_family_id="drift",
        expected_metrics={},
        expected_law_eligible=False,
        expected_law_rejection_reason_codes=["legacy_compatibility_projection"],
    )
    assert descriptive_fit["semantic_audit"]["classification"] == (
        "level_fit"
    )
    assert (
        descriptive_fit["equation"]["label"]
        == normalized["operator_point"]["equation"]["label"]
    )
    assert normalized.get("claim_class") == "descriptive_fit"
    assert normalized.get("gap_report") == GLD_GAP_REPORT
    assert normalized.get("not_holistic_because") == normalized.get("gap_report")
    assert normalized.get("holistic_equation") is None
    assert normalized.get("predictive_law") is None
    assert normalized.get("would_have_abstained_because") == ABSTENTION_REASONS
