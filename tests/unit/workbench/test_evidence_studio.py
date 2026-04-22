from __future__ import annotations

from euclid.workbench.service import normalize_analysis_payload


def test_normalize_analysis_payload_builds_sanitized_evidence_studio_surface() -> None:
    payload = {
        "workspace_root": "/tmp/workbench-run",
        "analysis_path": "/tmp/workbench-run/analysis.json",
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "daily_return"},
        },
        "operator_point": {
            "status": "completed",
            "result_mode": "abstention_only_publication",
            "candidate_id": "analytic_lag1_affine",
            "selected_family": "analytic",
            "manifest_path": "/tmp/workbench-run/operator-point.yaml",
            "replay_ref": {"schema_name": "replay_bundle", "object_id": "point-run"},
            "engine_provenance": {
                "engine_id": "bounded_symbolic_search",
                "engine_version": "fixture",
            },
            "publication": {
                "status": "abstained",
                "reason_codes": ["robustness_failed"],
            },
            "abstention": {
                "blocked_ceiling": "descriptive_structure",
                "reason_codes": ["robustness_failed"],
            },
            "claim_card": {
                "claim_type": "point_forecast",
                "claim_ceiling": "descriptive_structure",
                "predictive_support_status": "unsupported",
            },
            "scorecard": {
                "descriptive_status": "passed",
                "predictive_status": "failed",
                "falsification": {"status": "failed", "reason_codes": ["stress"]},
                "residual_diagnostics": {"status": "failed", "max_abs_z": 3.2},
            },
        },
        "benchmark": {
            "portfolio_selection": {
                "selection_explanation": "analytic_backend selected by code length"
            },
            "submitters": [
                {
                    "submitter_id": "analytic_backend",
                    "status": "selected",
                    "selected_candidate_id": "analytic_lag1_affine",
                    "replay_contract": {
                        "selection_scope": "shared_planning_cir_only"
                    },
                }
            ],
        },
        "descriptive_fit": {
            "status": "completed",
            "source": "benchmark_local_selection",
            "submitter_id": "analytic_backend",
            "candidate_id": "analytic_lag1_affine",
            "family_id": "analytic",
            "metrics": {"total_code_bits": 12.0, "description_gain_bits": 4.0},
            "law_rejection_reason_codes": ["descriptive_structure"],
            "equation": {
                "candidate_id": "analytic_lag1_affine",
                "family_id": "analytic",
                "parameter_summary": {"intercept": 0.0, "lag_coefficient": 0.5},
                "curve": [],
            },
        },
        "probabilistic": {
            "distribution": {
                "status": "completed",
                "replay_verification": "verified",
                "evidence": {"strength": "thin", "sample_size": 8},
                "calibration": {
                    "status": "failed",
                    "diagnostics": [
                        {"diagnostic_id": "pit_or_randomized_pit_uniformity"}
                    ],
                },
            }
        },
        "live_api_evidence": {
            "evidence_kind": "live_api_gate",
            "provider": "fmp",
            "endpoint_class": "historical-price-eod",
            "status": "passed",
            "FMP_API_KEY": "fmp-workbench-secret",
            "semantic_checks": {
                "schema_valid": True,
                "claim_published": False,
                "request_url": (
                    "https://example.test/history?apikey=fmp-workbench-secret"
                ),
                "headers": {
                    "Authorization": "Bearer fmp-workbench-secret",
                },
                "message": "provider accepted fmp-workbench-secret",
            },
        },
    }

    normalized = normalize_analysis_payload(payload)

    rendered = repr(normalized["evidence_studio"])
    assert "fmp-workbench-secret" not in rendered
    studio = normalized["evidence_studio"]
    assert studio["redaction_status"] == "sanitized"
    assert studio["claim_surface"]["claim_class"] == "descriptive_fit"
    assert studio["claim_surface"]["claim_lane"] == "descriptive"
    assert studio["claim_surface"]["claim_ceiling"] == "descriptive_structure"
    assert studio["claim_surface"]["publication_status"] == "abstained"
    assert studio["claim_surface"]["abstention_reason_codes"] == [
        "robustness_failed",
        "descriptive_structure",
    ]
    assert "operator_not_publishable" in studio["claim_surface"]["downgrade_reason_codes"]
    assert studio["live_evidence"]["claim_boundary"] == {
        "counts_as_scientific_claim_evidence": False,
        "reason_code": "live_api_gate_validates_access_and_payload_shape_only",
        "status": "non_claim_evidence",
    }
    assert studio["live_evidence"]["sanitized_evidence"]["semantic_checks"][
        "claim_published"
    ] is False
    assert studio["replay_artifacts"]["links"]
    assert {
        "section": "operator_point",
        "role": "manifest_path",
        "value": "/tmp/workbench-run/operator-point.yaml",
    } in studio["replay_artifacts"]["links"]
    assert studio["engine_provenance"]["point_lane"]["engine_id"] == (
        "bounded_symbolic_search"
    )
    assert studio["diagnostics"]["fitting"]["metrics"]["total_code_bits"] == 12.0
    assert studio["diagnostics"]["scoring"]["scorecard"]["predictive_status"] == (
        "failed"
    )
    assert studio["diagnostics"]["falsification"]["gap_report"]


def test_normalize_analysis_payload_marks_missing_and_malformed_live_evidence() -> None:
    base_payload = {
        "dataset": {"symbol": "SPY", "target": {"id": "daily_return"}},
        "operator_point": {
            "status": "completed",
            "result_mode": "candidate_publication",
            "publication": {"status": "candidate_only", "reason_codes": []},
        },
    }

    missing = normalize_analysis_payload(base_payload)
    malformed = normalize_analysis_payload(
        {**base_payload, "live_api_evidence": "not a mapping"}
    )

    assert missing["evidence_studio"]["live_evidence"]["status"] == "unavailable"
    assert missing["evidence_studio"]["live_evidence"]["reason_codes"] == [
        "live_evidence_missing"
    ]
    assert malformed["evidence_studio"]["live_evidence"]["status"] == "malformed"
    assert malformed["evidence_studio"]["live_evidence"]["reason_codes"] == [
        "live_evidence_malformed"
    ]
    assert (
        malformed["evidence_studio"]["live_evidence"]["claim_boundary"][
            "counts_as_scientific_claim_evidence"
        ]
        is False
    )
