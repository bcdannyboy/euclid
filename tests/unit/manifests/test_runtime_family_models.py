from __future__ import annotations

from pathlib import Path

from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.runtime_models import (
    ArtifactHashRecord,
    NullResultManifest,
    PerturbationFamilyResultManifest,
    PublicationRecordManifest,
    ReplayStageRecord,
    ReproducibilityBundleManifest,
    RobustnessReportManifest,
    RunResultManifest,
    SeedRecord,
    SensitivityAnalysisManifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _ref(schema_name: str, object_id: str) -> TypedRef:
    return TypedRef(schema_name=schema_name, object_id=object_id)


def test_run_result_manifest_candidate_roundtrip_omits_abstention_placeholder() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    model = RunResultManifest(
        object_id="demo_run_result",
        run_id="demo_run",
        scope_ledger_ref=_ref("scope_ledger_manifest@1.0.0", "scope"),
        search_plan_ref=_ref("search_plan_manifest@1.0.0", "search"),
        evaluation_plan_ref=_ref("evaluation_plan_manifest@1.1.0", "evaluation"),
        comparison_universe_ref=_ref(
            "comparison_universe_manifest@1.0.0", "comparison"
        ),
        evaluation_event_log_ref=_ref(
            "evaluation_event_log_manifest@1.0.0", "event_log"
        ),
        evaluation_governance_ref=_ref(
            "evaluation_governance_manifest@1.1.0", "governance"
        ),
        result_mode="candidate_publication",
        primary_reducer_artifact_ref=_ref(
            "reducer_artifact_manifest@1.0.0", "candidate"
        ),
        primary_scorecard_ref=_ref("scorecard_manifest@1.1.0", "scorecard"),
        primary_claim_card_ref=_ref("claim_card_manifest@1.1.0", "claim"),
        prediction_artifact_refs=(
            _ref("prediction_artifact_manifest@1.1.0", "prediction"),
        ),
        primary_external_evidence_ref=_ref(
            "external_evidence_manifest@1.0.0", "external_bundle"
        ),
        robustness_report_refs=(
            _ref("robustness_report_manifest@1.1.0", "robustness"),
        ),
        reproducibility_bundle_ref=_ref(
            "reproducibility_bundle_manifest@1.0.0", "bundle"
        ),
        deferred_scope_policy_refs=(),
    )

    manifest = model.to_manifest(catalog)
    restored = RunResultManifest.from_manifest(manifest)

    assert model.schema_family == "run_result_manifest"
    assert model.schema_version == "1.1.0"
    assert manifest.schema_name == "run_result_manifest@1.1.0"
    assert manifest.body["forecast_object_type"] == "point"
    assert "primary_abstention_ref" not in manifest.body
    assert manifest.body["primary_external_evidence_ref"] == {
        "schema_name": "external_evidence_manifest@1.0.0",
        "object_id": "external_bundle",
    }
    assert restored == model


def test_run_result_manifest_abstention_roundtrip_preserves_support_refs() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    model = RunResultManifest(
        object_id="demo_abstention_run_result",
        run_id="demo_abstention_run",
        scope_ledger_ref=_ref("scope_ledger_manifest@1.0.0", "scope"),
        search_plan_ref=_ref("search_plan_manifest@1.0.0", "search"),
        evaluation_plan_ref=_ref("evaluation_plan_manifest@1.1.0", "evaluation"),
        comparison_universe_ref=_ref(
            "comparison_universe_manifest@1.0.0", "comparison"
        ),
        evaluation_event_log_ref=_ref(
            "evaluation_event_log_manifest@1.0.0", "event_log"
        ),
        evaluation_governance_ref=_ref(
            "evaluation_governance_manifest@1.1.0", "governance"
        ),
        result_mode="abstention_only_publication",
        primary_abstention_ref=_ref("abstention_manifest@1.1.0", "abstention"),
        primary_validation_scope_ref=_ref(
            "validation_scope_manifest@1.0.0", "validation_scope"
        ),
        primary_external_evidence_ref=_ref(
            "external_evidence_manifest@1.0.0", "external_bundle"
        ),
        prediction_artifact_refs=(
            _ref("prediction_artifact_manifest@1.1.0", "prediction"),
        ),
        robustness_report_refs=(
            _ref("robustness_report_manifest@1.1.0", "robustness"),
        ),
        reproducibility_bundle_ref=_ref(
            "reproducibility_bundle_manifest@1.0.0", "bundle"
        ),
    )

    manifest = model.to_manifest(catalog)
    restored = RunResultManifest.from_manifest(manifest)

    assert manifest.body["primary_abstention_ref"] == {
        "schema_name": "abstention_manifest@1.1.0",
        "object_id": "abstention",
    }
    assert manifest.body["primary_external_evidence_ref"] == {
        "schema_name": "external_evidence_manifest@1.0.0",
        "object_id": "external_bundle",
    }
    assert "primary_reducer_artifact_ref" not in manifest.body
    assert restored == model


def test_bundle_and_publication_record_roundtrip_keep_required_refs() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    run_result_ref = _ref("run_result_manifest@1.1.0", "run_result")
    abstention_ref = _ref("abstention_manifest@1.1.0", "abstention")
    bundle = ReproducibilityBundleManifest(
        object_id="demo_bundle",
        bundle_id="demo_bundle",
        scope_id="euclid_v1_binding_scope@1.0.0",
        bundle_mode="abstention_only_publication",
        dataset_snapshot_ref=_ref("dataset_snapshot_manifest@1.0.0", "snapshot"),
        feature_view_ref=_ref("feature_view_manifest@1.0.0", "feature_view"),
        search_plan_ref=_ref("search_plan_manifest@1.0.0", "search"),
        evaluation_plan_ref=_ref("evaluation_plan_manifest@1.1.0", "evaluation"),
        comparison_universe_ref=_ref(
            "comparison_universe_manifest@1.0.0", "comparison"
        ),
        evaluation_event_log_ref=_ref(
            "evaluation_event_log_manifest@1.0.0", "event_log"
        ),
        evaluation_governance_ref=_ref(
            "evaluation_governance_manifest@1.1.0", "governance"
        ),
        run_result_ref=run_result_ref,
        required_manifest_refs=(abstention_ref,),
        artifact_hash_records=(
            ArtifactHashRecord(
                artifact_role="dataset_snapshot",
                sha256="sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            ),
        ),
        seed_records=(
            SeedRecord(seed_scope="search", seed_value="0"),
            SeedRecord(seed_scope="surrogate_generation", seed_value="0"),
            SeedRecord(seed_scope="perturbation", seed_value="0"),
        ),
        environment_metadata={
            "python_version": "3.11.0",
            "python_implementation": "CPython",
        },
        stage_order_records=(
            ReplayStageRecord(
                stage_id="dataset_snapshot_frozen",
                manifest_ref=_ref("dataset_snapshot_manifest@1.0.0", "snapshot"),
            ),
            ReplayStageRecord(
                stage_id="run_result_assembled",
                manifest_ref=run_result_ref,
            ),
        ),
        replay_verification_status="verified",
        failure_reason_codes=(),
    )
    publication = PublicationRecordManifest(
        object_id="demo_publication_record",
        publication_id="demo_publication_record",
        run_result_ref=run_result_ref,
        catalog_scope="public",
        publication_mode="abstention_only_publication",
        replay_verification_status="verified",
        comparator_exposure_status="not_applicable_abstention_only",
        reproducibility_bundle_ref=_ref(
            "reproducibility_bundle_manifest@1.0.0", "demo_bundle"
        ),
        readiness_judgment_ref=_ref("readiness_judgment_manifest@1.0.0", "readiness"),
        schema_lifecycle_integration_closure_ref=_ref(
            "schema_lifecycle_integration_closure_manifest@1.0.0", "closure"
        ),
        published_at="2026-04-12T00:00:00Z",
    )

    bundle_manifest = bundle.to_manifest(catalog)
    publication_manifest = publication.to_manifest(catalog)

    assert ReproducibilityBundleManifest.from_manifest(bundle_manifest) == bundle
    assert PublicationRecordManifest.from_manifest(publication_manifest) == publication
    assert bundle.lineage_refs == (
        bundle.dataset_snapshot_ref,
        bundle.feature_view_ref,
        bundle.search_plan_ref,
        bundle.evaluation_plan_ref,
        bundle.comparison_universe_ref,
        bundle.evaluation_event_log_ref,
        bundle.evaluation_governance_ref,
        bundle.run_result_ref,
        abstention_ref,
    )
    assert publication.lineage_refs == (
        publication.run_result_ref,
        publication.reproducibility_bundle_ref,
        publication.readiness_judgment_ref,
        publication.schema_lifecycle_integration_closure_ref,
    )


def test_robustness_report_roundtrip_preserves_machine_readable_detail() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    null_result = NullResultManifest(
        object_id="demo_null_result",
        null_result_id="demo_null_result",
        null_protocol_ref=_ref("null_protocol_manifest@1.1.0", "null"),
        candidate_id="candidate",
        status="null_not_rejected",
        failure_reason_code=None,
        observed_statistic=0.4,
        surrogate_statistics=(0.1, 0.2, 0.3),
        monte_carlo_p_value=0.4,
        max_p_value=0.5,
        resample_count=3,
    )
    perturbation_family_result = PerturbationFamilyResultManifest(
        object_id="demo_perturbation_family_result",
        perturbation_family_result_id="demo_perturbation_family_result",
        perturbation_protocol_ref=_ref(
            "perturbation_protocol_manifest@1.1.0", "perturb"
        ),
        candidate_id="candidate",
        family_id="recent_history_truncation",
        status="completed",
        valid_run_count=2,
        invalid_run_count=0,
        metric_results=(
            {
                "metric_ref": _ref(
                    "retention_metric_manifest@1.0.0", "match"
                ).as_dict(),
                "metric_id": "canonical_form_exact_match_rate",
                "applicability_status": "applicable",
                "value": 1.0,
                "failure_reason_code": None,
            },
        ),
        perturbation_runs=(
            {
                "perturbation_id": "history-50",
                "canonical_form_matches": True,
                "description_gain_bits": 0.5,
                "failure_reason_code": None,
                "metadata": {},
            },
        ),
    )
    sensitivity_analysis = SensitivityAnalysisManifest(
        object_id="demo_sensitivity_analysis",
        sensitivity_analysis_id="demo_sensitivity_analysis",
        perturbation_family_result_ref=perturbation_family_result.ref,
        candidate_id="candidate",
        family_id="recent_history_truncation",
        perturbation_id="history-50",
        canonical_form_matches=True,
        description_gain_bits=0.5,
        outer_candidate_score=0.4,
        outer_baseline_score=0.8,
        failure_reason_code=None,
        metadata={},
    )
    model = RobustnessReportManifest(
        object_id="demo_robustness_report_object",
        robustness_report_id="demo_robustness_report",
        null_protocol_ref=_ref("null_protocol_manifest@1.1.0", "null"),
        null_result_ref=null_result.ref,
        perturbation_protocol_ref=_ref(
            "perturbation_protocol_manifest@1.1.0", "perturb"
        ),
        perturbation_family_result_refs=(perturbation_family_result.ref,),
        leakage_canary_result_refs=(
            _ref("leakage_canary_result_manifest@1.0.0", "future"),
            _ref("leakage_canary_result_manifest@1.0.0", "late"),
            _ref("leakage_canary_result_manifest@1.0.0", "holdout"),
            _ref("leakage_canary_result_manifest@1.0.0", "revision"),
        ),
        sensitivity_analysis_refs=(sensitivity_analysis.ref,),
        status="failed",
        candidate_id="candidate",
        aggregate_metric_results=(
            {
                "metric_ref": _ref(
                    "retention_metric_manifest@1.0.0", "match"
                ).as_dict(),
                "metric_id": "canonical_form_exact_match_rate",
                "status": "computed",
                "value": 0.5,
                "failure_reason_code": None,
            },
        ),
        stability_status="passed",
        required_canary_type_coverage=(
            "future_target_level_feature",
            "late_available_target_copy",
            "holdout_membership_feature",
            "post_cutoff_revision_level_feature",
        ),
        final_robustness_status="failed",
        candidate_context={"frozen_candidate_set_id": "shortlist"},
    )

    manifest = model.to_manifest(catalog)
    restored = RobustnessReportManifest.from_manifest(manifest)

    assert manifest.body["candidate_id"] == "candidate"
    assert manifest.body["final_robustness_status"] == "failed"
    assert manifest.body["null_result_ref"] == {
        "schema_name": "null_result_manifest@1.0.0",
        "object_id": "demo_null_result",
    }
    assert manifest.body["perturbation_family_result_refs"] == [
        {
            "schema_name": "perturbation_family_result_manifest@1.0.0",
            "object_id": "demo_perturbation_family_result",
        }
    ]
    assert manifest.body["sensitivity_analysis_refs"] == [
        {
            "schema_name": "sensitivity_analysis_manifest@1.0.0",
            "object_id": "demo_sensitivity_analysis",
        }
    ]
    assert "null_result" not in manifest.body
    assert "perturbation_family_results" not in manifest.body
    assert "sensitivity_analyses" not in manifest.body
    assert restored == model
