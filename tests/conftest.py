from __future__ import annotations

import shutil
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def contract_catalog():
    from euclid.contracts.loader import load_contract_catalog

    return load_contract_catalog(PROJECT_ROOT)


@pytest.fixture(scope="session")
def phase01_demo_manifest(project_root: Path) -> Path:
    return project_root / "fixtures/runtime/prototype-demo.yaml"


@pytest.fixture(scope="session")
def phase03_runtime_fixture_dir(project_root: Path) -> Path:
    return project_root / "fixtures/runtime/phase03"


@pytest.fixture(scope="session")
def phase04_runtime_fixture_dir(project_root: Path) -> Path:
    return project_root / "fixtures/runtime/phase04"


@pytest.fixture(scope="session")
def phase01_demo_output_template(
    tmp_path_factory: pytest.TempPathFactory,
    phase01_demo_manifest: Path,
) -> Path:
    import euclid

    output_root = tmp_path_factory.mktemp("phase01-demo-output")
    euclid.run_demo(
        manifest_path=phase01_demo_manifest,
        output_root=output_root,
    )
    return output_root


@pytest.fixture
def phase01_demo_output_root(
    tmp_path: Path,
    phase01_demo_output_template: Path,
) -> Path:
    output_root = tmp_path / "phase01-demo-output"
    shutil.copytree(phase01_demo_output_template, output_root)
    return output_root


def _build_phase07_candidate_demo_output(
    output_root: Path, manifest_path: Path
) -> None:
    import euclid
    import euclid.prototype.workflow as workflow
    from euclid.contracts.refs import TypedRef
    from euclid.modules.robustness import (
        PerturbationRunRecord,
        build_null_result_manifest,
        build_perturbation_family_result_manifest,
        build_robustness_report,
        build_sensitivity_analysis_manifest,
        evaluate_aggregate_metric_results,
        evaluate_null_comparison,
        evaluate_perturbation_family,
    )

    original_materialize_robustness = workflow._materialize_robustness_artifacts

    def _typed_ref(payload: object) -> TypedRef:
        assert isinstance(payload, dict)
        return TypedRef(
            schema_name=str(payload["schema_name"]),
            object_id=str(payload["object_id"]),
        )

    def _patched_materialize_robustness(
        *,
        catalog,
        registry,
        intake,
        feature_rows,
        folds,
        baseline_registry,
        point_score_policy,
        frozen_shortlist,
        selected_candidate_runtime,
        minimum_description_gain_bits,
    ):
        artifacts = original_materialize_robustness(
            catalog=catalog,
            registry=registry,
            intake=intake,
            feature_rows=feature_rows,
            folds=folds,
            baseline_registry=baseline_registry,
            point_score_policy=point_score_policy,
            frozen_shortlist=frozen_shortlist,
            selected_candidate_runtime=selected_candidate_runtime,
            minimum_description_gain_bits=minimum_description_gain_bits,
        )
        report_body = artifacts.robustness_report.manifest.body
        metric_refs_by_id = {
            str(item["metric_id"]): _typed_ref(item["metric_ref"])
            for item in report_body["aggregate_metric_results"]
        }
        passing_family_results = (
            evaluate_perturbation_family(
                family_id="recent_history_truncation",
                metric_refs_by_id=metric_refs_by_id,
                runs=(
                    PerturbationRunRecord(
                        perturbation_id="recent_history_truncation_0.5",
                        canonical_form_matches=True,
                        description_gain_bits=max(
                            0.1,
                            float(selected_candidate_runtime.description_gain_bits),
                        ),
                        outer_candidate_score=max(
                            0.0,
                            float(selected_candidate_runtime.exploratory_primary_score)
                            - 0.1,
                        ),
                        outer_baseline_score=float(
                            selected_candidate_runtime.baseline_primary_score
                        ),
                    ),
                    PerturbationRunRecord(
                        perturbation_id="recent_history_truncation_0.75",
                        canonical_form_matches=True,
                        description_gain_bits=max(
                            0.1,
                            float(selected_candidate_runtime.description_gain_bits),
                        ),
                        outer_candidate_score=max(
                            0.0,
                            float(selected_candidate_runtime.exploratory_primary_score)
                            - 0.05,
                        ),
                        outer_baseline_score=float(
                            selected_candidate_runtime.baseline_primary_score
                        ),
                    ),
                ),
            ),
            evaluate_perturbation_family(
                family_id="quantization_coarsening",
                metric_refs_by_id=metric_refs_by_id,
                runs=(
                    PerturbationRunRecord(
                        perturbation_id="quantization_coarsening_2x",
                        canonical_form_matches=True,
                        description_gain_bits=max(
                            0.1,
                            float(selected_candidate_runtime.description_gain_bits),
                        ),
                        outer_candidate_score=max(
                            0.0,
                            float(selected_candidate_runtime.exploratory_primary_score)
                            - 0.1,
                        ),
                        outer_baseline_score=float(
                            selected_candidate_runtime.baseline_primary_score
                        ),
                    ),
                    PerturbationRunRecord(
                        perturbation_id="quantization_coarsening_4x",
                        canonical_form_matches=True,
                        description_gain_bits=max(
                            0.1,
                            float(selected_candidate_runtime.description_gain_bits),
                        ),
                        outer_candidate_score=max(
                            0.0,
                            float(selected_candidate_runtime.exploratory_primary_score)
                            - 0.05,
                        ),
                        outer_baseline_score=float(
                            selected_candidate_runtime.baseline_primary_score
                        ),
                    ),
                ),
            ),
        )
        aggregate_metric_results, stability_status = evaluate_aggregate_metric_results(
            family_results=passing_family_results,
            required_metric_refs=tuple(metric_refs_by_id.values()),
            metric_thresholds={
                ref.object_id: 0.5 for ref in metric_refs_by_id.values()
            },
        )
        null_result = evaluate_null_comparison(
            observed_statistic=max(
                0.1,
                float(selected_candidate_runtime.description_gain_bits),
            ),
            surrogate_statistics=(0.0, 0.0, 0.1, 0.2),
            max_p_value=0.25,
        )
        null_result_manifest = registry.register(
            build_null_result_manifest(
                catalog,
                null_result_id="phase07_candidate_publication_null_result_v1",
                null_protocol_ref=artifacts.null_protocol.manifest.ref,
                candidate_id=str(report_body["candidate_id"]),
                evaluation=null_result,
            ).to_manifest(catalog),
        )
        perturbation_family_manifests = tuple(
            registry.register(
                build_perturbation_family_result_manifest(
                    catalog,
                    perturbation_family_result_id=(
                        f"phase07_{evaluation.family_id}_perturbation_result_v1"
                    ),
                    perturbation_protocol_ref=artifacts.perturbation_protocol.manifest.ref,
                    candidate_id=str(report_body["candidate_id"]),
                    evaluation=evaluation,
                ).to_manifest(catalog),
            )
            for evaluation in passing_family_results
        )
        perturbation_family_refs_by_id = {
            str(item.manifest.body["family_id"]): item.manifest.ref
            for item in perturbation_family_manifests
        }
        sensitivity_analyses = tuple(
            {
                "analysis_id": run["perturbation_id"],
                "family_id": str(item.manifest.body["family_id"]),
                "perturbation_id": run["perturbation_id"],
                "canonical_form_matches": run["canonical_form_matches"],
                "description_gain_bits": run["description_gain_bits"],
                "outer_candidate_score": run.get("outer_candidate_score"),
                "outer_baseline_score": run.get("outer_baseline_score"),
                "failure_reason_code": run.get("failure_reason_code"),
                "metadata": dict(run.get("metadata", {})),
            }
            for item in perturbation_family_manifests
            for run in item.manifest.body["perturbation_runs"]
        )
        sensitivity_analysis_manifests = tuple(
            registry.register(
                build_sensitivity_analysis_manifest(
                    catalog,
                    sensitivity_analysis_id=(
                        f"phase07_{analysis['family_id']}_{analysis['analysis_id']}"
                    ),
                    perturbation_family_result_ref=perturbation_family_refs_by_id[
                        str(analysis["family_id"])
                    ],
                    candidate_id=str(report_body["candidate_id"]),
                    analysis=analysis,
                ).to_manifest(catalog),
            )
            for analysis in sensitivity_analyses
        )
        passing_report = registry.register(
            build_robustness_report(
                candidate_id=str(report_body["candidate_id"]),
                null_protocol_ref=artifacts.null_protocol.manifest.ref,
                null_result=null_result,
                null_result_ref=null_result_manifest.manifest.ref,
                perturbation_protocol_ref=artifacts.perturbation_protocol.manifest.ref,
                perturbation_family_results=passing_family_results,
                perturbation_family_result_refs=tuple(
                    item.manifest.ref for item in perturbation_family_manifests
                ),
                aggregate_metric_results=aggregate_metric_results,
                stability_status=stability_status,
                leakage_canary_result_refs=tuple(
                    item.manifest.ref for item in artifacts.leakage_canary_results
                ),
                leakage_canary_results=tuple(
                    item.manifest.body for item in artifacts.leakage_canary_results
                ),
                candidate_context=dict(report_body.get("candidate_context", {})),
                sensitivity_analysis_refs=tuple(
                    item.manifest.ref for item in sensitivity_analysis_manifests
                ),
                report_id="phase07_candidate_publication_robustness_report_v1",
            ).to_manifest(catalog),
        )
        return replace(artifacts, robustness_report=passing_report)

    with patch.object(
        workflow,
        "_materialize_robustness_artifacts",
        _patched_materialize_robustness,
    ):
        euclid.run_demo(
            manifest_path=manifest_path,
            output_root=output_root,
        )


@pytest.fixture(scope="session")
def phase07_candidate_demo_output_template(
    tmp_path_factory: pytest.TempPathFactory,
    phase01_demo_manifest: Path,
) -> Path:
    output_root = tmp_path_factory.mktemp("phase07-candidate-demo-output")
    _build_phase07_candidate_demo_output(output_root, phase01_demo_manifest)
    return output_root


@pytest.fixture
def phase07_candidate_demo_output_root(
    tmp_path: Path,
    phase07_candidate_demo_output_template: Path,
) -> Path:
    output_root = tmp_path / "phase07-candidate-demo-output"
    shutil.copytree(phase07_candidate_demo_output_template, output_root)
    return output_root
