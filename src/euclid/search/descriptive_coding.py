from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any, Mapping, Sequence

import numpy as np

from euclid.algorithmic_dsl import (
    evaluate_algorithmic_program,
    parse_algorithmic_program,
)
from euclid.cir.models import CandidateIntermediateRepresentation
from euclid.contracts.errors import ContractValidationError
from euclid.math.codelength import (
    CodelengthComparisonKey,
    coding_row_set_id,
    data_code_diagnostics,
    data_code_length,
    signed_integer_code_length,
    strict_single_class_law_eligibility,
)
from euclid.math.lattice import LatticePolicy
from euclid.math.quantization import FixedStepMidTreadQuantizer
from euclid.math.quantization import QuantizationPolicy, resolve_quantizer
from euclid.math.reference_descriptions import (
    ReferenceDescriptionPolicy,
    build_reference_description,
)
from euclid.modules.features import FeatureView
from euclid.reducers.composition import (
    AdditiveResidualComposition,
    PiecewiseComposition,
    RegimeConditionedComposition,
    composition_runtime_signature,
    extract_component_mapping,
    resolve_regime_weights,
    select_piecewise_segment,
)
from euclid.reducers.models import BoundObservationModel

_DEFAULT_QUANTIZATION_STEP = "0.5"


@dataclass(frozen=True)
class DescriptionGainArtifact:
    candidate_id: str
    primitive_family: str
    candidate_hash: str
    L_family_bits: float
    L_structure_bits: float
    L_literals_bits: float
    L_params_bits: float
    L_state_bits: float
    L_data_bits: float
    L_total_bits: float
    reference_bits: float
    description_gain_bits: float
    quantization_mode: str = "fixed_step_mid_tread"
    quantization_step: str = _DEFAULT_QUANTIZATION_STEP
    reference_policy_id: str = "raw_quantized_transformed_sequence_v1"
    reference_family_id: str = "raw_quantized_transformed_sequence"
    reference_scope: str = "raw_observation_reference"
    data_code_family: str = "residual_signed_integer_elias_delta_v1"
    coding_row_set_id: str = ""
    codelength_comparison_key: Mapping[str, Any] = field(default_factory=dict)
    data_code_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    lattice_policy: Mapping[str, Any] = field(default_factory=dict)
    model_code_decomposition: Mapping[str, Any] = field(default_factory=dict)
    comparable_group_id: str = ""
    comparable_group_size: int = 0
    comparable_group_rank: int = 0
    comparable_group_selected: bool = False


@dataclass(frozen=True)
class DescriptiveAdmissibilityDiagnostic:
    candidate_id: str
    primitive_family: str
    candidate_hash: str
    is_admissible: bool
    codelength_comparability: bool
    support_valid: bool
    reason_codes: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DescriptiveCodingResult:
    accepted_candidates: tuple[CandidateIntermediateRepresentation, ...]
    description_artifacts: tuple[DescriptionGainArtifact, ...]
    admissibility_diagnostics: tuple[DescriptiveAdmissibilityDiagnostic, ...]
    selected_candidates: tuple[CandidateIntermediateRepresentation, ...] = ()


@dataclass(frozen=True)
class _ColumnarFeatureRows:
    target_values: np.ndarray
    lag_1_values: np.ndarray | None
    row_count: int


def evaluate_descriptive_candidates(
    candidates: Sequence[CandidateIntermediateRepresentation],
    *,
    feature_view: FeatureView,
    quantizer: FixedStepMidTreadQuantizer | None = None,
    quantization_policy: QuantizationPolicy | None = None,
    reference_policy: ReferenceDescriptionPolicy | Mapping[str, Any] | None = None,
    data_code_family: str = "residual_signed_integer_elias_delta_v1",
    seasonal_period: int | None = None,
    horizon_geometry: Sequence[int] = (1,),
    residual_history_construction: str = "none",
    parameter_lattice_step: str | None = None,
    state_lattice_step: str | None = None,
    lattice_policy: LatticePolicy | Mapping[str, Any] | None = None,
    comparison_key_overrides: Mapping[str, CodelengthComparisonKey] | None = None,
    minimum_description_gain_bits: float | None = None,
) -> DescriptiveCodingResult:
    legal_feature_view = feature_view.require_stage_reuse("search")
    columnar_rows = _columnar_feature_rows(legal_feature_view.rows)
    observed_values = tuple(
        float(value) for value in columnar_rows.target_values.tolist()
    )
    if quantizer is None:
        resolved_quantization = resolve_quantizer(
            quantization_policy
            or QuantizationPolicy(
                quantization_mode="fixed_step_mid_tread",
                quantization_step=_DEFAULT_QUANTIZATION_STEP,
            ),
            observed_values=observed_values,
        )
        active_quantizer = resolved_quantization.quantizer
        quantization_mode = resolved_quantization.quantization_mode
    else:
        active_quantizer = quantizer
        quantization_mode = "fixed_step_mid_tread"
    active_reference_policy = _coerce_reference_policy(reference_policy)
    reference_description = build_reference_description(
        observed_values,
        quantizer=active_quantizer,
        policy=active_reference_policy,
        seasonal_period=seasonal_period,
        data_code_family=data_code_family,
    )
    row_set_id = coding_row_set_id(tuple(legal_feature_view.rows))
    active_lattice_policy = LatticePolicy.coerce(
        lattice_policy,
        residual_quantization_step=active_quantizer.step_string,
        parameter_lattice_step=parameter_lattice_step,
        state_lattice_step=state_lattice_step,
    )
    comparison_status = _comparison_status(
        candidates,
        quantization_mode=quantization_mode,
        quantizer=active_quantizer,
        reference_policy_id=reference_description.reference_policy_id,
        reference_family_id=reference_description.reference_family_id,
        reference_scope=reference_description.reference_scope,
        data_code_family=data_code_family,
        horizon_geometry=tuple(int(horizon) for horizon in horizon_geometry),
        coding_row_set_id=row_set_id,
        residual_history_construction=residual_history_construction,
        parameter_lattice_step=active_lattice_policy.parameter_lattice_step,
        state_lattice_step=active_lattice_policy.state_lattice_step,
        comparison_key_overrides=comparison_key_overrides or {},
    )
    minimum_gain = (
        0.0 if minimum_description_gain_bits is None else minimum_description_gain_bits
    )

    accepted_candidates: list[CandidateIntermediateRepresentation] = []
    description_artifacts: list[DescriptionGainArtifact] = []
    diagnostics: list[DescriptiveAdmissibilityDiagnostic] = []
    candidate_by_hash = {
        candidate.canonical_hash(): candidate for candidate in candidates
    }

    for candidate in candidates:
        candidate_id = (
            candidate.evidence_layer.backend_origin_record.source_candidate_id
        )
        primitive_family = candidate.structural_layer.cir_family_id
        candidate_hash = candidate.canonical_hash()
        reason_codes: list[str] = []
        details: dict[str, Any] = {}
        comparable = candidate_hash in comparison_status.comparable_hashes
        if not comparable:
            reason_codes.append("codelength_comparability_failed")
            details.update(comparison_status.diagnostic_details(candidate_hash))

        model_terms = _model_terms(candidate)
        nonfinite_fields = [
            field_name
            for field_name, value in model_terms.items()
            if not math.isfinite(value)
        ]
        if nonfinite_fields:
            reason_codes.append("nonfinite_code_term")
            details["nonfinite_fields"] = nonfinite_fields

        fitted_values: tuple[float, ...] | None = None
        support_valid = True
        if not nonfinite_fields and comparable:
            fitted_values = _fitted_values(
                candidate=candidate,
                rows=legal_feature_view.rows,
                columnar_rows=columnar_rows,
            )
            if not all(math.isfinite(value) for value in fitted_values):
                reason_codes.append("nonfinite_code_term")
                details["nonfinite_fields"] = ["fitted_values"]
            else:
                support_valid = _values_match_support(
                    candidate.execution_layer.observation_model_binding,
                    (*observed_values, *fitted_values),
                )
                if not support_valid:
                    reason_codes.append("support_invalid")
                    details["support_kind"] = (
                        candidate.execution_layer.observation_model_binding.support_kind
                    )
        if not comparable:
            support_valid = False

        if not reason_codes and fitted_values is not None:
            _annotate_model_code_decomposition(
                candidate,
                lattice_policy=active_lattice_policy,
            )
            residual_indices = tuple(
                active_quantizer.quantize_index(actual - fitted)
                for actual, fitted in zip(observed_values, fitted_values, strict=True)
            )
            L_data_bits = data_code_length(
                residual_indices,
                data_code_family=data_code_family,
            )
            L_total_bits = (
                model_terms["L_family_bits"]
                + model_terms["L_structure_bits"]
                + model_terms["L_literals_bits"]
                + model_terms["L_params_bits"]
                + model_terms["L_state_bits"]
                + L_data_bits
            )
            description_gain_bits = float(reference_description.reference_bits) - float(
                L_total_bits
            )
            artifact = DescriptionGainArtifact(
                candidate_id=candidate_id,
                primitive_family=primitive_family,
                candidate_hash=candidate_hash,
                L_family_bits=model_terms["L_family_bits"],
                L_structure_bits=model_terms["L_structure_bits"],
                L_literals_bits=model_terms["L_literals_bits"],
                L_params_bits=model_terms["L_params_bits"],
                L_state_bits=model_terms["L_state_bits"],
                L_data_bits=float(L_data_bits),
                L_total_bits=float(L_total_bits),
                reference_bits=float(reference_description.reference_bits),
                description_gain_bits=float(description_gain_bits),
                quantization_mode=quantization_mode,
                quantization_step=active_quantizer.step_string,
                reference_policy_id=reference_description.reference_policy_id,
                reference_family_id=reference_description.reference_family_id,
                reference_scope=reference_description.reference_scope,
                data_code_family=data_code_family,
                coding_row_set_id=row_set_id,
                codelength_comparison_key=comparison_status.key_by_hash.get(
                    candidate_hash,
                    {},
                ),
                data_code_diagnostics=data_code_diagnostics(
                    residual_indices,
                    data_code_family=data_code_family,
                ),
                lattice_policy=active_lattice_policy.as_dict(),
                model_code_decomposition=_model_code_decomposition(
                    candidate,
                    lattice_policy=active_lattice_policy,
                ),
                comparable_group_id=comparison_status.group_id_by_hash.get(
                    candidate_hash,
                    "",
                ),
                comparable_group_size=comparison_status.group_size_by_hash.get(
                    candidate_hash,
                    0,
                ),
            )
            description_artifacts.append(artifact)
            if description_gain_bits <= minimum_gain:
                reason_codes.append(_description_gain_reason(minimum_gain))
                details["minimum_description_gain_bits"] = float(minimum_gain)
                details["description_gain_bits"] = float(description_gain_bits)
            else:
                accepted_candidates.append(candidate)

        diagnostics.append(
            DescriptiveAdmissibilityDiagnostic(
                candidate_id=candidate_id,
                primitive_family=primitive_family,
                candidate_hash=candidate_hash,
                is_admissible=not reason_codes,
                codelength_comparability=comparable,
                support_valid=support_valid,
                reason_codes=tuple(reason_codes),
                details=details,
            )
        )

    ranked_artifacts, selected_candidate_hashes = _rank_comparable_group_artifacts(
        description_artifacts,
        comparable_groups=comparison_status.comparable_groups,
    )

    return DescriptiveCodingResult(
        accepted_candidates=tuple(accepted_candidates),
        description_artifacts=ranked_artifacts,
        admissibility_diagnostics=tuple(diagnostics),
        selected_candidates=tuple(
            candidate_by_hash[candidate_hash]
            for candidate_hash in selected_candidate_hashes
            if candidate_hash in candidate_by_hash
        ),
    )


@dataclass(frozen=True)
class _ComparableGroup:
    group_id: str
    candidate_hashes: tuple[str, ...]
    comparison_key: Mapping[str, Any]


@dataclass(frozen=True)
class _ComparisonStatus:
    comparable_hashes: frozenset[str]
    details: Mapping[str, Any] = field(default_factory=dict)
    key_by_hash: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    comparable_groups: tuple[_ComparableGroup, ...] = ()
    group_id_by_hash: Mapping[str, str] = field(default_factory=dict)
    group_size_by_hash: Mapping[str, int] = field(default_factory=dict)
    non_comparable_details_by_hash: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )

    def diagnostic_details(self, candidate_hash: str) -> Mapping[str, Any]:
        return self.non_comparable_details_by_hash.get(candidate_hash, self.details)


def _comparison_status(
    candidates: Sequence[CandidateIntermediateRepresentation],
    *,
    quantization_mode: str,
    quantizer: FixedStepMidTreadQuantizer,
    reference_policy_id: str,
    reference_family_id: str,
    reference_scope: str,
    data_code_family: str,
    horizon_geometry: tuple[int, ...],
    coding_row_set_id: str,
    residual_history_construction: str,
    parameter_lattice_step: str,
    state_lattice_step: str,
    comparison_key_overrides: Mapping[str, CodelengthComparisonKey],
) -> _ComparisonStatus:
    keys_by_hash: dict[str, CodelengthComparisonKey] = {}
    for candidate in candidates:
        candidate_hash = candidate.canonical_hash()
        observation_model = candidate.execution_layer.observation_model_binding
        runtime_signature = composition_runtime_signature(
            candidate.structural_layer.composition_graph
        )
        keys_by_hash[candidate_hash] = comparison_key_overrides.get(
            candidate_hash,
            CodelengthComparisonKey(
                quantization_mode=quantization_mode,
                quantization_step=quantizer.step_string,
                reference_policy_id=reference_policy_id,
                reference_family_id=reference_family_id,
                reference_scope=reference_scope,
                data_code_family=data_code_family,
                support_kind=observation_model.support_kind,
                horizon_geometry=horizon_geometry,
                coding_row_set_id=coding_row_set_id,
                residual_history_construction=residual_history_construction,
                parameter_lattice_step=parameter_lattice_step,
                state_lattice_step=state_lattice_step,
                runtime_signature=runtime_signature or "none",
            ),
        )
    comparability = strict_single_class_law_eligibility(tuple(keys_by_hash.values()))
    serial_keys = {
        candidate_hash: key.as_dict() for candidate_hash, key in keys_by_hash.items()
    }
    if comparability.comparable:
        comparable_group = _ComparableGroup(
            group_id="codelength_comparison_group:1",
            candidate_hashes=tuple(candidate.canonical_hash() for candidate in candidates),
            comparison_key=(
                tuple(serial_keys.values())[0] if serial_keys else {}
            ),
        )
        return _ComparisonStatus(
            comparable_hashes=frozenset(
                candidate.canonical_hash() for candidate in candidates
            ),
            key_by_hash=serial_keys,
            comparable_groups=(comparable_group,) if candidates else (),
            group_id_by_hash={
                candidate.canonical_hash(): comparable_group.group_id
                for candidate in candidates
            },
            group_size_by_hash={
                candidate.canonical_hash(): len(candidates) for candidate in candidates
            },
        )
    grouped_hashes = _group_candidate_hashes_by_comparison_key(keys_by_hash)
    comparable_group_hashes = tuple(
        candidate_hashes
        for candidate_hashes in grouped_hashes
        if len(candidate_hashes) > 1
    )
    if comparable_group_hashes:
        comparable_groups = tuple(
            _ComparableGroup(
                group_id=f"codelength_comparison_group:{index}",
                candidate_hashes=candidate_hashes,
                comparison_key=serial_keys[candidate_hashes[0]],
            )
            for index, candidate_hashes in enumerate(comparable_group_hashes, start=1)
        )
        group_id_by_hash = {
            candidate_hash: group.group_id
            for group in comparable_groups
            for candidate_hash in group.candidate_hashes
        }
        group_size_by_hash = {
            candidate_hash: len(group.candidate_hashes)
            for group in comparable_groups
            for candidate_hash in group.candidate_hashes
        }
        comparable_hashes = frozenset(group_id_by_hash)
        singleton_hashes = tuple(
            candidate_hashes[0]
            for candidate_hashes in grouped_hashes
            if len(candidate_hashes) == 1
        )
        return _ComparisonStatus(
            comparable_hashes=comparable_hashes,
            key_by_hash=serial_keys,
            comparable_groups=comparable_groups,
            group_id_by_hash=group_id_by_hash,
            group_size_by_hash=group_size_by_hash,
            non_comparable_details_by_hash={
                candidate_hash: {
                    "comparison_failure_reason_code": "no_comparable_peer_in_batch",
                    "comparison_failure_details": {
                        "candidate_hash": candidate_hash,
                        "comparison_key": serial_keys[candidate_hash],
                    },
                    "comparison_key": serial_keys[candidate_hash],
                    "comparison_keys": serial_keys,
                    "comparable_group_size": 1,
                }
                for candidate_hash in singleton_hashes
            },
        )
    return _ComparisonStatus(
        comparable_hashes=frozenset(),
        details={
            "comparison_failure_reason_code": comparability.reason_code,
            "comparison_failure_details": dict(comparability.details or {}),
            "comparison_keys": serial_keys,
        },
        key_by_hash=serial_keys,
    )


def _group_candidate_hashes_by_comparison_key(
    keys_by_hash: Mapping[str, CodelengthComparisonKey],
) -> tuple[tuple[str, ...], ...]:
    grouped: dict[tuple[Any, ...], list[str]] = {}
    for candidate_hash, key in keys_by_hash.items():
        fingerprint = _comparison_key_fingerprint(key)
        grouped.setdefault(fingerprint, []).append(candidate_hash)
    return tuple(tuple(candidate_hashes) for candidate_hashes in grouped.values())


def _comparison_key_fingerprint(key: CodelengthComparisonKey) -> tuple[Any, ...]:
    return (
        key.quantization_mode,
        key.quantization_step,
        key.reference_scope,
        key.reference_family_id,
        key.reference_policy_id,
        key.data_code_family,
        key.support_kind,
        tuple(key.horizon_geometry),
        key.coding_row_set_id,
        key.residual_history_construction,
        key.parameter_lattice_step,
        key.state_lattice_step,
        key.runtime_signature,
    )


def _rank_comparable_group_artifacts(
    artifacts: Sequence[DescriptionGainArtifact],
    *,
    comparable_groups: Sequence[_ComparableGroup],
) -> tuple[tuple[DescriptionGainArtifact, ...], tuple[str, ...]]:
    artifact_by_hash = {artifact.candidate_hash: artifact for artifact in artifacts}
    rank_by_hash: dict[str, int] = {}
    selected_hashes: list[str] = []
    selected_hash_set: set[str] = set()
    for group in comparable_groups:
        group_artifacts = [
            artifact_by_hash[candidate_hash]
            for candidate_hash in group.candidate_hashes
            if candidate_hash in artifact_by_hash
        ]
        if not group_artifacts:
            continue
        ranked_group = sorted(group_artifacts, key=_description_artifact_sort_key)
        selected_hashes.append(ranked_group[0].candidate_hash)
        selected_hash_set.add(ranked_group[0].candidate_hash)
        for rank, artifact in enumerate(ranked_group, start=1):
            rank_by_hash[artifact.candidate_hash] = rank
    return (
        tuple(
            replace(
                artifact,
                comparable_group_rank=rank_by_hash.get(artifact.candidate_hash, 0),
                comparable_group_selected=artifact.candidate_hash in selected_hash_set,
            )
            for artifact in artifacts
        ),
        tuple(selected_hashes),
    )


def _description_artifact_sort_key(
    artifact: DescriptionGainArtifact,
) -> tuple[float, float, float, int, str]:
    return (
        float(artifact.L_total_bits),
        -float(artifact.description_gain_bits),
        float(artifact.L_structure_bits),
        len(str(artifact.candidate_hash)),
        artifact.candidate_id,
    )


def _model_terms(
    candidate: CandidateIntermediateRepresentation,
) -> dict[str, float]:
    decomposition = candidate.evidence_layer.model_code_decomposition
    return {
        "L_family_bits": float(decomposition.L_family_bits),
        "L_structure_bits": float(decomposition.L_structure_bits),
        "L_literals_bits": float(decomposition.L_literals_bits),
        "L_params_bits": float(decomposition.L_params_bits),
        "L_state_bits": float(decomposition.L_state_bits),
    }


def _model_code_decomposition(
    candidate: CandidateIntermediateRepresentation,
    *,
    lattice_policy: LatticePolicy,
) -> dict[str, Any]:
    decomposition = candidate.evidence_layer.model_code_decomposition.as_dict()
    return {
        **decomposition,
        **_lattice_decomposition_fields(lattice_policy),
        "lattice_policy": lattice_policy.as_dict(),
    }


def _annotate_model_code_decomposition(
    candidate: CandidateIntermediateRepresentation,
    *,
    lattice_policy: LatticePolicy,
) -> None:
    decomposition = candidate.evidence_layer.model_code_decomposition
    base_payload = {
        "L_family_bits": decomposition.L_family_bits,
        "L_structure_bits": decomposition.L_structure_bits,
        "L_literals_bits": decomposition.L_literals_bits,
        "L_params_bits": decomposition.L_params_bits,
        "L_state_bits": decomposition.L_state_bits,
    }
    payload = {
        **_lattice_decomposition_fields(lattice_policy),
        "lattice_policy": lattice_policy.as_dict(),
    }
    object.__setattr__(decomposition, "annotations", payload)


def _lattice_decomposition_fields(lattice_policy: LatticePolicy) -> dict[str, Any]:
    return {
        "parameter_lattice_step": lattice_policy.parameter_lattice_step,
        "state_lattice_step": lattice_policy.state_lattice_step,
        "parameter_lattice_policy_id": (
            f"fixed_step_mid_tread:{lattice_policy.parameter_lattice_step}"
        ),
        "state_lattice_policy_id": (
            f"fixed_step_mid_tread:{lattice_policy.state_lattice_step}"
        ),
    }


def _fitted_values(
    *,
    candidate: CandidateIntermediateRepresentation,
    rows: Sequence[Mapping[str, Any]],
    columnar_rows: _ColumnarFeatureRows,
) -> tuple[float, ...]:
    family_id = candidate.structural_layer.cir_family_id
    parameters = {
        parameter.name: parameter.value
        for parameter in candidate.structural_layer.parameter_block.parameters
    }
    literals = {
        literal.name: literal.value
        for literal in candidate.structural_layer.literal_block.literals
    }
    state = {
        slot.name: slot.value
        for slot in candidate.structural_layer.state_signature.persistent_state.slots
    }
    targets = columnar_rows.target_values
    operator_id = candidate.structural_layer.composition_graph.operator_id

    if family_id == "analytic":
        if operator_id == "piecewise":
            return _piecewise_fitted_values(
                candidate=candidate,
                parameters=parameters,
                rows=rows,
            )
        if operator_id == "additive_residual":
            return _additive_residual_fitted_values(
                candidate=candidate,
                parameters=parameters,
                rows=rows,
            )
        if operator_id == "regime_conditioned":
            return _regime_conditioned_fitted_values(
                candidate=candidate,
                parameters=parameters,
                rows=rows,
            )
        return _analytic_fitted_values_for_rows(parameters=parameters, rows=rows)

    if family_id == "recursive":
        if "level" in state:
            alpha = float(literals.get("alpha", 0.5))
            level = float(state["level"])
            fitted: list[float] = []
            for observed in targets:
                fitted.append(level)
                level = (alpha * float(observed)) + ((1.0 - alpha) * level)
            return tuple(fitted)
        if "running_mean" in state and "step_count" in state:
            running_mean = float(state["running_mean"])
            step_count = int(state["step_count"])
            fitted = []
            for observed in targets:
                fitted.append(running_mean)
                next_step_count = step_count + 1
                running_mean = (
                    (running_mean * step_count) + float(observed)
                ) / next_step_count
                step_count = next_step_count
            return tuple(fitted)

    if family_id == "spectral":
        harmonics = _spectral_harmonics(literals)
        season_length = int(literals["season_length"])
        phase_index = int(state.get("phase_index", 0))
        phase_indices = (
            phase_index + np.arange(columnar_rows.row_count)
        ) % season_length
        fitted = np.zeros(columnar_rows.row_count)
        harmonic_group = len(harmonics) > 1 or "harmonic_group" in literals
        for harmonic in harmonics:
            angles = (2.0 * math.pi * harmonic * phase_indices) / season_length
            if harmonic_group:
                cosine = float(parameters.get(f"cosine_{harmonic}_coefficient", 0.0))
                sine = float(parameters.get(f"sine_{harmonic}_coefficient", 0.0))
            else:
                cosine = float(parameters.get("cosine_coefficient", 0.0))
                sine = float(parameters.get("sine_coefficient", 0.0))
            fitted += (cosine * np.cos(angles)) + (sine * np.sin(angles))
        return tuple(float(value) for value in fitted.tolist())
    if family_id == "algorithmic":
        max_lag = int(candidate.execution_layer.history_access_contract.max_lag or 0)
        program = parse_algorithmic_program(
            str(literals["algorithmic_program"]),
            state_slot_count=int(literals.get("algorithmic_state_slot_count", 1)),
            max_program_nodes=int(literals.get("program_node_count", 8)),
            allowed_observation_lags=_algorithmic_allowed_observation_lags(
                literals=literals,
                max_lag=max_lag,
            ),
        )
        state_tuple = tuple(
            Fraction(str(state[f"state_{index}"]))
            for index in range(program.state_slot_count)
        )
        fitted: list[float] = []
        current_state = state_tuple
        for row, observed in zip(rows, targets, strict=True):
            observation_window = _algorithmic_observation_window(
                row=row,
                max_lag=max_lag,
                current_value=float(observed),
            )
            step = evaluate_algorithmic_program(
                program,
                state=current_state,
                observation=observation_window,
            )
            fitted.append(float(step.emit_value))
            current_state = step.next_state
        return tuple(fitted)

    raise ContractValidationError(
        code="unsupported_descriptive_coding_candidate",
        message=(
            "retained descriptive coding supports analytic, recursive, spectral, "
            "and algorithmic CIR candidates only"
        ),
        field_path="candidate.structural_layer.cir_family_id",
        details={"family_id": family_id},
    )


def _analytic_fitted_values_for_rows(
    *,
    parameters: Mapping[str, float | int],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, ...]:
    intercept = float(parameters.get("intercept", 0.0))
    feature_terms = _analytic_feature_terms(parameters)
    if not feature_terms:
        return tuple(float(intercept) for _ in rows)
    return tuple(
        float(
            intercept
            + sum(
                float(parameters[parameter_name]) * float(row[feature_name])
                for feature_name, parameter_name in feature_terms
            )
        )
        for row in rows
    )


def _analytic_feature_terms(
    parameters: Mapping[str, float | int],
) -> tuple[tuple[str, str], ...]:
    if "lag_coefficient" in parameters:
        return (("lag_1", "lag_coefficient"),)
    terms = []
    for parameter_name in sorted(parameters):
        if not parameter_name.endswith("__coefficient"):
            continue
        feature_name = parameter_name[: -len("__coefficient")]
        terms.append((feature_name, parameter_name))
    return tuple(terms)


def _spectral_harmonics(literals: Mapping[str, Any]) -> tuple[int, ...]:
    raw_group = literals.get("harmonic_group")
    if isinstance(raw_group, str) and raw_group.strip():
        return tuple(int(token) for token in raw_group.split(",") if token.strip())
    raw_harmonics = literals.get("harmonics")
    if isinstance(raw_harmonics, str) and raw_harmonics.strip():
        return tuple(int(token) for token in raw_harmonics.split(",") if token.strip())
    return (int(literals["harmonic"]),)


def _piecewise_fitted_values(
    *,
    candidate: CandidateIntermediateRepresentation,
    parameters: Mapping[str, float | int],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, ...]:
    composition = candidate.structural_layer.composition_graph.composition
    assert isinstance(composition, PiecewiseComposition)
    fitted: list[float] = []
    for row in rows:
        segment, _ = select_piecewise_segment(
            candidate.structural_layer.composition_graph,
            row=row,
        )
        branch_parameters = extract_component_mapping(parameters, segment.reducer_id)
        fitted.append(
            _analytic_fitted_values_for_rows(
                parameters=branch_parameters,
                rows=(row,),
            )[0]
        )
    return tuple(fitted)


def _additive_residual_fitted_values(
    *,
    candidate: CandidateIntermediateRepresentation,
    parameters: Mapping[str, float | int],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, ...]:
    composition = candidate.structural_layer.composition_graph.composition
    assert isinstance(composition, AdditiveResidualComposition)
    base_fitted = _analytic_fitted_values_for_rows(
        parameters=extract_component_mapping(parameters, composition.base_reducer),
        rows=rows,
    )
    residual_targets = tuple(
        float(row["target"]) - base
        for row, base in zip(rows, base_fitted, strict=True)
    )
    residual_rows = []
    for index, row in enumerate(rows):
        residual_row = dict(row)
        residual_row["target"] = residual_targets[index]
        residual_row["lag_1"] = 0.0 if index == 0 else residual_targets[index - 1]
        residual_rows.append(residual_row)
    residual_fitted = _analytic_fitted_values_for_rows(
        parameters=extract_component_mapping(
            parameters,
            composition.residual_reducer,
        ),
        rows=tuple(residual_rows),
    )
    return tuple(
        float(base + residual)
        for base, residual in zip(base_fitted, residual_fitted, strict=True)
    )


def _regime_conditioned_fitted_values(
    *,
    candidate: CandidateIntermediateRepresentation,
    parameters: Mapping[str, float | int],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, ...]:
    composition = candidate.structural_layer.composition_graph.composition
    assert isinstance(composition, RegimeConditionedComposition)
    fitted: list[float] = []
    for row in rows:
        branch_weights, _ = resolve_regime_weights(
            candidate.structural_layer.composition_graph,
            row=row,
        )
        branch_predictions = {
            branch.reducer_id: _analytic_fitted_values_for_rows(
                parameters=extract_component_mapping(parameters, branch.reducer_id),
                rows=(row,),
            )[0]
            for branch in composition.branch_reducers
            if branch.reducer_id in branch_weights
        }
        fitted.append(
            float(
                sum(
                    weight * branch_predictions[branch_id]
                    for branch_id, weight in branch_weights.items()
                )
            )
        )
    return tuple(fitted)


def _values_match_support(
    observation_model: BoundObservationModel,
    values: Sequence[float],
) -> bool:
    support_kind = observation_model.support_kind
    if support_kind == "all_real":
        return all(math.isfinite(value) for value in values)
    if support_kind == "positive_real":
        return all(math.isfinite(value) and value > 0.0 for value in values)
    if support_kind == "non_negative_real":
        return all(math.isfinite(value) and value >= 0.0 for value in values)
    return False


def _algorithmic_allowed_observation_lags(
    *,
    literals: Mapping[str, Any],
    max_lag: int,
) -> tuple[int, ...]:
    raw_lags = literals.get("algorithmic_allowed_observation_lags")
    if isinstance(raw_lags, str) and raw_lags.strip():
        return tuple(
            int(token)
            for token in raw_lags.split(",")
            if token.strip()
        )
    return tuple(range(max_lag + 1))


def _algorithmic_observation_window(
    *,
    row: Mapping[str, Any],
    max_lag: int,
    current_value: float,
) -> tuple[Fraction, ...]:
    values: list[Fraction] = [Fraction(str(float(current_value)))]
    for lag in range(1, max_lag + 1):
        field_name = f"lag_{lag}"
        if field_name not in row:
            raise ContractValidationError(
                code="unsupported_descriptive_coding_candidate",
                message=(
                    "algorithmic lagged observation access requires explicit "
                    f"{field_name} features on every scored row"
                ),
                field_path=f"rows.{field_name}",
            )
        values.append(Fraction(str(float(row[field_name]))))
    return tuple(values)


def _code_bits(indices: Sequence[int]) -> int:
    return sum(signed_integer_code_length(index) for index in indices)


def _coerce_reference_policy(
    reference_policy: ReferenceDescriptionPolicy | Mapping[str, Any] | None,
) -> ReferenceDescriptionPolicy:
    if reference_policy is None:
        return ReferenceDescriptionPolicy()
    if isinstance(reference_policy, ReferenceDescriptionPolicy):
        return reference_policy
    return ReferenceDescriptionPolicy(
        reference_family_id=str(
            reference_policy.get(
                "reference_family_id",
                "raw_quantized_transformed_sequence",
            )
        ),
        policy_id=(
            str(reference_policy["policy_id"])
            if reference_policy.get("policy_id") is not None
            else None
        ),
        reference_scope=str(
            reference_policy.get("reference_scope", "raw_observation_reference")
        ),
    )


def _description_gain_reason(minimum_description_gain_bits: float) -> str:
    if minimum_description_gain_bits <= 0.0:
        return "description_gain_non_positive"
    return "description_gain_below_floor"


def _columnar_feature_rows(
    rows: Sequence[Mapping[str, Any]],
) -> _ColumnarFeatureRows:
    lag_1_present = all("lag_1" in row for row in rows)
    return _ColumnarFeatureRows(
        target_values=np.fromiter((float(row["target"]) for row in rows), dtype=float),
        lag_1_values=(
            np.fromiter((float(row["lag_1"]) for row in rows), dtype=float)
            if lag_1_present
            else None
        ),
        row_count=len(rows),
    )


__all__ = [
    "DescriptionGainArtifact",
    "DescriptiveAdmissibilityDiagnostic",
    "DescriptiveCodingResult",
    "evaluate_descriptive_candidates",
]
