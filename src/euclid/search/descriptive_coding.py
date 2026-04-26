from __future__ import annotations

import math
from dataclasses import dataclass, field
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
    data_code_family: str = "residual_signed_integer_elias_delta_v1"
    coding_row_set_id: str = ""
    codelength_comparison_key: Mapping[str, Any] = field(default_factory=dict)
    data_code_diagnostics: Mapping[str, Any] = field(default_factory=dict)


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
    )
    row_set_id = coding_row_set_id(tuple(legal_feature_view.rows))
    parameter_lattice = parameter_lattice_step or active_quantizer.step_string
    state_lattice = state_lattice_step or active_quantizer.step_string
    comparison_status = _comparison_status(
        candidates,
        quantization_mode=quantization_mode,
        quantizer=active_quantizer,
        reference_policy_id=reference_description.reference_policy_id,
        data_code_family=data_code_family,
        horizon_geometry=tuple(int(horizon) for horizon in horizon_geometry),
        coding_row_set_id=row_set_id,
        residual_history_construction=residual_history_construction,
        parameter_lattice_step=parameter_lattice,
        state_lattice_step=state_lattice,
        comparison_key_overrides=comparison_key_overrides or {},
    )
    minimum_gain = (
        0.0 if minimum_description_gain_bits is None else minimum_description_gain_bits
    )

    accepted_candidates: list[CandidateIntermediateRepresentation] = []
    description_artifacts: list[DescriptionGainArtifact] = []
    diagnostics: list[DescriptiveAdmissibilityDiagnostic] = []

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
            details.update(comparison_status.details)

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

    return DescriptiveCodingResult(
        accepted_candidates=tuple(accepted_candidates),
        description_artifacts=tuple(description_artifacts),
        admissibility_diagnostics=tuple(diagnostics),
    )


@dataclass(frozen=True)
class _ComparisonStatus:
    comparable_hashes: frozenset[str]
    details: Mapping[str, Any] = field(default_factory=dict)
    key_by_hash: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def _comparison_status(
    candidates: Sequence[CandidateIntermediateRepresentation],
    *,
    quantization_mode: str,
    quantizer: FixedStepMidTreadQuantizer,
    reference_policy_id: str,
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
        return _ComparisonStatus(
            comparable_hashes=frozenset(
                candidate.canonical_hash() for candidate in candidates
            ),
            key_by_hash=serial_keys,
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
