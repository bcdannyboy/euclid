from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from euclid.contracts.errors import ContractValidationError
from euclid.math.quantization import FixedStepMidTreadQuantizer


_DEFAULT_POLICY_ID = "active_lattice_policy_v1"
_FALLBACK_REASON = "defaults_to_residual_quantization_step"


@dataclass(frozen=True)
class LatticePolicy:
    parameter_lattice_step: str
    state_lattice_step: str
    parameter_lattice_reason: str
    state_lattice_reason: str
    policy_id: str = _DEFAULT_POLICY_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameter_lattice_step",
            _canonical_step(
                self.parameter_lattice_step,
                field_path="parameter_lattice_step",
            ),
        )
        object.__setattr__(
            self,
            "state_lattice_step",
            _canonical_step(
                self.state_lattice_step,
                field_path="state_lattice_step",
            ),
        )
        object.__setattr__(
            self,
            "parameter_lattice_reason",
            _require_reason(
                self.parameter_lattice_reason,
                field_path="parameter_lattice_reason",
            ),
        )
        object.__setattr__(
            self,
            "state_lattice_reason",
            _require_reason(
                self.state_lattice_reason,
                field_path="state_lattice_reason",
            ),
        )
        object.__setattr__(
            self,
            "policy_id",
            _require_reason(self.policy_id, field_path="policy_id"),
        )

    @classmethod
    def from_steps(
        cls,
        *,
        parameter_lattice_step: str | None,
        state_lattice_step: str | None,
        residual_quantization_step: str,
        parameter_lattice_reason: str | None = None,
        state_lattice_reason: str | None = None,
    ) -> "LatticePolicy":
        fallback_step = _canonical_step(
            residual_quantization_step,
            field_path="residual_quantization_step",
        )
        return cls(
            parameter_lattice_step=parameter_lattice_step or fallback_step,
            state_lattice_step=state_lattice_step or fallback_step,
            parameter_lattice_reason=(
                parameter_lattice_reason
                or (
                    _FALLBACK_REASON
                    if parameter_lattice_step is None
                    else "explicit_parameter_lattice_step"
                )
            ),
            state_lattice_reason=(
                state_lattice_reason
                or (
                    _FALLBACK_REASON
                    if state_lattice_step is None
                    else "explicit_state_lattice_step"
                )
            ),
        )

    @classmethod
    def coerce(
        cls,
        value: "LatticePolicy | Mapping[str, Any] | None",
        *,
        residual_quantization_step: str,
        parameter_lattice_step: str | None = None,
        state_lattice_step: str | None = None,
    ) -> "LatticePolicy":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.from_steps(
                parameter_lattice_step=parameter_lattice_step,
                state_lattice_step=state_lattice_step,
                residual_quantization_step=residual_quantization_step,
            )
        return cls(
            parameter_lattice_step=str(value["parameter_lattice_step"]),
            state_lattice_step=str(value["state_lattice_step"]),
            parameter_lattice_reason=str(
                value.get("parameter_lattice_reason", "explicit_parameter_lattice_step")
            ),
            state_lattice_reason=str(
                value.get("state_lattice_reason", "explicit_state_lattice_step")
            ),
            policy_id=str(value.get("policy_id", _DEFAULT_POLICY_ID)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "parameter_lattice_step": self.parameter_lattice_step,
            "state_lattice_step": self.state_lattice_step,
            "parameter_lattice_reason": self.parameter_lattice_reason,
            "state_lattice_reason": self.state_lattice_reason,
        }

    def as_artifact(self) -> dict[str, Any]:
        return {
            "artifact_kind": "lattice_policy",
            "artifact_status": "active",
            **self.as_dict(),
        }


def _canonical_step(value: str, *, field_path: str) -> str:
    try:
        return FixedStepMidTreadQuantizer.from_string(str(value)).step_string
    except ContractValidationError as exc:
        raise ContractValidationError(
            code="invalid_lattice_policy",
            message=f"{field_path} must be a strictly positive decimal string",
            field_path=field_path,
            details={"source_code": exc.code},
        ) from exc


def _require_reason(value: str, *, field_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(
            code="invalid_lattice_policy",
            message=f"{field_path} must be a non-empty string",
            field_path=field_path,
        )
    return value


__all__ = ["LatticePolicy"]
