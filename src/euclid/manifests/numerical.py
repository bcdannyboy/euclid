from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import yaml


@dataclass(frozen=True)
class NumericalPolicy:
    policy_id: str
    absolute_tolerance: float
    relative_tolerance: float
    optimizer_max_iterations: int
    deterministic_seed_namespace: str
    failure_thresholds: Mapping[str, float]
    allowed_instability_downgrades: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "optimizer_max_iterations": self.optimizer_max_iterations,
            "deterministic_seed_namespace": self.deterministic_seed_namespace,
            "failure_thresholds": dict(self.failure_thresholds),
            "allowed_instability_downgrades": list(
                self.allowed_instability_downgrades
            ),
        }


DEFAULT_NUMERICAL_POLICY = NumericalPolicy(
    policy_id="euclid_numerical_policy_v1",
    absolute_tolerance=1e-9,
    relative_tolerance=1e-9,
    optimizer_max_iterations=10_000,
    deterministic_seed_namespace="euclid-numerical-v1",
    failure_thresholds={
        "max_condition_number": 1e12,
        "max_replay_score_delta": 1e-9,
        "max_optimizer_failure_rate": 0.0,
    },
    allowed_instability_downgrades=(
        "optimizer_nonconvergence",
        "ill_conditioned_design",
        "external_runtime_unavailable",
    ),
)


def derive_deterministic_seed(task_id: str, scope: str) -> int:
    payload = (
        f"{DEFAULT_NUMERICAL_POLICY.deterministic_seed_namespace}:{task_id}:{scope}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def load_numerical_policy(path: Path | str) -> NumericalPolicy:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    policy_payload = payload.get("default_policy", payload)
    return NumericalPolicy(
        policy_id=str(policy_payload["policy_id"]),
        absolute_tolerance=float(policy_payload["absolute_tolerance"]),
        relative_tolerance=float(policy_payload["relative_tolerance"]),
        optimizer_max_iterations=int(policy_payload["optimizer_max_iterations"]),
        deterministic_seed_namespace=str(
            policy_payload["deterministic_seed_namespace"]
        ),
        failure_thresholds={
            str(key): float(value)
            for key, value in dict(policy_payload["failure_thresholds"]).items()
        },
        allowed_instability_downgrades=tuple(
            str(item)
            for item in policy_payload.get("allowed_instability_downgrades", ())
        ),
    )


__all__ = [
    "DEFAULT_NUMERICAL_POLICY",
    "NumericalPolicy",
    "derive_deterministic_seed",
    "load_numerical_policy",
]
