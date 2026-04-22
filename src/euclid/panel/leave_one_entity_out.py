from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LeaveOneEntityOutSplit:
    heldout_entity: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "heldout_entity": self.heldout_entity,
            "train_indices": list(self.train_indices),
            "test_indices": list(self.test_indices),
        }


@dataclass(frozen=True)
class LeaveOneEntityOutResult:
    status: str
    fold_scores: tuple[Mapping[str, Any], ...]
    mean_loss: float | None
    transport_evidence_allowed: bool
    evidence_role: str
    reason_codes: tuple[str, ...]
    replay_identity: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": "leave_one_entity_out_evidence@1.0.0",
            "status": self.status,
            "fold_scores": [dict(score) for score in self.fold_scores],
            "mean_loss": self.mean_loss,
            "transport_evidence_allowed": self.transport_evidence_allowed,
            "evidence_role": self.evidence_role,
            "reason_codes": list(self.reason_codes),
            "replay_identity": self.replay_identity,
        }


def make_leave_one_entity_out_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    entity_field: str = "entity",
) -> tuple[LeaveOneEntityOutSplit, ...]:
    entities = sorted(
        {
            str(row.get(entity_field, "")).strip()
            for row in rows
            if str(row.get(entity_field, "")).strip()
        }
    )
    splits: list[LeaveOneEntityOutSplit] = []
    for entity in entities:
        test_indices = tuple(
            index
            for index, row in enumerate(rows)
            if str(row.get(entity_field, "")).strip() == entity
        )
        train_indices = tuple(
            index
            for index, row in enumerate(rows)
            if str(row.get(entity_field, "")).strip() != entity
        )
        splits.append(
            LeaveOneEntityOutSplit(
                heldout_entity=entity,
                train_indices=train_indices,
                test_indices=test_indices,
            )
        )
    return tuple(splits)


def validate_leave_one_entity_out(
    rows: Sequence[Mapping[str, Any]],
    *,
    entity_field: str = "entity",
    target_field: str = "target",
    min_entities: int = 3,
    max_mean_loss: float = 0.1,
) -> LeaveOneEntityOutResult:
    rows_tuple = tuple(rows)
    splits = make_leave_one_entity_out_splits(rows_tuple, entity_field=entity_field)
    if len(splits) < int(min_entities):
        return _result(
            status="failed",
            fold_scores=(),
            mean_loss=None,
            transport_evidence_allowed=False,
            reason_codes=("insufficient_entities",),
        )

    fold_scores: list[dict[str, Any]] = []
    for split in splits:
        train_targets = [
            float(rows_tuple[index][target_field]) for index in split.train_indices
        ]
        test_targets = [
            float(rows_tuple[index][target_field]) for index in split.test_indices
        ]
        prediction = fmean(train_targets)
        loss = fmean((target - prediction) ** 2 for target in test_targets)
        fold_scores.append(
            {
                "heldout_entity": split.heldout_entity,
                "loss": float(round(loss, 12)),
                "train_count": len(split.train_indices),
                "test_count": len(split.test_indices),
            }
        )
    mean_loss = float(round(fmean(score["loss"] for score in fold_scores), 12))
    passed = mean_loss <= float(max_mean_loss)
    return _result(
        status="passed" if passed else "failed",
        fold_scores=tuple(fold_scores),
        mean_loss=mean_loss,
        transport_evidence_allowed=passed,
        reason_codes=() if passed else ("unseen_entity_holdout_failed",),
    )


def _result(
    *,
    status: str,
    fold_scores: tuple[Mapping[str, Any], ...],
    mean_loss: float | None,
    transport_evidence_allowed: bool,
    reason_codes: tuple[str, ...],
) -> LeaveOneEntityOutResult:
    identity_payload = {
        "fold_scores": [dict(score) for score in fold_scores],
        "mean_loss": mean_loss,
        "reason_codes": list(reason_codes),
        "status": status,
    }
    return LeaveOneEntityOutResult(
        status=status,
        fold_scores=fold_scores,
        mean_loss=mean_loss,
        transport_evidence_allowed=transport_evidence_allowed,
        evidence_role="unseen_entity_holdout",
        reason_codes=reason_codes,
        replay_identity=f"leave-one-entity-out:{_digest(identity_payload)}",
    )


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "LeaveOneEntityOutResult",
    "LeaveOneEntityOutSplit",
    "make_leave_one_entity_out_splits",
    "validate_leave_one_entity_out",
]
