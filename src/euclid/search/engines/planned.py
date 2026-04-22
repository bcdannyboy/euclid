from __future__ import annotations

from euclid.search.engine_contracts import (
    EngineFailureDiagnostic,
    EngineInputContext,
    EngineRunResult,
    SearchEngine,
)


class PlannedEnginePlaceholder(SearchEngine):
    engine_id: str
    engine_version = "0.0-planned"
    phase_id = "future"
    replacement_note = "engine implementation is planned but not production-ready"

    def run(self, context: EngineInputContext) -> EngineRunResult:
        return EngineRunResult(
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            status="failed",
            candidates=(),
            failure_diagnostics=(
                EngineFailureDiagnostic(
                    engine_id=self.engine_id,
                    reason_code="engine_not_implemented",
                    message=self.replacement_note,
                    recoverable=True,
                    details={"phase_id": self.phase_id},
                ),
            ),
            trace={
                "planned_engine_placeholder": True,
                "phase_id": self.phase_id,
                "replacement_note": self.replacement_note,
            },
            omission_disclosure={
                "omitted_due_to_planned_engine_placeholder": True,
                "phase_id": self.phase_id,
            },
            replay_metadata={
                **context.replay_metadata(),
                "engine_id": self.engine_id,
                "engine_version": self.engine_version,
                "phase_id": self.phase_id,
            },
        )


__all__ = ["PlannedEnginePlaceholder"]
