from __future__ import annotations

from pathlib import Path

from euclid.runtime import profiling


class _PathsStub:
    def __init__(self, sealed_run_root: Path) -> None:
        self.sealed_run_root = sealed_run_root


def test_resolved_request_id_prefers_summary_payload_value() -> None:
    request_id = profiling._resolved_request_id(  # noqa: SLF001
        paths=_PathsStub(Path("/tmp/sealed-runs/fallback-run")),
        summary_payload={"request_id": "declared-run"},
    )

    assert request_id == "declared-run"


def test_resolved_request_id_falls_back_to_sealed_run_root_name() -> None:
    request_id = profiling._resolved_request_id(  # noqa: SLF001
        paths=_PathsStub(Path("/tmp/sealed-runs/fallback-run")),
        summary_payload={},
    )

    assert request_id == "fallback-run"
