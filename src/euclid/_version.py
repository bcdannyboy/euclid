from __future__ import annotations

PACKAGE_VERSION = "1.0.0"
RELEASE_TARGET_VERSION = "1.0.0"
RELEASE_WORKFLOW_ID = "euclid_current_release_candidate"
RELEASE_CERTIFICATION_TEST_TARGETS = (
    "tests/unit",
    "tests/integration",
    "tests/golden",
    "tests/benchmarks",
    "tests/perf",
)
RELEASE_CERTIFICATION_COMMANDS = (
    "python -m euclid release repo-test-matrix",
    "python -m euclid benchmarks run --suite current-release.yaml --no-resume",
    "python -m euclid benchmarks run --suite full-vision.yaml --no-resume",
    "python -m euclid run --config examples/full_vision_run.yaml",
    "python -m euclid replay --run-id full-vision-run",
    "python -m euclid release certify-clean-install",
    "python -m euclid release status",
    "python -m euclid release verify-completion",
    "python -m euclid release certify-research-readiness",
)

__all__ = [
    "PACKAGE_VERSION",
    "RELEASE_CERTIFICATION_COMMANDS",
    "RELEASE_CERTIFICATION_TEST_TARGETS",
    "RELEASE_TARGET_VERSION",
    "RELEASE_WORKFLOW_ID",
]
