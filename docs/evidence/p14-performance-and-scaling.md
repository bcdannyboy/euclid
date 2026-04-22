# P14 Performance And Scaling Evidence

Date: 2026-04-21

P14 adds deterministic performance controls without turning runtime success into
scientific claim evidence.

Implemented controls:

- Evaluation cache keys are content-addressed, category scoped, and replay-visible.
  Covered cache categories are feature matrices, expression evaluations, subtree
  evaluations, fitted constants, and simplification results.
- Replay-safe parallel execution aggregates by stable item id, reports worker
  failures as recoverable diagnostics, and preserves deterministic replay identity
  across worker counts when outputs and diagnostics match.
- Runtime timeout and resource exhaustion surfaces fail closed into timeout,
  degraded, or abstained decisions instead of silently promoting partial work.
- Candidate throughput and engine runtime budget reports record measured rates,
  declared thresholds, resource limits, and degradation reason codes.
- Benchmark task telemetry includes a `benchmark_budget_report` attribute with
  candidate limit, wall-clock budget, worker count, and submitter count.

Fixture and golden rationale:

- No golden fixture was updated in this P14 pass. The new behavior is covered by
  unit/performance assertions and benchmark telemetry checks rather than snapshot
  churn.
- E-graph resource exhaustion uses the existing bounded equality saturation
  fixture and records the partial result as graceful degradation.

Primary evidence:

- `tests/unit/runtime/test_cache.py`
- `tests/unit/runtime/test_parallel.py`
- `tests/perf/test_candidate_throughput.py`
- `tests/perf/test_engine_runtime_budgets.py`
- `tests/unit/benchmarks/test_runtime.py`
