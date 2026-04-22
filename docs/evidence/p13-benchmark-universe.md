# P13 Benchmark Universe Evidence

Date: 2026-04-21

P13 replaces artifact-existence benchmark success with semantic benchmark assertions.
Task result artifacts now emit `semantic_assertions` beside the existing semantic
summary. The assertion block records metric-threshold declarations, engine
requirements, expected claim ceilings, false-claim expectations, and semantic
readiness row ids.

Benchmark success remains non-closing claim evidence. Surface evidence reports
`claim_evidence_status: not_claim_evidence`, and surfaces fail closed when a task
result omits semantic assertions or marks them failed, even if benchmark artifacts
and replay refs exist.

Fixture and expectation rationale:

- Current-release and full-vision suite/example manifests were restored as the
  required benchmark/operator entrypoints.
- Portfolio benchmark expectations now assert the declared selection semantics
  instead of pinning one candidate id or old code-bit values.
- Real-series benchmark expectations now assert claim-scope boundaries. A
  benchmark-local reconstruction may remain descriptive, but it cannot promote
  an operator abstention into a holistic or predictive-within-scope claim.
- No golden fixture was updated in this P13 pass. The changed behavior is covered
  by semantic benchmark/regression tests rather than snapshot churn.

Primary evidence:

- `tests/benchmarks/test_p13_benchmark_universe.py`
- `tests/benchmarks/test_multi_backend_portfolio.py`
- `tests/benchmarks/test_real_series_honesty_regressions.py`
- `tests/regression/test_benchmark_evidence_redaction.py`
- `tests/integration/test_release_candidate_workflow.py`
