# Euclid Full-Vision Wave 4 Progress

Date: 2026-05-25

Status: active, fail-closed. This wave continues the full objective and does
not redefine readiness around a smaller target.

## Starting State

Authoritative inputs:

- Current worktree in `/Users/danielbloom/Desktop/euclid`.
- Wave 3 ledger:
  `docs/progress/2026-05-25-full-vision-wave-3.md`.
- Top-level release status after Wave 3 remained `Target ready: no`.

Known remaining blockers entering Wave 4:

- `release.evidence_freshness_repo_test_matrix_source_digest_mismatch`
- `evidence_lane.boundary_specific_external_evidence_missing`
- `evidence_lane.descriptive_compression_missing`
- `evidence_lane.predictive_generalization_missing`
- `evidence_lane.readiness_and_closure_missing`
- `evidence_lane.replay_verification_missing`
- `evidence_lane.robustness_missing`

Wave 3 jury also flagged stale integration tests that still expected
`certify-clean-install` to fail even though clean-install certification now
correctly proves installed-runtime surfaces while global release readiness
remains blocked.

## Wave 4 Planning Swarm

Planning swarm size: 5 read-only agents.

Agents:

- Release engineering / certification contract:
  `019e6210-17f0-7be1-888d-4e90d2e332c0`
- Completion report / evidence-lane semantics:
  `019e6210-1998-7e20-8e41-282c51be75af`
- Benchmarks / repo-test matrix:
  `019e6210-1ae7-79d2-9fbe-1eda698c29fd`
- Mathematical realism / CIR / reducers:
  `019e6210-1c5e-7971-9b23-4d504b44879f`
- UX / contracts / security / supply-chain:
  `019e6210-1dce-73c3-8f52-63afa039a40e`

Scope:

- Design the next actual implementation swarm.
- Keep file/module ownership disjoint.
- Identify tasks that should stay local with the coordinator.
- Preserve fail-closed release truth and avoid reclassifying missing evidence
  as proof.

## Local Critical Path

The coordinator is reproducing the stale clean-install expectations before
editing:

`PYTHONPATH=src python3.11 -m pytest -q tests/integration/test_final_release_certification.py::test_shipped_releasable_is_not_alias_of_current_release tests/integration/test_completion_report_generation.py::test_release_certification_flow_captures_clean_install_surfaces_and_packaging_evidence --tb=short`

Expected red condition:

- Tests that still assert `certify-clean-install` exits `1` should fail
  against the current clean-install semantics.
- The fix must update tests/contracts to assert clean-install pass as runtime
  install evidence while preserving top-level release blocked status.

## Evidence Log

### Local Stale Clean-Install Test Red

Command:

`PYTHONPATH=src python3.11 -m pytest -q tests/integration/test_final_release_certification.py::test_shipped_releasable_is_not_alias_of_current_release tests/integration/test_completion_report_generation.py::test_release_certification_flow_captures_clean_install_surfaces_and_packaging_evidence --tb=short`

Result:

- Exit `1`
- `2 failed in 571.36s`
- Both failures showed `certify-clean-install` exiting `0` and printing all
  seven surfaces as `passed`, while the tests still expected exit `1`.

Interpretation:

- The tests were stale against Wave 3 semantics.
- Correct behavior is: clean-install certification can pass as an
  installed-runtime evidence surface, but final release readiness remains
  blocked until all policy/evidence gates close.

### Local Patch

Changed files:

- `tests/integration/test_final_release_certification.py`
- `tests/integration/test_full_vision_closure_report.py`
- `tests/integration/test_completion_report_generation.py`
- `src/euclid/cli/__init__.py`

Patch summary:

- Integration tests now expect `certify-clean-install` to exit `0` when all
  installed-runtime surfaces pass.
- Tests assert the CLI states the scope explicitly:
  `installed-runtime certification only; not final release readiness`.
- Completion-report tests now require `packaging_install` to be available from
  clean-install evidence while `readiness_and_closure` remains partial because
  `governance_spec` is still missing.
- Shipped releasable policy tests still require the policy verdict to remain
  `blocked`.
- CLI output now prints the installed-runtime scope distinction before surface
  completion.

Focused verification:

`PYTHONPATH=src python3.11 -m pytest -q tests/integration/test_final_release_certification.py::test_shipped_releasable_is_not_alias_of_current_release tests/integration/test_final_release_certification.py::test_shipped_releasable_uses_packaging_install_without_aliasing_readiness tests/integration/test_completion_report_generation.py::test_release_certification_flow_captures_clean_install_surfaces_and_packaging_evidence tests/integration/test_full_vision_closure_report.py::test_release_status_emits_closure_metadata_and_scope_evidence_bundles tests/integration/test_full_vision_closure_report.py::test_full_vision_only_rows_do_not_close_from_current_release_bundle --tb=short`

Result:

- Exit `0`

- `5 passed in 2082.02s (0:34:42)`

Whitespace hygiene:

`git diff --check -- src/euclid/cli/__init__.py tests/integration/test_final_release_certification.py tests/integration/test_full_vision_closure_report.py tests/integration/test_completion_report_generation.py docs/progress/2026-05-25-full-vision-wave-4.md`

Result:

- Exit `0`

### Portfolio Runtime Worker Result

Worker:

- `019e6240-6145-7c10-bad0-009ef2755f82`

Changed files:

- `src/euclid/performance.py`
- `src/euclid/benchmarks/runtime.py`

Patch summary:

- Added `TelemetryRecorder.allocation_tracing_paused()`.
- Preserves the pre-pause observed peak memory before stopping tracing.
- Wraps benchmark threshold-metric enrichment in a telemetry span and pauses
  allocation tracing inside the expensive replay block.
- Did not raise performance budgets.

Worker root-cause finding:

- `tracemalloc` allocation tracking inflated post-submit threshold metric
  replay, with seasonal threshold replay moving from roughly `20s+` to about
  `2.95s` when allocation tracing is paused only for that block.

Coordinator verification:

`PYTHONPATH=src python3.11 -m pytest -q tests/perf/test_portfolio_runtime.py::test_current_release_portfolio_runtime_stays_within_budget --tb=short`

Result:

- Exit `0`

### Fresh Repo Matrix Attempt 1

Command:

`PYTHONPATH=src python3.11 -m euclid release repo-test-matrix --project-root .`

Result:

- Exit `1`
- Report: `build/reports/repo_test_matrix.json`
- Source digest:
  `repo_checkout_digest:635fc6acdd5ce4b061eaf4f02a6f71aa2ab55169a2e9375c2a481f894a211d88`
- Counts:
  - failed: `3`
  - passed: `1350`
  - skipped: `0`
  - xfailed: `0`
  - xpassed: `0`
- Summary: `3 failed, 1350 passed, 11 warnings in 1953.61s (0:32:33)`

Failures:

- `tests/unit/test_dev_scripts_smoke.py::test_release_and_install_smoke_scripts_use_packaged_runtime_surfaces`
- `tests/integration/test_completion_regression_ci_contract.py::test_ci_executes_every_required_certification_command`
- `tests/benchmarks/test_current_release_readiness.py::test_current_release_suite_is_truthfully_narrow`

Patch:

- `.github/workflows/ci.yml` now uses the frozen clean-install wheel directory
  from the command contract:
  `build/certification/clean_install/wheels`.
- `tests/unit/test_dev_scripts_smoke.py` now asserts the release smoke script
  reads `run-result.json` and replays the run id from that artifact instead of
  expecting a hard-coded replay literal.
- `tests/benchmarks/test_current_release_readiness.py` now expects the
  current-release benchmark suite readiness judgment to be `ready` and
  `public` when its four narrow retained surfaces pass. This does not imply
  global release readiness; release status remains governed by evidence lanes
  and freshness gates.

Focused verification:

`PYTHONPATH=src python3.11 -m pytest -q tests/unit/test_dev_scripts_smoke.py::test_release_and_install_smoke_scripts_use_packaged_runtime_surfaces tests/integration/test_completion_regression_ci_contract.py::test_ci_executes_every_required_certification_command tests/benchmarks/test_current_release_readiness.py::test_current_release_suite_is_truthfully_narrow --tb=short`

Result:

- Exit `0`
- `3 passed in 11.43s`

Whitespace hygiene:

`git diff --check -- .github/workflows/ci.yml tests/unit/test_dev_scripts_smoke.py tests/benchmarks/test_current_release_readiness.py`

Result:

- Exit `0`

### Fresh Repo Matrix Attempt 2

Command:

`PYTHONPATH=src python3.11 -m euclid release repo-test-matrix --project-root .`

Result:

- Exit `0`
- Report: `build/reports/repo_test_matrix.json`
- Source digest:
  `repo_checkout_digest:5b870d05a146e121afbeb8799ed86cd3f3deaa61d255886a23ab341dac7c3605`
- Counts:
  - failed: `0`
  - passed: `1353`
  - skipped: `0`
  - xfailed: `0`
  - xpassed: `0`
- Summary: `1353 passed, 11 warnings in 1962.25s (0:32:42)`

Interpretation:

- The repo-test matrix freshness blocker is eligible to clear after
  downstream certification artifacts are regenerated against the same source
  state.

### Post-Matrix Certification Evidence Refresh

Current-release benchmark evidence:

`PYTHONPATH=src python3.11 -m euclid benchmarks run --suite current-release.yaml --no-resume --benchmark-root build/certification/current_release_suite`

Result:

- Exit `0`
- Suite: `current_release`

Full-vision benchmark evidence:

`PYTHONPATH=src python3.11 -m euclid benchmarks run --suite full-vision.yaml --no-resume --benchmark-root build/certification/full_vision_suite`

Result:

- Exit `0`
- Suite: `full_vision`

Operator run evidence:

`PYTHONPATH=src python3.11 -m euclid run --config examples/full_vision_run.yaml --output-root build/certification/full_vision_run --evidence-report build/reports/full_vision_operator_run_evidence.json`

Result:

- Exit `0`
- Run id: `full-vision-run`
- Run result ref:
  `run_result_manifest@1.1.0:full-vision-run_run_result`

Operator replay evidence:

`PYTHONPATH=src python3.11 -m euclid replay --run-id full-vision-run --output-root build/certification/full_vision_run --evidence-report build/reports/full_vision_operator_replay_evidence.json`

Result:

- Exit `0`
- Replay verification: `verified`

Clean-install certification:

`PYTHONPATH=src python3.11 -m euclid release certify-clean-install --project-root . --wheel-dir build/certification/clean_install/wheels --output-root build/certification/clean_install`

Result:

- Exit `0`
- Surface completion: `1.000000`
- All seven installed-runtime surfaces passed.
- CLI output explicitly says this is installed-runtime certification only, not
  final release readiness.

Release status after these refreshes:

`PYTHONPATH=src python3.11 -m euclid release status --project-root .`

Result:

- Exit `0`
- `Target ready: no`
- The six evidence-lane blockers cleared.
- Remaining blockers:
  - `release.evidence_freshness_repo_test_matrix_source_digest_mismatch`
  - `release.evidence_freshness_full_vision_operator_run_source_digest_mismatch`
  - `release.evidence_freshness_full_vision_operator_replay_source_digest_mismatch`

Root cause:

- Clean-install/build generated `src/euclid.egg-info`.
- The release source digest included `*.egg-info` metadata under `src`, so the
  source digest changed after packaging itself.

TDD red:

`PYTHONPATH=src python3.11 -m pytest -q tests/unit/test_release.py::test_release_source_digest_ignores_build_egg_info_metadata --tb=short`

Result:

- Exit `1`
- The digest changed after adding `src/euclid.egg-info/PKG-INFO` and
  `SOURCES.txt`.

Patch:

- `src/euclid/release.py` now excludes any path part ending in `.egg-info`
  from release source digest inputs.
- `tests/unit/test_release.py` covers the regression.

Green:

`PYTHONPATH=src python3.11 -m pytest -q tests/unit/test_release.py::test_release_source_digest_ignores_build_egg_info_metadata --tb=short`

Result:

- Exit `0`
- `1 passed in 1.80s`

Current digest after the fix:

- `repo_checkout_digest:41e4a469ac5744849a1b3892accdf8e09dc0906cba3e9e93383144a5cc5f94fa`

Whitespace hygiene:

`git diff --check -- src/euclid/release.py tests/unit/test_release.py`

Result:

- Exit `0`
- `1 passed in 11.81s`

Nearby verification:

`PYTHONPATH=src python3.11 -m pytest -q tests/unit/benchmarks/test_runtime.py tests/perf/test_runtime_smoke.py --tb=short`

Result:

- Exit `0`
- `13 passed in 5.50s`

Whitespace hygiene:

`git diff --check -- src/euclid/performance.py src/euclid/benchmarks/runtime.py`

Result:

- Exit `0`

### Release Status After Stale-Test Patch

Command:

`PYTHONPATH=src python3.11 -m euclid release status --project-root .`

Result:

- Exit `0`
- `Target ready: no`
- Current release verdict: `blocked`
- Full vision verdict: `blocked`
- Shipped or releasable verdict: `blocked`

Current-release reason codes:

- `evidence_lane.descriptive_compression_missing`
- `evidence_lane.predictive_generalization_missing`
- `evidence_lane.readiness_and_closure_missing`
- `evidence_lane.replay_verification_missing`
- `evidence_lane.robustness_missing`
- `release.evidence_freshness_repo_test_matrix_source_digest_mismatch`
- `release.evidence_freshness_full_vision_operator_run_source_digest_mismatch`
- `release.evidence_freshness_full_vision_operator_replay_source_digest_mismatch`

Interpretation:

- The clean-install stale expectation has been repaired in tests.
- The new source changes made the Wave 3 operator run/replay evidence stale
  again; those artifacts must be regenerated once Wave 4 source churn stops.
- The repo-test matrix remains stale.
- The six evidence-lane blockers remain real release blockers.

Completion verification:

`PYTHONPATH=src python3.11 -m euclid release verify-completion --project-root .`

Result:

- Exit `1`
- Policies `current_release_v1`, `full_vision_v1`, and
  `shipped_releasable_v1` remain blocked past the transition window ending
  `2026-05-15T00:00:00Z`.
- Incomplete rows remain:
  - `evidence_lane:boundary_specific_external_evidence`
  - `evidence_lane:descriptive_compression`
  - `evidence_lane:predictive_generalization`
  - `evidence_lane:readiness_and_closure`
  - `evidence_lane:replay_verification`
  - `evidence_lane:robustness`
- Freshness failures remain:
  - `repo_test_matrix_source_digest_mismatch`
  - `full_vision_operator_run_source_digest_mismatch`
  - `full_vision_operator_replay_source_digest_mismatch`

Stale matrix snapshot:

- `build/reports/repo_test_matrix.json` was generated at
  `2026-05-24T16:32:07Z`.
- It recorded `19 failed, 1321 passed, 0 skipped`.
- Because the snapshot predates Wave 3 and Wave 4 source changes, each failure
  cluster must be rerun before it is treated as current.

### Current Failure Cluster Reruns

No longer current failures:

- `PYTHONPATH=src python3.11 -m pytest -q tests/unit/benchmarks/test_reporting_phase31_worker.py::test_missing_required_observed_metric_fails_closed_with_source_submitter tests/unit/benchmarks/test_reporting_phase31_worker.py::test_safe_abstention_missing_metrics_pass_when_expected_and_no_winner --tb=short`
  reported `2 passed in 1.99s`.
- `PYTHONPATH=src python3.11 -m pytest -q tests/unit/test_completion_report_logic.py::test_completion_report_confidence_reflects_real_signal_coverage tests/unit/test_completion_report_models.py::test_completion_report_schema_declares_split_completion_fields_and_status_values --tb=short`
  reported `2 passed in 1.99s`.
- `PYTHONPATH=src python3.11 -m pytest -q tests/benchmarks/test_suite_runner.py::test_current_release_suite_uses_canonical_active_scope_name tests/benchmarks/test_suite_runner.py::test_profile_benchmark_suite_uses_explicit_project_root_when_installed --tb=short`
  reported `2 passed in 37.57s`.
- `PYTHONPATH=src python3.11 -m pytest -q tests/benchmarks/test_multi_backend_portfolio.py::test_benchmark_portfolio_records_ranked_finalists_in_replay_contract --tb=short`
  reported `1 passed in 28.36s`.
- `PYTHONPATH=src python3.11 -m pytest -q tests/integration/test_completion_report_generation.py::test_completion_report_makes_incomplete_rows_and_blockers_explicit --tb=short`
  reported `1 passed in 134.86s`.
- `PYTHONPATH=src python3.11 -m pytest -q tests/benchmarks/test_full_vision_suite.py::test_profile_benchmark_suite_runs_declared_full_vision_tasks_and_writes_summary --tb=short`
  reported `1 passed in 81.87s`.

Still current failures:

- `PYTHONPATH=src python3.11 -m pytest -q tests/golden/test_algorithmic_publication_bundles.py::test_algorithmic_benchmark_publication_matches_golden_fixture tests/golden/test_probabilistic_publication_bundles.py::test_probabilistic_distribution_publication_matches_golden_fixture tests/golden/test_probabilistic_publication_bundles.py::test_probabilistic_distribution_downgrade_matches_golden_fixture --tb=short`
  reported `3 failed in 6.61s`.
- `PYTHONPATH=src python3.11 -m pytest -q tests/perf/test_portfolio_runtime.py::test_current_release_portfolio_runtime_stays_within_budget tests/perf/test_research_readiness_runtime_profile.py::test_research_readiness_evaluation_stays_within_runtime_budget --tb=short`
  reported `2 failed in 39.27s`.

Current implementation targets:

- Golden publication/fixture drift.
- Current-release portfolio runtime budget.
- Research-readiness seeded evidence now blocked by current release evidence
  freshness.

## Wave 4 Implementation Workers

Active workers:

- Golden publication/fixture drift:
  `019e6240-5ff6-7e13-9e37-8c0d18fdaf95`
- Current-release portfolio runtime budget:
  `019e6240-6145-7c10-bad0-009ef2755f82`
- Research-readiness current evidence semantics:
  `019e6240-639c-73f2-b2c2-a45641c8f1c1`

Coordinator-owned while workers run:

- Maintain this ledger.
- Avoid overlapping edits in worker-owned files.
- Review and verify worker patches before integrating claims.
- Regenerate operator run/replay, clean-install, repo matrix, release status,
  verify-completion, and research readiness only after source churn settles.

### Golden Worker Result

Worker:

- `019e6240-5ff6-7e13-9e37-8c0d18fdaf95`

Changed files:

- `fixtures/runtime/phase06/algorithmic-benchmark-publication-golden.json`
- `fixtures/runtime/phase06/probabilistic-distribution-publication-golden.json`
- `fixtures/runtime/phase06/probabilistic-distribution-downgrade-golden.json`
- `src/euclid/_assets/fixtures/runtime/phase06/algorithmic-benchmark-publication-golden.json`
- `src/euclid/_assets/fixtures/runtime/phase06/probabilistic-distribution-publication-golden.json`
- `src/euclid/_assets/fixtures/runtime/phase06/probabilistic-distribution-downgrade-golden.json`

Semantic review:

- Algorithmic publication now selects the canonical
  `algorithmic_last_observation` candidate and includes SHA-256 artifact refs.
- Probabilistic publication fixtures now preserve fail-closed predictive
  semantics: insufficient paired evidence blocks predictive support while
  descriptive and stochastic evidence remains explicit.
- Packaged fixture mirrors were updated with the source fixtures.

Coordinator verification:

`PYTHONPATH=src python3.11 -m pytest -q tests/golden/test_algorithmic_publication_bundles.py::test_algorithmic_benchmark_publication_matches_golden_fixture tests/golden/test_probabilistic_publication_bundles.py::test_probabilistic_distribution_publication_matches_golden_fixture tests/golden/test_probabilistic_publication_bundles.py::test_probabilistic_distribution_downgrade_matches_golden_fixture --tb=short`

Result:

- Exit `0`
- `3 passed in 5.60s`

### Authority Snapshot Hash Drift

Fresh spec compiler command:

`PYTHONPATH=src python3.11 -m pytest -q tests/spec_compiler --tb=short`

Result:

- Exit `1`
- `1 failed, 186 passed in 82.02s`
- Failing test:
  `tests/spec_compiler/test_authority_snapshot.py::test_authority_snapshot_hashes_match_authority_docs`
- Root cause: `README.md` live SHA-256 is
  `0e842d05ca48184ff6980188728183524ff9ae98522f769368871faf7cf9da5f`, while
  the authority snapshot still carried
  `8107491503a3e03bccbbe43b50978554b64acee642671ba445ba2497c468e8b4`.

Patch:

- Updated `README.md` hash in:
  - `docs/implementation/authority-snapshot.yaml`
  - `src/euclid/_assets/docs/implementation/authority-snapshot.yaml`

Verification:

`PYTHONPATH=src python3.11 -m pytest -q tests/spec_compiler --tb=short`

Result:

- Exit `0`
- `187 passed in 81.75s (0:01:21)`

### Research-Readiness Worker Result

Worker:

- `019e6240-639c-73f2-b2c2-a45641c8f1c1`

Changed files:

- `tests/fixtures/research_readiness.py`
- `tests/perf/test_research_readiness_runtime_profile.py`
- `tests/integration/test_research_readiness_certification.py`

Patch summary:

- Extracted a shared research-readiness evidence seeder.
- Seeded current-source matrix digest, clean-install surfaces, operator run
  binding, replay digest, and replay-to-run report binding fields.
- Updated perf and integration tests to share the current fixture instead of
  carrying stale pre-Wave-3 evidence payloads.
- Added negative research-readiness tests for failed full-vision surfaces and
  missing declared search-class surface coverage.

Coordinator verification:

`PYTHONPATH=src python3.11 -m pytest -q tests/perf/test_research_readiness_runtime_profile.py::test_research_readiness_evaluation_stays_within_runtime_budget tests/integration/test_research_readiness_certification.py --tb=short`

Result:

- Exit `0`
- `4 passed in 2.83s`

Whitespace hygiene:

`git diff --check -- tests/fixtures/research_readiness.py tests/perf/test_research_readiness_runtime_profile.py tests/integration/test_research_readiness_certification.py`

Result:

- Exit `0`

### Planning Swarm Partial Results

Returned planners:

- `019e6210-1998-7e20-8e41-282c51be75af`
- `019e6210-17f0-7be1-888d-4e90d2e332c0`
- `019e6210-1ae7-79d2-9fbe-1eda698c29fd`
- `019e6210-1c5e-7971-9b23-4d504b44879f`

Consensus:

- Keep stale clean-install tests and CLI wording on the immediate critical
  path.
- Run the fresh repo-test matrix only after source churn settles.
- Assign follow-on implementation over disjoint surfaces:
  release/completion semantics, benchmark/matrix triage, golden/fixture
  publication, evidence-lane governance, math/CIR/search/reducers,
  probabilistic/robustness support, claim-surface truth, and research
  readiness.
- Preserve `Target ready: no` until current-source evidence proves all release
  and full-vision gates.

### Fresh Matrix After Egg-Info Digest Fix

Command:

`PYTHONPATH=src python3.11 -m euclid release repo-test-matrix --project-root .`

Result:

- Exit `1`
- Report generated at `2026-05-26T05:06:22Z`
- Source digest:
  `repo_checkout_digest:41e4a469ac5744849a1b3892accdf8e09dc0906cba3e9e93383144a5cc5f94fa`
- Summary: `2 failed, 1352 passed, 11 warnings in 1963.87s (0:32:43)`
- Skips remain `0`; xfail/xpass remain `0`.

Failing tests:

- `tests/integration/test_completion_report_generation.py::test_release_certification_flow_captures_clean_install_surfaces_and_packaging_evidence`
- `tests/integration/test_final_release_certification.py::test_shipped_releasable_uses_packaging_install_without_aliasing_readiness`

Root cause candidate:

- These two stale clean-install/readiness tests still expect installed-runtime
  clean-install certification to leave the readiness-and-closure evidence lane
  partial.
- The completion report generated at `2026-05-26T04:59:00Z` shows all
  capability rows complete, with policy readiness still blocked by evidence
  freshness verdicts.
- The later repo-test matrix generated at `2026-05-26T05:06:22Z` is still red:
  `2 failed, 1352 passed`. Those failures are stale readiness tests that still
  assert `readiness_and_closure` is partial/missing governance, even though the
  current semantics allow clean-install evidence to complete that lane.
- Therefore: capability-lane completion is evidence progress only;
  shipped/release readiness remains blocked until the matrix is green and
  current-source matrix, operator run, and operator replay evidence are
  regenerated and accepted.

Current remaining work:

- Patch the two stale readiness tests and run their focused pytest command.
- Rerun the full repo-test matrix because the release gate is currently red.
- Once the matrix is green and source churn stops, regenerate benchmarks,
  operator run, operator replay, clean-install, release status, completion
  verification, and research-readiness certification in strict order.
- Run an odd-numbered independent expert jury after the evidence chain is
  refreshed and before any readiness claim.

### Readiness Test Patch Red/Green

Validation swarm:

- `019e62b2-3214-7362-866c-e0605cfe7b66` found the first patch was too broad:
  `readiness_and_closure == complete` depends on a passing ambient repo-matrix
  artifact, not clean-install alone.
- `019e62b2-336c-7452-b1c2-d151767e36b0` independently found the same issue
  and recommended asserting clean-install packaging evidence without masking
  clean-install freshness failures.
- `019e62b2-3486-7ea1-940c-e2baff7256f6` found the ledger needed to separate
  capability-lane progress from still-blocked policy readiness.

TDD red:

`PYTHONPATH=src python3.11 -m pytest -q tests/integration/test_completion_report_generation.py::test_release_certification_flow_captures_clean_install_surfaces_and_packaging_evidence tests/integration/test_final_release_certification.py::test_shipped_releasable_uses_packaging_install_without_aliasing_readiness --tb=short`

Result:

- Exit `1`
- `2 failed in 357.09s (0:05:57)`
- Both failures proved the overbroad assertion:
  `readiness_and_closure` was `partial`, not `complete`, when the ambient
  repo-test matrix artifact was red.

Patch:

- The two tests now assert only the stable clean-install contract:
  `packaging_install` is available, the missing-packaging reason is absent, the
  canonical `build/reports/clean-install-certification.json` report is attached,
  the row carries `shipped_releasable_clean_install_bundle`, and policy verdicts
  do not blame clean-install freshness.
- They no longer assume repo-matrix governance evidence has passed before the
  matrix command itself refreshes that artifact.

Focused green:

`PYTHONPATH=src python3.11 -m pytest -q tests/integration/test_completion_report_generation.py::test_release_certification_flow_captures_clean_install_surfaces_and_packaging_evidence tests/integration/test_final_release_certification.py::test_shipped_releasable_uses_packaging_install_without_aliasing_readiness --tb=short`

Result:

- Exit `0`
- `2 passed in 354.00s (0:05:54)`

Whitespace hygiene:

`git diff --check -- tests/integration/test_completion_report_generation.py tests/integration/test_final_release_certification.py docs/progress/2026-05-25-full-vision-wave-4.md`

Result:

- Exit `0`

### Fresh Matrix After Readiness Test Patch

Command:

`PYTHONPATH=src python3.11 -m euclid release repo-test-matrix --project-root .`

Result:

- Exit `0`
- Report generated at `2026-05-26T05:58:41Z`
- Source digest:
  `repo_checkout_digest:29bb2d72ef2d0c1b234bfda3cd11c42091be75d7c72d28d8bd5613f12367b2f4`
- Summary: `1354 passed, 11 warnings in 1965.84s (0:32:45)`
- Counts:
  - failed: `0`
  - passed: `1354`
  - skipped: `0`
  - xfailed: `0`
  - xpassed: `0`

Interpretation:

- The repo-test matrix is green for the current source digest.
- The next required work is to refresh source-bound certification artifacts in
  order: current-release benchmark suite, full-vision benchmark suite, operator
  run, operator replay, clean-install, release status, verify-completion, and
  research-readiness certification.

### Certification Evidence Refresh

Current-release benchmark suite:

`PYTHONPATH=src python3.11 -m euclid benchmarks run --suite current-release.yaml --no-resume --benchmark-root build/certification/current_release_suite`

Result:

- Exit `0`
- Suite: `current_release`
- Wrote summary:
  `build/certification/current_release_suite/results/suites/current_release/benchmark-suite-summary.json`

Full-vision benchmark suite:

`PYTHONPATH=src python3.11 -m euclid benchmarks run --suite full-vision.yaml --no-resume --benchmark-root build/certification/full_vision_suite`

Result:

- Exit `0`
- Suite: `full_vision`
- Wrote summary:
  `build/certification/full_vision_suite/results/suites/full_vision/benchmark-suite-summary.json`

Operator run:

`PYTHONPATH=src python3.11 -m euclid run --config examples/full_vision_run.yaml --output-root build/certification/full_vision_run --evidence-report build/reports/full_vision_operator_run_evidence.json`

Result:

- Exit `0`
- Run id: `full-vision-run`
- Evidence digest:
  `repo_checkout_digest:29bb2d72ef2d0c1b234bfda3cd11c42091be75d7c72d28d8bd5613f12367b2f4`
- Run summary SHA:
  `runtime_sha256:4273d65834cf844ae491d9b7d88278ba5bcc30a1c9888a5d6db45dc3affbd9f5`

Operator replay:

`PYTHONPATH=src python3.11 -m euclid replay --run-id full-vision-run --output-root build/certification/full_vision_run --evidence-report build/reports/full_vision_operator_replay_evidence.json`

Result:

- Exit `0`
- Replay verification: `verified`
- Evidence digest:
  `repo_checkout_digest:29bb2d72ef2d0c1b234bfda3cd11c42091be75d7c72d28d8bd5613f12367b2f4`

Clean-install certification:

`PYTHONPATH=src python3.11 -m euclid release certify-clean-install --project-root . --wheel-dir build/certification/clean_install/wheels --output-root build/certification/clean_install`

Result:

- Exit `0`
- Scope caveat: installed-runtime certification only; not final release
  readiness.
- Surface completion: `1.000000`
- All seven surfaces passed:
  `release_status`, `operator_run`, `operator_replay`, `determinism_same_seed`,
  `performance_runtime_smoke`, `packaged_notebook_smoke`, and
  `benchmark_execution`.
- Build digest:
  `repo_checkout_digest:29bb2d72ef2d0c1b234bfda3cd11c42091be75d7c72d28d8bd5613f12367b2f4`

Release status:

`PYTHONPATH=src python3.11 -m euclid release status --project-root .`

Result:

- Exit `0`
- `Target ready: yes`
- Current release verdict: `ready`
- Full vision verdict: `ready`
- Shipped or releasable verdict: `ready`
- All reason-code lists were empty.
- Completion report generated at `2026-05-26T06:03:24Z`.
- Completion values:
  - `current_gate_completion`: `1`
  - `full_vision_completion`: `1`
  - `shipped_releasable_completion`: `1`
- `unresolved_blockers`: `[]`
- Claim truth summary: `ready: true`, `truth_status: ready`,
  zero failed/missing/policy-blocked proofs.

Completion verification:

`PYTHONPATH=src python3.11 -m euclid release verify-completion --project-root .`

Result:

- Exit `0`
- Passed: `yes`

Research-readiness certification:

`PYTHONPATH=src python3.11 -m euclid release certify-research-readiness --project-root .`

Result:

- Exit `0`
- Status: `ready`
- Evidence freshness failures: `[]`
- Full-vision surface status failures: `[]`
- Policy verdicts:
  - `current_release_v1`: `ready`
  - `full_vision_v1`: `ready`
  - `shipped_releasable_v1`: `ready`

Remaining required work before completion claim:

- Run the post-wave odd-numbered independent expert jury across mathematics,
  statistics, forecasting, symbolic regression, software architecture,
  reproducibility, security, UX, and release engineering.
- Address any jury blocker or formally justify any non-blocking residual risk.

### Post-Wave Expert Jury

Thread-cap note:

- Attempted to launch all nine jurors in parallel.
- Two jurors launched; seven launches returned `agent thread limit reached`.
- Jury is continuing in two-at-a-time batches, preserving the nine-seat odd
  panel.

Returned verdicts:

- Mathematics/CIR/Reducers: `APPROVE`, confidence `0.86`, no blockers.
  - Rationale: mathematical realism is bounded and credible; codelength
    comparability is strict; CIR normalization keeps provenance/replay metadata
    auditable without changing scientific identity; descriptive-vs-predictive
    separation is explicitly implemented and tested.
  - Non-blocking caveat: `tests/live` and `tests/spec_compiler` are outside the
    repo-test-matrix target set. Spec compiler was verified separately in this
    wave, and live-provider success is not represented as required release
    evidence.
- Statistics/Calibration/Probabilistic: `APPROVE`, confidence `0.86`, no
  blockers.
  - Rationale: probabilistic forecast surfaces require type-matched
    calibration; calibration blocks mismatched types, confirmatory-row misuse,
    insufficient samples, insufficient partitions, and failed diagnostics;
    paired evidence and effective sample-size policies fail closed.
  - Non-blocking caveat: some benchmark fixtures are intentionally small, but
    publication/claim surfaces block predictive support when paired evidence is
    insufficient or calibration fails.
- Forecasting/Time-Safe Evidence: `APPROVE`, confidence `0.84`, no blockers.
  - Rationale: readiness reports are current and clean; operator run/replay are
    source-bound to the same digest; the inspected operator publication does
    not overclaim predictive support and correctly downgrades to descriptive
    structure when calibration blocks promotion.
  - Non-blocking caveat: the current operator publication must not be described
    as an affirmative out-of-sample probabilistic forecasting win.
- Symbolic Regression/Search: `APPROVE`, confidence `0.84`, no blockers.
  - Rationale: exact finite enumeration, bounded heuristic, equality
    saturation heuristic, and stochastic heuristic are declared separately with
    exactness ceilings; full-vision evidence covers search-class honesty,
    algorithmic backend, composition semantics, and portfolio replay.
  - Non-blocking caveat: equality-saturation readiness is an honesty-contract
    proof over declared candidates, not a broad independent e-graph discovery
    benchmark.
- Software Architecture/Runtime-Control: `APPROVE`, confidence `0.84`, no
  blockers.
  - Rationale: source-bound repo matrix, benchmark suites, operator run/replay,
    clean-install wheel install outside the repo, release status,
    verify-completion, and research-readiness certification exercise the
    intended production paths.
  - Non-blocking caveat: this seat fills one jury role only; the full nine-seat
    jury record must still be completed before a final jury-complete claim.
- Reproducibility/Replay/Clean-Install: `APPROVE`, confidence `0.89`, no
  blockers.
  - Rationale: current release source digest matches the matrix digest;
    operator run/replay evidence is source-fresh and replay verified;
    clean-install report records all seven surfaces passed and digest/hash
    spot-checks matched files on disk.
  - Non-blocking caveat: verdict is based on read-only inspection plus
    read-only digest/hash checks, after coordinator reran mutating
    certification commands.
- UX/Workbench/Claim-Surface: `APPROVE`, confidence `0.88`, no blockers.
  - Rationale: completion report keeps `claim_truth_summary` primary and
    labels completion values as secondary progress evidence, not readiness;
    workbench claim surfaces separate claim ceiling from candidate/equation
    display and expose replay/artifact links.
  - Non-blocking caveat: the juror ran frontend verification read-only from
    their context (`npm run test:frontend -- --run`, `42 passed`), but the
    coordinator is keeping release readiness grounded in canonical release and
    certification gates.
- Security/Supply-Chain: `REVISE`, confidence `0.80`.
  - Blocker 1: CI hard-coded `--run-id full-vision-run` for replay instead of
    deriving the run id from `build/certification/full_vision_run/run-result.json`.
  - Blocker 2: release source freshness did not include `.github/workflows` or
    `uv.lock`, so CI-command and dependency-lock drift could remain invisible
    to source-bound evidence.
  - Blocker 3: CI used live editable pip resolution (`pip install -e ".[dev]"`)
    despite a hash-bearing `uv.lock`.

### Security Jury Revision

Focused RED:

`PYTHONPATH=src python3.11 -m pytest -q tests/integration/test_completion_regression_ci_contract.py::test_ci_executes_every_required_certification_command tests/unit/test_release.py::test_release_source_digest_tracks_ci_workflow_and_dependency_lock --tb=short`

Result:

- Exit `1`
- `2 failed in 2.09s`
- Failures proved:
  - CI still contained `--run-id full-vision-run`.
  - Editing `.github/workflows/ci.yml` did not change the release source
    digest.

Patch:

- `.github/workflows/ci.yml` now uses `astral-sh/setup-uv@v5`, runs
  `uv sync --locked --all-extras --dev`, and places `.venv/bin` on `GITHUB_PATH`
  for Python jobs.
- Release-status CI now reads `run_id` from
  `build/certification/full_vision_run/run-result.json` before invoking replay.
- `_RELEASE_SOURCE_DIGEST_PATHS` now includes `.github/workflows` and `uv.lock`.
- CI contract tests reject hard-coded replay ids and live editable pip install
  shortcuts.
- Release digest tests prove workflow and lockfile changes alter the source
  digest, while generated `.egg-info` metadata remains ignored.

Focused GREEN:

`PYTHONPATH=src python3.11 -m pytest -q tests/unit/test_release.py::test_release_source_digest_ignores_build_egg_info_metadata tests/unit/test_release.py::test_release_source_digest_tracks_ci_workflow_and_dependency_lock tests/integration/test_completion_regression_ci_contract.py::test_ci_executes_every_required_certification_command --tb=short`

Result:

- Exit `0`
- `3 passed in 2.02s`

Consequence:

- All previous source-bound release artifacts are stale because digest-included
  CI, source, and test files changed.
- Required next step: rerun the full repo matrix, then rerun the ordered
  certification evidence chain and resume the jury.

### Fresh Matrix After Security Revision

Command:

`PYTHONPATH=src python3.11 -m euclid release repo-test-matrix --project-root .`

Result:

- Exit `0`
- Report generated at `2026-05-26T07:14:18Z`
- Source digest:
  `repo_checkout_digest:31f05b506e7b9d19878df57f4cae67fa1d5ff45cf3be3d6fac8e7e366409e673`
- Summary: `1355 passed, 11 warnings in 1936.19s (0:32:16)`
- Counts:
  - failed: `0`
  - passed: `1355`
  - skipped: `0`
  - xfailed: `0`
  - xpassed: `0`

Whitespace hygiene:

`git diff --check -- .github/workflows/ci.yml src/euclid/release.py tests/integration/test_completion_regression_ci_contract.py tests/unit/test_release.py docs/progress/2026-05-25-full-vision-wave-4.md`

Result:

- Exit `0`

### Certification Refresh After Action-Pinning Revision

Current-release benchmark suite:

`PYTHONPATH=src python3.11 -m euclid benchmarks run --suite current-release.yaml --no-resume --benchmark-root build/certification/current_release_suite`

Result:

- Exit `0`
- Suite: `current_release`
- Evidence generated at `2026-05-26T08:16:15Z`
- Source digest:
  `repo_checkout_digest:3c9bcef0e9c0ac97cc6215f55a9800709563081e635cd86c1285d1ea3e03c8f2`
- Surface statuses: all benchmark and replay statuses passed for
  `retained_core_release`, `algorithmic_backend`,
  `shared_plus_local_decomposition`, and `mechanistic_lane`.

Full-vision benchmark suite:

`PYTHONPATH=src python3.11 -m euclid benchmarks run --suite full-vision.yaml --no-resume --benchmark-root build/certification/full_vision_suite`

Result:

- Exit `0`
- Suite: `full_vision`
- Evidence generated at `2026-05-26T08:16:15Z`
- Source digest:
  `repo_checkout_digest:3c9bcef0e9c0ac97cc6215f55a9800709563081e635cd86c1285d1ea3e03c8f2`
- Surface statuses: all benchmark and replay statuses passed for
  `retained_core_release`, `probabilistic_forecast_surface`,
  `algorithmic_backend`, `search_class_honesty`,
  `composition_operator_semantics`, `shared_plus_local_decomposition`,
  `mechanistic_lane`, `external_evidence_ingestion`, `robustness_lane`,
  and `portfolio_orchestration`.

Operator run:

`PYTHONPATH=src python3.11 -m euclid run --config examples/full_vision_run.yaml --output-root build/certification/full_vision_run --evidence-report build/reports/full_vision_operator_run_evidence.json`

Result:

- Exit `0`
- Run id: `full-vision-run`
- Run result ref:
  `run_result_manifest@1.1.0:full-vision-run_run_result`
- Bundle ref:
  `reproducibility_bundle_manifest@1.0.0:full-vision-run_bundle`
- Forecast object type: `distribution`
- Selected family: `algorithmic`
- Source digest:
  `repo_checkout_digest:3c9bcef0e9c0ac97cc6215f55a9800709563081e635cd86c1285d1ea3e03c8f2`

Operator replay:

`PYTHONPATH=src python3.11 -m euclid replay --run-id full-vision-run --output-root build/certification/full_vision_run --evidence-report build/reports/full_vision_operator_replay_evidence.json`

Result:

- Exit `0`
- Replay verification status: `verified`
- Evidence generated at `2026-05-26T08:13:03Z`
- Source digest:
  `repo_checkout_digest:3c9bcef0e9c0ac97cc6215f55a9800709563081e635cd86c1285d1ea3e03c8f2`
- Run-id binding ties requested run id, run-result object id, and run-summary
  request id to `full-vision-run`.

Clean-install certification:

`PYTHONPATH=src python3.11 -m euclid release certify-clean-install --project-root . --wheel-dir build/certification/clean_install/wheels --output-root build/certification/clean_install`

Result:

- Exit `0`
- Report generated at `2026-05-26T08:15:33Z`
- Surface completion: `1`
- Runtime dependency wheel count: `92`
- Source digest at build:
  `repo_checkout_digest:3c9bcef0e9c0ac97cc6215f55a9800709563081e635cd86c1285d1ea3e03c8f2`
- Wheel digest:
  `wheel_digest:525ad2d3c9f390066b0f2ab6a2276d9f09fccbad93f73a8bf72496e7d5c0bddc`
- All seven surfaces passed with empty reason codes:
  `release_status`, `operator_run`, `operator_replay`,
  `determinism_same_seed`, `performance_runtime_smoke`,
  `packaged_notebook_smoke`, and `benchmark_execution`.

Release status:

`PYTHONPATH=src python3.11 -m euclid release status --project-root .`

Result:

- Exit `0`
- `Target ready: yes`
- Current release verdict: `ready`
- Full vision verdict: `ready`
- Shipped or releasable verdict: `ready`
- All reason-code lists were empty.
- All enhancement phase gates `P00` through `P16` were complete.
- Completion report generated at `2026-05-26T08:16:22Z`.
- Completion values:
  - `current_gate_completion`: `1`
  - `full_vision_completion`: `1`
  - `shipped_releasable_completion`: `1`
- `unresolved_blockers`: `[]`
- Claim truth summary: `ready: true`, `truth_status: ready`,
  zero failed/missing/policy-blocked proofs.

Completion verification:

`PYTHONPATH=src python3.11 -m euclid release verify-completion --project-root .`

Result:

- Exit `0`
- Passed: `yes`

Research-readiness certification:

`PYTHONPATH=src python3.11 -m euclid release certify-research-readiness --project-root .`

Result:

- Exit `0`
- Report generated at `2026-05-26T08:16:35Z`
- Status: `ready`
- Reason codes: `[]`
- Evidence freshness failures: `[]`
- Full-vision surface status failures: `[]`
- Policy verdicts:
  - `current_release_v1`: `ready`
  - `full_vision_v1`: `ready`
  - `shipped_releasable_v1`: `ready`

Auxiliary supply-chain/spec checks:

- `UV_CACHE_DIR=/private/tmp/euclid-uv-cache uv lock --check`
  - Exit `0`
  - `Resolved 127 packages in 10ms`
- `PYTHONPATH=src python3.11 -m pytest -q tests/spec_compiler --tb=short`
  - Exit `0`
  - `187 passed in 80.29s (0:01:20)`

Consequence:

- Source-bound matrix and certification artifacts now agree on digest
  `repo_checkout_digest:3c9bcef0e9c0ac97cc6215f55a9800709563081e635cd86c1285d1ea3e03c8f2`.
- Proceeded to final independent re-review against the action-pinning revision
  and fresh certification chain.

### Final Independent Re-Review Panel

Odd-numbered final panel:

- Security/Supply-Chain juror: `APPROVE`, confidence `0.90`, blockers: none.
- Release Engineering/CI/Packaging juror: `APPROVE`, confidence `0.90`,
  blockers: none.
- Reproducibility/Replay/Clean-Install juror: `APPROVE`,
  confidence `0.92`, blockers: none.

Security evidence accepted by juror:

- External GitHub Actions are pinned to immutable 40-character SHAs.
- Python CI uses `uv sync --locked --all-extras --dev`, with no editable live
  install.
- Replay run id is derived from `build/certification/full_vision_run/run-result.json`
  via `FULL_VISION_RUN_ID`.
- Release source digest includes `.github/workflows`, `uv.lock`,
  `package.json`, and `package-lock.json`.
- Current evidence is bound to
  `repo_checkout_digest:3c9bcef0e9c0ac97cc6215f55a9800709563081e635cd86c1285d1ea3e03c8f2`.

Release-engineering evidence accepted by juror:

- CI release smoke runs the repo matrix, both benchmark suites, operator run,
  operator replay, clean-install certification, release status,
  verify-completion, and research-readiness certification.
- Replay is dynamically bound from `run-result.json`.
- Clean-install certification is wheel-backed, canonical, source-fresh, and all
  seven required surfaces passed with empty reason codes.
- Completion and research-readiness artifacts are fail-closed clean:
  no reason codes, no evidence freshness failures, all policies ready, and no
  unresolved blockers.

Reproducibility evidence accepted by juror:

- Latest matrix is bound to digest
  `repo_checkout_digest:3c9bcef0e9c0ac97cc6215f55a9800709563081e635cd86c1285d1ea3e03c8f2`
  and has `1356 passed`, `0 failed`, `0 skipped`, `0 xfailed`, `0 xpassed`.
- Operator run/replay evidence binds `full-vision-run`, the run-summary hash,
  and verified replay status.
- Current-release and full-vision benchmark evidence are digest-bound and every
  benchmark/replay surface passed.
- Clean install records the source digest at build, wheel digest, surface
  completion `1.0`, and all seven clean-install surfaces passed.

Final wave verdict:

- No unresolved correctness, realism, efficacy, reproducibility, security, or
  release-readiness blockers were identified by the final odd-numbered
  adversarial panel.
- Release-readiness truth is no longer fail-closed blocked for the current
  evidence digest.

### Certification Refresh After Security Revision

Current-release benchmark suite:

`PYTHONPATH=src python3.11 -m euclid benchmarks run --suite current-release.yaml --no-resume --benchmark-root build/certification/current_release_suite`

Result:

- Exit `0`
- Suite: `current_release`

Full-vision benchmark suite:

`PYTHONPATH=src python3.11 -m euclid benchmarks run --suite full-vision.yaml --no-resume --benchmark-root build/certification/full_vision_suite`

Result:

- Exit `0`
- Suite: `full_vision`

Operator run:

`PYTHONPATH=src python3.11 -m euclid run --config examples/full_vision_run.yaml --output-root build/certification/full_vision_run --evidence-report build/reports/full_vision_operator_run_evidence.json`

Result:

- Exit `0`
- Source digest:
  `repo_checkout_digest:31f05b506e7b9d19878df57f4cae67fa1d5ff45cf3be3d6fac8e7e366409e673`
- Run summary SHA:
  `runtime_sha256:4273d65834cf844ae491d9b7d88278ba5bcc30a1c9888a5d6db45dc3affbd9f5`

Operator replay:

`PYTHONPATH=src python3.11 -m euclid replay --run-id full-vision-run --output-root build/certification/full_vision_run --evidence-report build/reports/full_vision_operator_replay_evidence.json`

Result:

- Exit `0`
- Replay verification: `verified`
- Source digest:
  `repo_checkout_digest:31f05b506e7b9d19878df57f4cae67fa1d5ff45cf3be3d6fac8e7e366409e673`

Clean-install certification:

`PYTHONPATH=src python3.11 -m euclid release certify-clean-install --project-root . --wheel-dir build/certification/clean_install/wheels --output-root build/certification/clean_install`

Result:

- Exit `0`
- Surface completion: `1.000000`
- Runtime dependency wheel count: `92`
- Wheel digest:
  `wheel_digest:4ad2da67286984eb38f702bffb940638ebc0f90d34d5fb9a1b705bdb510e2462`
- Source digest:
  `repo_checkout_digest:31f05b506e7b9d19878df57f4cae67fa1d5ff45cf3be3d6fac8e7e366409e673`
- All seven surfaces passed with empty reason codes.

Release status:

`PYTHONPATH=src python3.11 -m euclid release status --project-root .`

Result:

- Exit `0`
- `Target ready: yes`
- Current release verdict: `ready`
- Full vision verdict: `ready`
- Shipped or releasable verdict: `ready`
- All reason-code lists were empty.
- Completion report generated at `2026-05-26T07:19:22Z`.
- Completion values:
  - `current_gate_completion`: `1`
  - `full_vision_completion`: `1`
  - `shipped_releasable_completion`: `1`
- `unresolved_blockers`: `[]`
- Claim truth summary: `ready: true`, `truth_status: ready`,
  zero failed/missing/policy-blocked proofs.

Completion verification:

`PYTHONPATH=src python3.11 -m euclid release verify-completion --project-root .`

Result:

- Exit `0`
- Passed: `yes`

Research-readiness certification:

`PYTHONPATH=src python3.11 -m euclid release certify-research-readiness --project-root .`

Result:

- Exit `0`
- Status: `ready`
- Reason codes: `[]`
- Evidence freshness failures: `[]`
- Full-vision surface status failures: `[]`

Lock and spec verification:

- `UV_CACHE_DIR=/private/tmp/euclid-uv-cache uv lock --check`
  - Exit `0`
  - `Resolved 127 packages in 7ms`
- `PYTHONPATH=src python3.11 -m pytest -q tests/spec_compiler --tb=short`
  - Exit `0`
  - `187 passed in 78.86s (0:01:18)`

### Security Re-Review Follow-Up

Security re-review verdict:

- `REVISE`, confidence `0.82`.
- Prior blockers were confirmed resolved:
  CI derives replay run id from `run-result.json`, Python jobs use
  `uv sync --locked --all-extras --dev`, and source digest includes
  `.github/workflows` plus `uv.lock`.
- Remaining blockers:
  - CI actions were pinned to mutable major tags.
  - Node supply-chain inputs (`package.json`, `package-lock.json`) were outside
    the release source digest while CI runs `npm ci`.

Focused RED:

`PYTHONPATH=src python3.11 -m pytest -q tests/integration/test_completion_regression_ci_contract.py::test_ci_pins_external_actions_to_immutable_shas tests/unit/test_release.py::test_release_source_digest_tracks_ci_workflow_and_dependency_lock --tb=short`

Result:

- Exit `1`
- `2 failed in 2.05s`
- Failures proved:
  - CI still had 15 mutable action refs such as `actions/checkout@v4`.
  - Editing `package.json` did not change the release source digest.

Patch:

- `.github/workflows/ci.yml` action refs now use full commit SHAs:
  - `actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5`
  - `actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065`
  - `actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020`
  - `astral-sh/setup-uv@e58605a9b6da7c637471fab8847a5e5a6b8df081`
  - `actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02`
- `_RELEASE_SOURCE_DIGEST_PATHS` now includes `package.json` and
  `package-lock.json`.
- CI contract tests reject mutable action refs.
- Release digest tests prove `package.json` and `package-lock.json` changes
  alter the release source digest.

Focused GREEN:

`PYTHONPATH=src python3.11 -m pytest -q tests/integration/test_completion_regression_ci_contract.py tests/unit/test_release.py::test_release_source_digest_tracks_ci_workflow_and_dependency_lock --tb=short`

Result:

- Exit `0`
- `3 passed in 1.99s`

Consequence:

- Source-bound release artifacts are stale again because `.github/workflows`,
  source, and tests changed and because package files are now digest inputs.
- Required next step: rerun the full repo matrix and the ordered certification
  evidence chain once more, then rerun security/release-engineering re-review.

### Fresh Matrix After Action-Pinning Revision

Command:

`PYTHONPATH=src python3.11 -m euclid release repo-test-matrix --project-root .`

Result:

- Exit `0`
- Report generated at `2026-05-26T08:08:11Z`
- Source digest:
  `repo_checkout_digest:3c9bcef0e9c0ac97cc6215f55a9800709563081e635cd86c1285d1ea3e03c8f2`
- Summary: `1356 passed, 11 warnings in 1944.78s (0:32:24)`
- Counts:
  - failed: `0`
  - passed: `1356`
  - skipped: `0`
  - xfailed: `0`
  - xpassed: `0`

Whitespace hygiene:

`git diff --check -- .github/workflows/ci.yml src/euclid/release.py tests/integration/test_completion_regression_ci_contract.py tests/unit/test_release.py docs/progress/2026-05-25-full-vision-wave-4.md`

Result:

- Exit `0`

### Closure After Action-Pinning Revision

The required post-action-pinning reruns were completed after the stale-artifact
note above.

Certification chain:

- Current-release benchmark suite: exit `0`; evidence generated at
  `2026-05-26T08:16:15Z`; digest
  `repo_checkout_digest:3c9bcef0e9c0ac97cc6215f55a9800709563081e635cd86c1285d1ea3e03c8f2`;
  all current-release benchmark/replay surfaces passed.
- Full-vision benchmark suite: exit `0`; evidence generated at
  `2026-05-26T08:16:15Z`; digest
  `repo_checkout_digest:3c9bcef0e9c0ac97cc6215f55a9800709563081e635cd86c1285d1ea3e03c8f2`;
  all full-vision benchmark/replay surfaces passed.
- Operator run: exit `0`; run id `full-vision-run`; source digest
  `repo_checkout_digest:3c9bcef0e9c0ac97cc6215f55a9800709563081e635cd86c1285d1ea3e03c8f2`.
- Operator replay: exit `0`; replay verification status `verified`; source
  digest
  `repo_checkout_digest:3c9bcef0e9c0ac97cc6215f55a9800709563081e635cd86c1285d1ea3e03c8f2`.
- Clean-install certification: exit `0`; generated at
  `2026-05-26T08:15:33Z`; source digest at build
  `repo_checkout_digest:3c9bcef0e9c0ac97cc6215f55a9800709563081e635cd86c1285d1ea3e03c8f2`;
  wheel digest
  `wheel_digest:525ad2d3c9f390066b0f2ab6a2276d9f09fccbad93f73a8bf72496e7d5c0bddc`;
  surface completion `1`; all seven surfaces passed.
- Release status: exit `0`; target ready `yes`; current release, full vision,
  and shipped/releasable verdicts all `ready`; all reason-code lists empty.
- Completion verification: exit `0`; passed `yes`.
- Research-readiness certification: exit `0`; status `ready`; reason codes,
  evidence freshness failures, and full-vision surface status failures all empty.
- `UV_CACHE_DIR=/private/tmp/euclid-uv-cache uv lock --check`: exit `0`.
- `PYTHONPATH=src python3.11 -m pytest -q tests/spec_compiler --tb=short`:
  exit `0`; `187 passed in 80.29s (0:01:20)`.

Final odd-numbered independent re-review panel:

- Security/Supply-Chain: `APPROVE`, confidence `0.90`, blockers none.
- Release Engineering/CI/Packaging: `APPROVE`, confidence `0.90`,
  blockers none.
- Reproducibility/Replay/Clean-Install: `APPROVE`, confidence `0.92`,
  blockers none.

Final closure verdict:

- Matrix, benchmark, run, replay, clean-install, completion, research-readiness,
  lock, and spec-compiler evidence all pass for the latest source digest.
- No final-panel juror identified an unresolved correctness, realism, efficacy,
  reproducibility, security, or release-readiness blocker.
