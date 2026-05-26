# 2026-05-25 Full-Vision Wave 3 Progress

Goal: continue implementation and validation loops toward 100% Euclid
full-vision readiness without weakening evidence gates or release truth.

Starting evidence:

- Wave 2 cleared the top-level `retained_core_release_failed`,
  `algorithmic_backend_failed`, and `portfolio_orchestration_failed` blockers.
- Fresh `PYTHONPATH=src python3.11 -m euclid release status --project-root .`
  still reported `Target ready: no`.
- Fresh `PYTHONPATH=src python3.11 -m euclid release verify-completion --project-root .`
  exited `1`.
- The Wave 2 seven-agent jury unanimously returned `REVISE`.

Highest-priority Wave 3 blockers:

1. Clean-install `release_status` appears self-blocking: installed-wheel
   `release status` is run before the clean-install report it is expected to
   help prove exists.
2. The certification command contract declares
   `build/certification/full_vision_run/run-result.json`, but current runtime
   evidence is centered on sealed run summaries and does not prove that
   artifact exists.
3. Full-vision operator run/replay evidence must be regenerated with source
   digest, run-summary digest, run-id binding, replay-result digest, and
   replay-to-run binding.
4. Repo-test matrix evidence is stale and failed; at least one benchmark test
   contract still expects retained core and algorithmic current-release
   surfaces to fail.
5. Evidence lanes for boundary-specific external evidence, descriptive
   compression, predictive generalization, readiness/closure, replay
   verification, and robustness remain incomplete.
6. Claim-surface presentation must make blockers harder to miss than passed
   gate summaries.
7. Math realism still needs stronger provenance binding, positive descriptive
   compression evidence, more diverse search-class fixtures, CIR hash
   validation, and reducer/portfolio semantic tightening.

## Planning Swarm

Planning swarm size: 5 read-only agents.

Agents:

- `019e61e4-efea-7a61-859f-899147bd9d5f`
- `019e61e5-0a08-7353-9d7d-17487594ab40`
- `019e61e5-2249-7493-b1b5-28d859c7de55`
- `019e61e5-37d3-7792-b29b-929f87a999d9`
- `019e61e5-4a8a-7272-8335-e3f783fbed71`

Prompt: design the next disjoint implementation swarm, including file
ownership, TDD red tests, validation commands, risks, and local-only tasks.

Status: running.

### Planning Swarm Synthesis

Planning result: all collected planners recommend separating the next wave into
clean-install release truth, operator run/replay evidence contract, repo-matrix
and benchmark drift, evidence-lane closure, claim-surface presentation,
mathematical realism, and fixture/provenance realism.

Implementation slice selected for this coordinator wave:

1. Clean-install release-status semantics.
2. Operator run/replay command-contract artifact emission.
3. Current-release benchmark expectation drift.

Reason for narrowing the active implementation slice:

- These are the highest-priority P0 blockers that prevent evidence
  regeneration from even becoming meaningful.
- They have mostly disjoint write sets.
- Evidence-lane and mathematical-realism closure remain active blockers, but
  they depend on the run/replay and release-evidence contract being stable.

Guardrails:

- Do not make clean install green by redefining final release readiness.
- Do not remove or weaken `release status` fail-closed behavior.
- Do not treat benchmark execution as claim evidence.
- Do not regenerate or bless `build/reports/**` until source churn stops.

Actual implementation agents:

- Clean-install release-status worker: `019e61ec-bd44-7f22-934f-09c8fc589d43`
- Operator run/replay contract worker: `019e61ec-f3b5-7832-99fa-f453237d834b`
- Benchmark expectation drift worker: `019e61ed-3176-7313-94ae-0bb078f2decd`

## Implementation Results

### Clean-Install Release-Status Semantics

Agent: `019e61ec-bd44-7f22-934f-09c8fc589d43`

Changed files:

- `src/euclid/release.py`
- `tests/unit/test_release.py`

Patch summary:

- The clean-install `release_status` surface now validates the installed
  `release status` command as a coherent policy snapshot instead of requiring
  final target readiness before the clean-install report can be written.
- It accepts both `Target ready: yes` with ready policy verdicts and
  `Target ready: no` with truthful blocked/review-required policy verdicts and
  non-empty reason-code lines.
- It fails malformed/missing policy verdict lines and blocked policies that
  report no reason codes.
- This does not make final release readiness easier; final readiness remains
  governed by policy judgments, evidence freshness, clean-install report
  freshness, and completion verification.

Worker red evidence:

- `pytest tests/unit/test_release.py::test_clean_install_release_status_surface_passes_when_status_is_truthfully_blocked tests/unit/test_release.py::test_clean_install_release_status_surface_fails_when_policy_verdict_lines_are_missing_or_malformed -q`
  reported `3 failed in 2.39s`.

Coordinator verification:

- `PYTHONPATH=src python3.11 -m pytest -q tests/unit/test_release.py::test_clean_install_release_status_surface_passes_when_status_is_truthfully_blocked tests/unit/test_release.py::test_clean_install_release_status_surface_fails_when_policy_verdict_lines_are_missing_or_malformed`
  reported `3 passed in 1.14s`.

### Operator Run/Replay Contract Artifact

Agent: `019e61ec-f3b5-7832-99fa-f453237d834b`

Changed files:

- `src/euclid/cli/run.py`
- `src/euclid/operator_runtime/evidence.py`
- `tests/integration/test_operator_run_pipeline.py`

Patch summary:

- `euclid run` now writes the declared `<output_root>/run-result.json`
  artifact after a successful operator run.
- The artifact records `run_id`, `run_summary_path`, `run_summary_sha256`,
  `run_result_ref`, `bundle_ref`, and `output_root`.
- The artifact aligns with `scripts/release_smoke.sh` and
  `docs/implementation/certification-command-contract.yaml`, which derive the
  replay run id from `build/certification/full_vision_run/run-result.json`.

Worker red evidence:

- `PYTHONPATH=src pytest tests/integration/test_operator_run_pipeline.py::test_operator_run_with_evidence_report_writes_declared_run_result_artifact -q`
  failed with missing `<tmp>/operator-run/run-result.json`; result
  `1 failed in 1.97s`.

Coordinator verification:

- `PYTHONPATH=src python3.11 -m pytest -q tests/integration/test_operator_run_pipeline.py tests/integration/test_operator_replay_pipeline.py`
  reported `11 passed in 2.96s`.
- `PYTHONPATH=src python3.11 -m pytest -q tests/spec_compiler/test_certification_command_contract.py tests/spec_compiler/test_release_surface_truthfulness.py`
  reported `8 passed in 0.08s`.
- `bash -n scripts/release_smoke.sh` exited `0`.

### Current-Release Benchmark Expectation Drift

Agent: `019e61ed-3176-7313-94ae-0bb078f2decd`

Changed file:

- `tests/benchmarks/test_suite_runner.py`

Patch summary:

- Updated the current-release suite test to expect
  `retained_core_release`, `algorithmic_backend`,
  `shared_plus_local_decomposition`, and `mechanistic_lane` as passed.
- Updated the installed-project-root suite expectation to explicitly require
  retained-core pass.

Worker red evidence:

- `PYTHONPATH=src python3.11 -m pytest -q tests/benchmarks/test_suite_runner.py::test_current_release_suite_uses_canonical_active_scope_name --tb=short`
  failed because retained core and algorithmic backend now pass but the test
  still expected failure; result `1 failed in 35.33s`.

Coordinator verification:

- `PYTHONPATH=src python3.11 -m pytest -q tests/benchmarks/test_suite_runner.py --tb=short`
  reported `3 passed in 37.53s`.

### Cross-Slice Hygiene

- `git diff --check -- src/euclid/release.py tests/unit/test_release.py src/euclid/cli/run.py src/euclid/operator_runtime/evidence.py tests/integration/test_operator_run_pipeline.py tests/benchmarks/test_suite_runner.py`
  exited `0`.

### Clean-Install Output Root Contract Repair

Root cause:

- The frozen command contract and `scripts/release_smoke.sh` use
  `build/certification/clean_install`.
- The clean-install output-root validator only accepted path segments
  containing `clean-install`, so it rejected the contract path before any
  certifier work could start.

TDD red:

- Added
  `tests/unit/test_release.py::test_clean_install_certification_accepts_command_contract_output_root`.
- `PYTHONPATH=src python3.11 -m pytest -q tests/unit/test_release.py::test_clean_install_certification_accepts_command_contract_output_root`
  failed because `_clean_install_work_root_dedication_failure(...)` returned
  `not_dedicated`.

Patch:

- `src/euclid/release.py` now normalizes underscores to hyphens when checking
  whether an output-root path segment identifies a clean-install work root.
- Reserved roots such as `build` and `build/reports` remain rejected.

Verification:

- `PYTHONPATH=src python3.11 -m pytest -q tests/unit/test_release.py::test_clean_install_certification_accepts_command_contract_output_root tests/unit/test_release.py::test_clean_install_certification_rejects_reserved_output_root_before_delete tests/unit/test_release.py::test_release_evidence_freshness_rejects_clean_install_reserved_output_root`
  reported `5 passed in 1.01s`.
- `git diff --check -- src/euclid/release.py tests/unit/test_release.py`
  exited `0`.

### Certification Artifact Regeneration

Operator run evidence:

- `PYTHONPATH=src python3.11 -m euclid run --config examples/full_vision_run.yaml --output-root build/certification/full_vision_run --evidence-report build/reports/full_vision_operator_run_evidence.json`
  exited `0`.
- `build/certification/full_vision_run/run-result.json` now exists and records
  `run_id: full-vision-run`, the sealed run-summary digest, run-result ref,
  bundle ref, and output root.

Operator replay evidence:

- `PYTHONPATH=src python3.11 -m euclid replay --run-id full-vision-run --output-root build/certification/full_vision_run --evidence-report build/reports/full_vision_operator_replay_evidence.json`
  exited `0`.
- Replay verification reported `verified`.
- Replay evidence records `replay_result_sha256`, run-id binding, and
  replay-to-run evidence binding.

Clean-install certification:

- Initial run of
  `PYTHONPATH=src python3.11 -m euclid release certify-clean-install --project-root . --wheel-dir build/certification/clean_install/wheels --output-root build/certification/clean_install`
  failed before build work because `clean_install` was rejected as
  `not_dedicated`.
- After the output-root validator fix, the same command exited `0`.
- Result:
  - `Surface completion: 1.000000`
  - `release_status: passed`
  - `operator_run: passed`
  - `operator_replay: passed`
  - `determinism_same_seed: passed`
  - `performance_runtime_smoke: passed`
  - `packaged_notebook_smoke: passed`
  - `benchmark_execution: passed`

Important interpretation:

- The installed-wheel `release status` log is still blocked, which is correct.
  It runs from the clean-install `outside-repo` directory and has no outer
  certification bundle.
- The clean-install `release_status` surface now proves that the installed
  command runs and reports a coherent fail-closed policy snapshot. It does not
  prove final repo readiness by itself.

### Post-Regeneration Release Status

After regenerating operator and clean-install evidence, top-level release status:

Command:

`PYTHONPATH=src python3.11 -m euclid release status --project-root .`

Result: exit `0`; target ready remains `no`.

Cleared in Wave 3:

- `clean_install_surface.release_status_failed`
- `release.evidence_freshness_clean_install_source_digest_mismatch`
- `release.evidence_freshness_clean_install_surface_release_status_not_passed`
- `release.evidence_freshness_clean_install_surface_release_status_reason_codes_present`
- all full-vision operator run/replay source digest, summary digest, run-id
  binding, replay-result digest, and replay-to-run binding failures.

Remaining top-level blockers:

- `release.evidence_freshness_repo_test_matrix_source_digest_mismatch`
- `evidence_lane.boundary_specific_external_evidence_missing`
- `evidence_lane.descriptive_compression_missing`
- `evidence_lane.predictive_generalization_missing`
- `evidence_lane.readiness_and_closure_missing`
- `evidence_lane.replay_verification_missing`
- `evidence_lane.robustness_missing`

Post-regeneration completion verification:

- `PYTHONPATH=src python3.11 -m euclid release verify-completion --project-root .`
  exited `1`.
- Remaining failures:
  - `current_release_v1`, `full_vision_v1`, and `shipped_releasable_v1`
    remain blocked past the transition window ending `2026-05-15T00:00:00Z`.
  - Completion report has incomplete rows for boundary-specific external
    evidence, descriptive compression, predictive generalization,
    readiness/closure, replay verification, and robustness.
  - Completion report still contains unresolved blockers.
  - Release evidence freshness failure remains:
    `repo_test_matrix_source_digest_mismatch`.

## Post-Wave Independent Jury

Jury size: 5 independent read-only jurors.

Verdict split:

- `APPROVE` Wave 3 release-pipeline mechanics: 2
- `REVISE` overall readiness / remaining blockers: 3

Important: approvals were explicitly limited to Wave 3 mechanics. No juror
approved final release readiness.

Jurors:

- Mathematics, symbolic regression, CIR/reducer:
  `019e6203-5a67-7e90-afd1-d23191708bc7` -> `REVISE`
- Statistics, forecasting, probabilistic support, calibration:
  `019e6203-71bf-7472-9bea-df9f44aa2e16` -> `APPROVE`
- Reproducibility, benchmark, replay:
  `019e6203-87c3-77c2-8f27-d2102ab5392d` -> `REVISE`
- Software architecture, release engineering, packaging:
  `019e6203-9e41-7173-a9c2-47e1fcb52870` -> `APPROVE`
- Security, supply chain, UX, claim-surface truth:
  `019e6203-bea1-7a70-ba4f-316e175f70f0` -> `REVISE`

Shared conclusions:

- Wave 3 improved the release pipeline without weakening readiness gates.
- Clean-install `release_status` is now a valid installed-command truth surface,
  but clean-install pass is not release readiness.
- Operator run/replay evidence now has the declared run-result artifact,
  run-summary digests, replay digest, and replay-to-run evidence binding.
- Release status remains fail-closed and target ready remains `no`.
- The repo-test matrix is stale and failed relative to current source.
- The six evidence lanes remain real blockers and must be closed by executable
  evidence, not by reclassifying benchmark execution as claim evidence.

Additional jury findings:

- Some integration tests still appear stale against the new clean-install
  semantics, especially tests that expect `certify-clean-install` to exit `1`
  after a 7/7 clean-install surface pass.
- `build/reports/research-readiness.json` is stale relative to Wave 3 and still
  reflects pre-Wave-3 clean-install/operator evidence failures.
- Release source digest coverage should include every claim-facing document
  cited as policy/evidence, including `README.md` and `docs/reference/**`.
- Clean-install CLI output can be overread because it prints
  `Surface completion: 1.000000` and exits `0`; it should explicitly state
  that this is runtime install certification and not final release readiness.
- The local wheelhouse smoke is useful but not a full supply-chain provenance
  proof because it repacks ambient installed distributions.
- Workbench and report surfaces should expose global release readiness blocked
  wherever local publishability or claim publication language appears.
- Search-class provenance and compact-law evidence still need stronger
  mathematical realism evidence.

Next wave priority:

1. Update stale tests that still expect clean-install certification failure.
2. Run `PYTHONPATH=src python3.11 -m euclid release repo-test-matrix --project-root .`.
3. Fix any fresh repo-test matrix failures without weakening gates.
4. Close evidence lanes with governance/golden/replay/package evidence:
   boundary-specific external evidence, descriptive compression, predictive
   generalization, readiness/closure, replay verification, and robustness.
5. Regenerate `release status`, `verify-completion`, and
   `certify-research-readiness`.
6. Improve claim-surface and clean-install CLI wording so runtime-install
   certification cannot be mistaken for final readiness.
