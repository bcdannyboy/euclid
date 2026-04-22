# P16 Documentation Replacement Evidence

Date: 2026-04-21

P16 replaces retained root-document fragments with a live reference-doc set that
matches the implementation, contracts, tests, gates, and release tooling.

Implemented documentation controls:

- The authoritative public entrypoints are `README.md` and `docs/reference/`.
  The reference set covers system shape, runtime CLI behavior, modeling pipeline,
  search core, contracts and manifests, benchmark readiness, workbench evidence,
  and testing truthfulness.
- `schemas/core/euclid-system.yaml`, `schemas/core/source-map.yaml`, and
  `schemas/core/math-source-map.yaml` point to the live reference docs rather
  than removed or retained-slice documents.
- `docs/implementation/authority-reconciliation.yaml` records which prior
  authority fragments are mapped, excluded, or converted into matrix expansion
  rows.
- `docs/implementation/lifecycle-artifact-closure-contract.yaml`,
  `docs/implementation/subtask-test-traceability.yaml`,
  `docs/implementation/enhancement-traceability.yaml`, and `tests/gates/P16.yaml`
  map P16 tasks to code, tests, gates, and evidence.
- Claim-language checks reject unsupported broad claim wording and keep
  benchmark success, live provider success, and docs-only assertions outside the
  claim-evidence lane.

Fixture and golden rationale:

- `fixtures/canonical/golden-pack/expected-build.md` and
  `fixtures/canonical/golden-pack/expected-build.json` were regenerated only
  after the golden pack system manifest was restored to its canonical fixture
  references. The changed output is the deterministic compiler result for that
  corrected fixture root.
- `fixtures/canonical/golden-pack/README.md` and
  `fixtures/canonical/golden-pack/docs/system.md` were added because the golden
  pack fixture declares those docs as required manifest references.
- The contract-graph fixture roots now include `README.md` and `docs/system.md`
  so compiler tests reach the intended graph assertions instead of failing on
  missing example manifests.

Primary evidence:

- `tests/spec_compiler/test_authority_entrypoints.py`
- `tests/spec_compiler/test_authority_reconciliation.py`
- `tests/spec_compiler/test_source_map.py`
- `tests/spec_compiler/test_math_source_map.py`
- `tests/spec_compiler/test_certification_command_contract.py`
- `tests/spec_compiler/test_document_state_labels.py`
- `tests/spec_compiler/test_release_surface_truthfulness.py`
- `tests/spec_compiler/test_documentation_cleanup_register.py`
- `tests/spec_compiler/test_subtask_test_traceability.py`
- `tests/spec_compiler/test_enhancement_gate_traceability.py`
- `tests/spec_compiler/test_compiler_build.py`
- `tests/spec_compiler/test_contract_graph_fixtures.py`
- `tests/spec_compiler/test_canonical_math_fixtures.py`
