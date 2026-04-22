# P15 Workbench Evidence Studio Evidence

Date: 2026-04-21

P15 adds a sanitized workbench evidence surface and tightens live API gate
artifacts without allowing provider success to become scientific claim evidence.

Implemented controls:

- Workbench normalization now emits an `evidence_studio` object with claim lane,
  claim ceiling, downgrade reasons, abstention reasons, replay artifact links,
  engine provenance, fitting diagnostics, scoring diagnostics, and falsification
  diagnostics.
- Live gate evidence now carries `claim_evidence_status: not_claim_evidence` and
  a `claim_boundary` that marks live provider success as non-claim evidence.
- Workbench live evidence is sanitized before display, handles missing and
  malformed evidence as unavailable/malformed surfaces, and preserves the same
  non-claim boundary used by live gate artifacts.
- The workbench Evidence rail renders the Evidence Studio summary so claim lane,
  replay count, engine provenance, and live non-claim status are visible in the
  UI.
- Provider failures are mapped semantically: invalid credentials, rate limits,
  timeouts, malformed or empty payloads, and missing or blank credentials fail
  closed under strict mode.

Fixture and golden rationale:

- No golden fixture was updated in this P15 pass. The output change is a new
  additive evidence surface and explicit live-evidence boundary, so focused unit
  and frontend assertions cover the behavior without snapshot churn.
- The frontend workbench harness now loads the checked-in
  `tests/frontend/workbench/fixtures/analysis-saved.json` fixture instead of a
  generated `build/workbench/.../analysis.json` path. This makes the Evidence
  Studio UI check replayable from a clean checkout.
- All secret-bearing examples in tests use fake fixture values. The implementation
  reports live key presence only through `EuclidEnv.presence_metadata` and never
  stores key values in evidence artifacts.

Primary evidence:

- `tests/unit/testing/test_live_api_gate.py`
- `tests/unit/testing/test_secret_redaction.py`
- `tests/regression/test_live_api_evidence_redaction.py`
- `tests/unit/workbench/test_evidence_studio.py`
- `tests/frontend/workbench-ui.test.js`
- `tests/live/test_fmp_live_ingestion_smoke.py`
- `tests/live/test_openai_live_explainer_smoke.py`
- `tests/live/test_dependency_runtime_live.py`
