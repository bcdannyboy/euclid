# Live API Test Policy

Euclid live API tests are explicit release gates, not hidden side effects.

Live gates load credentials only through `src/euclid/runtime/env.py`. Local `.env`
files are allowed, but `.env` is ignored by git and must never be committed.
CI may inject the same variables through the process environment.

Required variables:

- `EUCLID_LIVE_API_TESTS`: enables live tests when set to `1`, `true`, `yes`, or `on`.
- `EUCLID_LIVE_API_STRICT`: fails missing credentials instead of skipping when enabled.
- `FMP_API_KEY`: used only for live FMP ordered-observation checks.
- `OPENAI_API_KEY`: used only for live workbench explanation checks.
- `EUCLID_OPENAI_EXPLAINER_MODEL`: optional workbench explanation model override.
- `EUCLID_LIVE_TEST_TIMEOUT_SECONDS`: optional timeout override.
- `EUCLID_LIVE_ARTIFACT_DIR`: optional directory for sanitized live evidence.

Live evidence may record provider names, endpoint classes, timestamps, schema
versions, row counts, semantic pass/fail reason codes, and latency. It must not
record API keys, authorization headers, raw secret-bearing URLs, prompt bodies
that contain secrets, or provider payloads that licensing forbids storing.

Fixture tests must cover missing keys, invalid keys, disabled live mode, strict
mode, rate limits, malformed payloads, redaction, and regression artifacts.

Removal note: direct provider-key reads from runtime surfaces are replaced by
`EuclidEnv`. Code may accept a key explicitly from a caller, but default runtime
lookup must go through the approved loader so `.env` and CI injection are audited
the same way.
