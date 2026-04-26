# Workbench

The workbench is Euclid’s local analysis studio: the fastest way to see what the system actually found on a time series, what claim level survived review, and what evidence kept stronger claims out. It brings deterministic, probabilistic, benchmark, and artifact outputs into one place, and it is where Euclid’s descriptive-to-predictive story becomes legible on a real run.

When the evidence is strong enough, the workbench can surface one joined deterministic-plus-stochastic equation. When it is not, it deliberately stops lower on the ladder and shows the reason codes instead of upgrading the claim by implication.

Its source lives in:

- `src/euclid/workbench/server.py`
- `src/euclid/workbench/service.py`
- `src/euclid/workbench/explainer.py`
- `src/euclid/_assets/workbench/index.html`
- `src/euclid/_assets/workbench/app.css`
- `src/euclid/_assets/workbench/app.js`

## API surface

The HTTP server is a `ThreadingHTTPServer` in `server.py`.

Routes:

- `GET /`
- `GET /index.html`
- `GET /app.css`
- `GET /app.js`
- `GET /vendor/*`
- `GET /api/config`
- `GET /api/analysis?analysis_path=...`
- `POST /api/analyze`

`/api/config` returns target specs, default dates, recent analyses, and env-backed API key state.

`/api/analysis` reloads a saved analysis and normalizes it before returning it.

`/api/analyze` validates input, resolves API keys, runs `create_workbench_analysis(...)`, normalizes the result, and attaches cached or generated explanations.

## Analysis flow

`create_workbench_analysis(...)` in `src/euclid/workbench/service.py`:

1. fetches FMP history
2. requires explicit date range selection
3. transforms the chosen target
4. runs point, probabilistic, and benchmark analysis lanes
5. writes workspace artifacts and output-root-local `analysis.json`

`normalize_analysis_payload(...)` then does the work that matters for interpretation: it rebuilds the saved run into a conservative UI-facing claim taxonomy. The workbench does not simply echo raw payloads back to the browser. It keeps only the surfaces that still clear structural and publication gates after normalization:

- `descriptive_fit`: a benchmark-local symbolic fit when the descriptive floor survives normalization
- `descriptive_reconstruction`: an explicit-time descriptive-only reconstruction of the series
- `predictive_law`: a point-lane symbolic law only when the operator publication path is complete, publishable, non-abstaining, predictive-capable, and backed by a curve-bearing equation plus complete evidence refs
- `holistic_equation`: a backend-authored joint deterministic-plus-stochastic claim only when the deterministic side is the surviving `predictive_law`, the probabilistic side is publishable, and both sides share the same validation scope and publication record
- `uncertainty_attachment`: the retained probabilistic companion only when it matches that same publishable joint claim
- `gap_report`, `not_holistic_because`, and `would_have_abstained_because`: explicit reason-code summaries for why the run stopped short of a stronger claim
- benchmark summaries and change-atlas views for comparative and local-shape inspection

`predictive_law` is therefore already a gated object, not a raw symbolic fit. It survives only when the point lane finished as `candidate_publication`, the publication status is `publishable`, no abstention artifact is present, the claim card remains predictive-capable and `confirmatory_supported`, both descriptive and predictive scorecard states passed, the allowed interpretation codes include predictive publication, and the equation does not depend on banned paths such as exact sample closure, residual wrappers, or posthoc symbolic synthesis.

`holistic_equation` is the highest-claim symbolic surface in the taxonomy. It survives only when the backend has already accepted a joint claim gate, the payload is not stale synthetic metadata, the deterministic source is `predictive_law`, the referenced probabilistic lane is completed and publishable, and both lanes point at the exact same `validation_scope_ref` and `publication_record_ref`. If those refs do not line up, the workbench withholds the unified equation.

## Target semantics

Built-in targets include:

- `price_close`
- `daily_return`
- `log_return`
- `next_day_up`

The code treats `daily_return` as the recommended default when the goal is interpretable descriptive structure rather than raw level tracking. That recommendation is explicit in `service.py`: daily returns make persistence visible instead of letting it hide inside raw levels, so descriptive equations have to earn their structure.

The other targets stay useful, but they answer different questions:

- `price_close` is helpful for replaying level paths, but descriptive equations on raw close often collapse into persistence-and-drift summaries.
- `log_return` is the scale-invariant return view when additive return algebra matters.
- `next_day_up` is a directional target for event-style forecasting and does not support descriptive path equations.

## Conservative interpretation

The workbench is designed to avoid overclaiming. Backend normalization and frontend rendering both treat claim promotion as something to be re-earned at display time, not something the UI should infer from a pretty curve.

In practice that means the app refuses to promote:

- stale synthetic holistic payloads
- exact-closure artifacts as predictive-within-scope symbolic claims or unified equations
- non-publishable saved outputs
- benchmark-local descriptive summaries into operator-level claims
- probabilistic lanes whose calibration status does not have a `publishable` gate effect
- runs with abstention or incomplete evidence refs into higher symbolic claim classes

This logic is split between `src/euclid/workbench/service.py`, which normalizes and classifies the payload, and `src/euclid/_assets/workbench/app.js`, which rechecks whether a card, overlay, or headline is allowed to present itself as authoritative.

## Frontend behavior

The UI is a single packaged app shell with tabs for:

- Overview
- Workspace
- Point
- Probabilistic
- Benchmark
- Artifacts

The internal key for the Workspace tab is `atlas`.

The overview intentionally starts from the strongest surviving mathematical object and then fans out into operator, probabilistic, benchmark, and artifact detail. If no publishable law survives, the interface falls back to descriptive reconstruction or benchmark-local context instead of implying a predictive result.

The app synchronizes URL state for:

- `analysis_path`
- `tab`
- `horizon`
- `overlay`
- `lane`
- `rail`

It also has explicit busy, failure, no-winner, and explainer-fallback states.

## Probabilistic evidence display

The probabilistic tab shows production stochastic evidence instead of assuming every distribution is location plus scale. When family-derived intervals or configured quantiles are present, the band display uses those values before falling back to Gaussian location-scale compatibility. The lane details surface stochastic model manifest refs, residual history refs, family, calibration bins, lane status, and downgrade reasons so a passed calibration badge cannot hide thin evidence or a heuristic Gaussian compatibility downgrade.

## Explainer flow

`src/euclid/workbench/explainer.py` builds page-scoped snapshots and uses the OpenAI Responses API to create structured explanation bundles when `OPENAI_API_KEY` is available. Without that key, it records `status="unavailable"` with `reason_code="missing_openai_api_key"` and leaves the rest of the workbench usable. It retries with a more compact request when the first pass times out or truncates.

The backend generates explainer pages for:

- overview
- point
- probabilistic
- benchmark
- artifacts

The Workspace guide is supplied by the frontend so it can stay aligned with the atlas-specific overlays, normalization warnings, and claim-gating cues shown in the browser.
