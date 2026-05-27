# 2026-05-26 Euclid Math + UX Improvement Swarm

## Scope

Goal: use independent subagent swarms to identify the highest-leverage ways to improve Euclid only along two axes:

1. Mathematical quality: more mathematically realistic, principled, expressive, and honest as a system for deriving compact laws from ordered observations.
2. User experience and understanding: clearer results, limits, evidence, uncertainty, and reasoning, especially in the workbench.

Out of scope unless directly tied to mathematical meaning or user comprehension: security, packaging, CI, release engineering, dependency hygiene, and generic infrastructure.

Operating mode: read-only analysis except for this ledger. Do not print secrets from `.env`.

## Baseline

- Date: 2026-05-26
- Repository: `/Users/danielbloom/Desktop/euclid`
- Branch: `codex-full-vision-wave-1`
- Commit: `c0374b10ad10ded57ac750938c0a738b7d5e7ae0`
- Baseline status before creating this ledger: clean (`git status --short` produced no output; `git status --branch --short` produced only `## codex-full-vision-wave-1`)

## Commands Run

Commands are listed in chronological order as closely as possible.

```bash
sed -n '1,220p' /Users/danielbloom/.codex/superpowers/skills/using-superpowers/SKILL.md
sed -n '1,240p' /Users/danielbloom/.codex/skills/dynamic-subagent-swarm/SKILL.md
rg -n "full-vision|docs/progress|mathematical|workbench|release status|target ready|cir/models|adapters/portfolio|full-vision-wave" /Users/danielbloom/.codex/memories/MEMORY.md
git status --short
git rev-parse HEAD
pwd
test -f docs/progress/2026-05-26-euclid-math-ux-improvement-swarm.md && sed -n '1,220p' docs/progress/2026-05-26-euclid-math-ux-improvement-swarm.md || true
rg --files | sed -n '1,220p'
find docs -maxdepth 2 -type f | sort | sed -n '1,220p'
find frontend -maxdepth 3 -type f | sort | sed -n '1,220p'
git status --branch --short
git branch --show-current
find . -maxdepth 3 -type f \( -name 'README.md' -o -name 'package.json' -o -name 'pyproject.toml' -o -name 'pytest.ini' -o -name 'vite.config.*' \) | sort
nl -ba README.md | sed -n '1,240p'
nl -ba docs/math.md | sed -n '1,260p'
nl -ba docs/system.md | sed -n '1,260p'
nl -ba docs/workbench.md | sed -n '1,260p'
nl -ba src/euclid/workbench/service.py | sed -n '1,280p'
nl -ba src/euclid/workbench/explainer.py | sed -n '1,280p'
find src/euclid/_assets/workbench -maxdepth 2 -type f | sort
nl -ba docs/reference/search-core.md | sed -n '1,260p'
nl -ba docs/reference/testing-truthfulness.md | sed -n '1,260p'
nl -ba src/euclid/search/descriptive_coding.py | sed -n '1,260p'
nl -ba src/euclid/math/codelength.py | sed -n '1,260p'
nl -ba src/euclid/cir/models.py | sed -n '1,260p'
nl -ba src/euclid/search/descriptive_coding.py | sed -n '260,620p'
nl -ba src/euclid/math/codelength.py | sed -n '253,620p'
nl -ba src/euclid/cir/models.py | sed -n '242,560p'
nl -ba src/euclid/modules/claims.py | sed -n '1,320p'
nl -ba src/euclid/modules/calibration.py | sed -n '1,320p'
nl -ba src/euclid/modules/effective_sample.py | sed -n '1,260p'
rg -n "claim|descriptive|predictive|holistic|uncertainty|evidence|abstain|calibration|workspace|atlas|explain|stale|synthetic|publish" src/euclid/_assets/workbench/app.js
nl -ba src/euclid/_assets/workbench/index.html | sed -n '1,260p'
nl -ba src/euclid/_assets/workbench/app.css | sed -n '1,260p'
nl -ba src/euclid/workbench/server.py | sed -n '1,300p'
find tests -maxdepth 3 -type f | sort | rg "workbench|math|cir|search|claims|calibration|effective_sample|descriptive|readiness"
sed -n '1,220p' /Users/danielbloom/.codex/plugins/cache/openai-bundled/browser/26.519.41501/skills/browser/SKILL.md
rg -n "workbench|serve|output-root|port|host" src/euclid/cli src/euclid/__main__.py src/euclid/cli/workbench.py
nl -ba src/euclid/cli/workbench.py | sed -n '1,240p'
nl -ba pyproject.toml | sed -n '1,240p'
sed -n '221,520p' /Users/danielbloom/.codex/plugins/cache/openai-bundled/browser/26.519.41501/skills/browser/SKILL.md
sed -n '521,900p' /Users/danielbloom/.codex/plugins/cache/openai-bundled/browser/26.519.41501/skills/browser/SKILL.md
PYTHONPATH=src python3.11 -m euclid workbench serve --host 127.0.0.1 --port 8765 --output-root /tmp/euclid-workbench-inspection --project-root . --no-open-browser
curl -sS http://127.0.0.1:8765/api/config
nl -ba package.json | sed -n '1,240p'
find node_modules -maxdepth 2 -type d -name playwright -o -name '@playwright' | sort
ls tests/frontend
sed -n '1,80p' .playwright-mcp/console-2026-05-26T11-03-11-631Z.log
sed -n '1,220p' .playwright-mcp/page-2026-05-26T11-03-11-790Z.yml
rg -n "def _env_value|EuclidEnv|OPENAI_API_KEY|load" src/euclid/workbench/server.py src/euclid/runtime/env.py
nl -ba src/euclid/workbench/server.py | sed -n '300,390p'
nl -ba src/euclid/runtime/env.py | sed -n '1,130p'
OPENAI_API_KEY= PYTHONPATH=src python3.11 -m euclid workbench serve --host 127.0.0.1 --port 8765 --output-root /tmp/euclid-workbench-inspection-no-llm --project-root . --no-open-browser
curl -sS http://127.0.0.1:8765/api/config
curl -sS --max-time 240 -X POST http://127.0.0.1:8765/api/analyze -H 'Content-Type: application/json' --data '{"symbol":"SPY","target_id":"daily_return","start_date":"2026-04-27","end_date":"2026-05-22","include_probabilistic":true,"include_benchmark":true}' > /tmp/euclid-workbench-live-analysis.json
jq '{error: .error, symbol: .dataset.symbol, target: .dataset.target, rows: .dataset.rows, claim_class: .claim_class, publishable: .publishable, point_publication: .operator_point.publication, descriptive_status: .descriptive_fit.status, probabilistic_keys: (.probabilistic | keys), analysis_path: .analysis_path, workspace_root: .workspace_root, llm: .llm_explanations}' /tmp/euclid-workbench-live-analysis.json
jq -r '.analysis_path // empty' /tmp/euclid-workbench-live-analysis.json
jq -r '.not_holistic_because[]? , .would_have_abstained_because[]?' /tmp/euclid-workbench-live-analysis.json | sed -n '1,80p'
sed -n '1,260p' .playwright-mcp/page-2026-05-26T11-04-36-051Z.yml
find . -maxdepth 2 -type f -name 'euclid-workbench-*.png' -o -path './.playwright-mcp/*' | sort
sed -n '1,260p' .playwright-mcp/page-2026-05-26-loaded-overview.yml
git status --short --branch
sed -n '1,360p' .playwright-mcp/page-2026-05-26-loaded-overview-depth6.yml
sed -n '1,320p' .playwright-mcp/page-2026-05-26-loaded-evidence-depth6.yml
sed -n '1,320p' .playwright-mcp/page-2026-05-26-loaded-calibration-depth6.yml
jq '.probabilistic | to_entries[] | {key, status: .value.status, evidence: .value.evidence, calibration: .value.calibration}' /tmp/euclid-workbench-live-analysis.json
jq '.probabilistic | to_entries[] | {key, payload: .value}' /tmp/euclid-workbench-live-analysis.json | sed -n '1,220p'
rg -n "description_gain_non_positive|unknown_evidence_reason_code|reason_codes" src schemas tests | sed -n '1,220p'
nl -ba src/euclid/testing/gate_manifest.py | sed -n '1,260p'
nl -ba src/euclid/modules/probabilistic_evaluation.py | sed -n '1,360p'
nl -ba src/euclid/modules/evidence_contracts.py | sed -n '1,300p'
rg -n "Evidence|allowed_reason_codes|stochastic|description_gain_non_positive|reason_codes" src/euclid/modules src/euclid/workbench/service.py | sed -n '1,260p'
```

Notable command observations:

- No top-level `frontend/` directory exists.
- `package.json`, `pyproject.toml`, `src/euclid/workbench/*`, and workbench docs exist, so workbench inspection should start from server/service/docs and any local UI assets exposed there.
- In-app Browser setup reported `Browser is not available: iab`; Playwright MCP was used as a fallback for local workbench snapshots and screenshots.
- Workbench server was run with `/tmp` output roots. The second server run explicitly set `OPENAI_API_KEY=` so the live SPY workbench payload would not call the OpenAI explainer while still using FMP key presence from `.env`.
- Local live workbench run: SPY `daily_return`, `2026-04-27` to `2026-05-22`, 19 transformed rows, `claim_class=descriptive_reconstruction`, `publishable=false`, point publication `abstained`, no benchmark-local winner.
- Screenshot files captured in the repo root by the browser fallback: `euclid-workbench-empty-overview-2026-05-26.png`, `euclid-workbench-loaded-overview-2026-05-26.png`, `euclid-workbench-loaded-evidence-2026-05-26.png`, `euclid-workbench-loaded-calibration-2026-05-26.png`.
- Browser fallback also wrote snapshots/logs under `.playwright-mcp/`. This is generated evidence from inspection, not source work.

## Subagent Roster and Actual Counts

### Wave 1: Planning Swarm

- Requested count: 5 independent planning subagents.
- Actual count: 5.
- Planner 1: `019e63f1-b719-7bd3-9bad-afe4f5ec7f3f` (`Schrodinger`) - MDL/search/statistical/claim/workbench/explanation plan.
- Planner 2: `019e63f1-cf40-7871-87a3-ad4ed1b1f73e` (`Confucius`) - CIR/search/reducer/MDL/planted-law/workbench/claim-language plan.
- Planner 3: `019e63f1-ebb9-7b33-b2f7-8301e14649a7` (`Noether`) - statistical evidence/calibration/N_eff-led plan.
- Planner 4: `019e63f2-0b84-75c1-b4ef-7d7b38d4573e` (`Poincare`) - workbench IA/visual hierarchy/user mental model plan.
- Planner 5: `019e63f2-23dc-74d0-8279-329eedb46736` (`Kepler`) - scientific communication and analyst mental-model plan.

### Wave 2: Math + UX Analysis Swarm

- Requested preferred count: 7 independent subagents.
- Actual count: 7 total roles completed.
- Actual simultaneous capacity observed: 6. The first Wave 2 spawn attempt hit the agent thread limit after one role was created; completed Wave 1 planners were closed, then six analysis agents ran concurrently. The seventh role was launched after a slot freed.
- Requested roles:
  1. Mathematical Foundations Reviewer
  2. Search and Representation Reviewer
  3. Statistical Meaning Reviewer
  4. Claim Semantics Reviewer
  5. Workbench UX Reviewer
  6. Explanation and Narrative Reviewer
  7. User Mental Model Reviewer
- Actual roster:
  1. Mathematical Foundations Reviewer: `019e63f7-edb5-7981-a2b6-112b491aae20` (`Anscombe`)
  2. Search and Representation Reviewer: `019e63f8-9c0a-7af1-8eb5-3626be7657b3` (`Galileo`)
  3. Statistical Meaning Reviewer: `019e63f8-c112-7862-9066-3797b1f39090` (`Zeno`)
  4. Claim Semantics Reviewer: `019e63f8-e453-75d2-ae14-ff3ea0f525b3` (`Bacon`)
  5. Workbench UX Reviewer: `019e63f9-0f9a-7ee2-b596-92e4086f80b1` (`Arendt`)
  6. Explanation and Narrative Reviewer: `019e63f9-36eb-7d00-ae39-043c61c83d23` (`Pauli`)
  7. User Mental Model Reviewer: `019e63fc-892c-7b62-b70d-88e416d1e8b7` (`Franklin`)

### Jury

- Requested count: 5 independent jurors.
- Actual count: 5.
- Requested roles:
  1. Pure/Applied Mathematics Juror
  2. Statistical Modeling Juror
  3. Symbolic Regression / Scientific Discovery Juror
  4. UX Research Juror
  5. Scientific Communication Juror
- Actual roster:
  1. Pure/Applied Mathematics Juror: `019e6404-04db-7f83-bbde-5988be595fd8` (`Kierkegaard`)
  2. Statistical Modeling Juror: `019e6404-0651-7d23-98b5-e1fc67177163` (`Euler`)
  3. Symbolic Regression / Scientific Discovery Juror: `019e6404-0760-7560-90d7-f33b9831d1af` (`Meitner`)
  4. UX Research Juror: `019e6404-08a3-7942-ad8f-f595ec631dfa` (`Ampere`)
  5. Scientific Communication Juror: `019e6404-09f4-73a0-a061-a718f9773484` (`Pasteur`)

## Wave 1 Planning Synthesis

All five planners independently recommended a 7-agent Wave 2 and a 5-person jury. No subagent capacity cap was encountered in Wave 1.

Synthesized Wave 2 plan:

1. **Mathematical Foundations Reviewer**: inspect MDL/code-length policy, compact-law framing, ordered-observation assumptions, split geometry, predictive boundaries, and whether formulas support claims. Key files: `docs/math.md`, `docs/reference/system.md`, `src/euclid/math/*`, `src/euclid/search/descriptive_coding.py`, `src/euclid/modules/split_planning.py`, `src/euclid/modules/predictive_tests.py`, `tests/unit/math/*`.
2. **Search and Representation Reviewer**: inspect CIR expressivity, reducers, composition, adapters, symbolic/search backends, runtime semantics, search-class disclosures, and fixture specificity. Key files: `src/euclid/cir`, `src/euclid/search`, `src/euclid/reducers`, `src/euclid/adapters`, `src/euclid/expr`, `src/euclid/rewrites`, `docs/reference/search-core.md`.
3. **Statistical Meaning Reviewer**: inspect calibration, scoring, uncertainty, N_eff, comparator validity, robustness, promotion/abstention, and whether evidence is statistically meaningful to a user. Key files: `src/euclid/modules/calibration.py`, `effective_sample.py`, `scoring.py`, `predictive_tests.py`, `probabilistic_evaluation.py`, `falsification/*`.
4. **Claim Semantics Reviewer**: inspect claim objects, publication gates, evidence contracts, mechanistic/probabilistic evaluation, normalized workbench claim taxonomy, and overclaim prevention. Key files: `src/euclid/modules/claims.py`, `gate_lifecycle.py`, `evidence_contracts.py`, `src/euclid/workbench/service.py`, claim/workbench tests.
5. **Workbench UX Reviewer**: inspect empty, loaded, abstained, no-winner, calibration, artifact, mobile, and long-equation states. Key files: `src/euclid/_assets/workbench/index.html`, `app.js`, `app.css`, `src/euclid/workbench/server.py`, `service.py`, `docs/reference/workbench.md`, frontend tests.
6. **Explanation and Narrative Reviewer**: inspect README, docs, workbench copy, explainer prompt/fallback, terminology, and non-hype scientific communication. Key files: `README.md`, `docs/system.md`, `docs/math.md`, `docs/reference/modeling-pipeline.md`, `docs/reference/workbench.md`, `src/euclid/workbench/explainer.py`.
7. **User Mental Model Reviewer**: inspect Euclid as a first-time scientist/analyst/forecaster: target choice, live/saved analysis, evidence hierarchy, “what did it find / what does it not mean / what should I do next?” Key files: target hints in `service.py`, tab flow in `app.js`, workbench screenshots, smoke script, saved-analysis behavior.

Common gap taxonomy:

- `math gap`: formal object, formula, comparison class, statistical gate, or artifact cannot support the stated meaning.
- `representation/recovery gap`: the law is not expressible, or it is expressible but search/fitting cannot recover/select it.
- `evidence gap`: structure exists but replay, comparability, calibration, N_eff, robustness, or publication support is insufficient or unclear.
- `presentation gap`: backend truth exists but UI/docs hide, mislabel, visually overweight, or under-explain it.
- `claim-truth/mixed gap`: normalization/UI/explainer implies a stronger interpretation than the normalized claim lane allows.

Overclaim guardrails for Wave 2:

- Descriptive compression is not predictive support.
- Benchmark-local winners are screening/context, not operator publication.
- Live API success is provider/runtime evidence, not scientific claim evidence.
- External engines and rewrite traces are proposal evidence only.
- `predictive_law` requires time-safe holdout/comparator evidence, claim card, scorecard, validation scope, replay, and publication record.
- `holistic_equation` requires backend-authored joint evidence with matching validation scope and publication record across deterministic and probabilistic lanes.

Evidence to collect or recommend:

- Read-only code/doc line evidence.
- Workbench screenshots at desktop and mobile for empty state, loaded Overview/Evidence/Forecast/Calibration/Search/Artifacts, no-winner, abstention, stale-holistic rejection, long equation, and explainer fallback.
- Targeted tests, preferably with `-p no:cacheprovider` for pytest:
  - `PYTHONPATH=src python3.11 -m pytest -q -p no:cacheprovider tests/unit/math/test_codelength.py tests/unit/search/test_descriptive_coding.py tests/unit/search/test_portfolio.py tests/unit/cir tests/unit/reducers`
  - `PYTHONPATH=src python3.11 -m pytest -q -p no:cacheprovider tests/unit/modules/test_claims.py tests/unit/modules/test_gate_lifecycle.py tests/unit/modules/test_predictive_tests.py tests/unit/modules/test_calibration.py tests/unit/modules/test_probabilistic_evaluation.py tests/unit/modules/test_robustness.py`
  - `PYTHONPATH=src python3.11 -m pytest -q -p no:cacheprovider tests/unit/workbench tests/integration/test_workbench_analysis.py`
  - `npm run test:frontend -- --run tests/frontend/workbench/app.test.js tests/frontend/workbench-ui.test.js tests/frontend/workbench/app.holistic-contract-worker4-20260418.test.js`
  - `PYTHONPATH=src python3.11 scripts/workbench_ui_smoke.py --output-root /tmp/euclid-wave2-workbench --hold-open`

Local evidence already collected before Wave 2:

- `README.md:14-20` states compact equation discovery and separation of descriptive equation from predictive claims.
- `docs/math.md:18-79` defines MDL-style total codelength and description gain.
- `docs/reference/search-core.md:25-32` scopes search-class promises; `docs/reference/search-core.md:88-90` makes codelength comparability normative.
- `src/euclid/search/descriptive_coding.py:295-300` admits candidates when description gain exceeds threshold, and `:547-556` ranks by total bits, description gain, structure bits, hash length, candidate id.
- `src/euclid/modules/evidence_contracts.py:12-95` omits `description_gain_non_positive` from known evidence reason codes. A live SPY daily-return workbench run then produced failed probabilistic lane payloads with `ContractValidationError: unknown_evidence_reason_code ... 'description_gain_non_positive'`, making Calibration show raw contract-error text instead of a user-comprehensible downgrade.
- Workbench loaded Overview explicitly preserves claim honesty: descriptive reconstruction over 19 observations, not operator publication, not benchmark-local compact winner, not predictive-within-scope claim.

## Wave 2 Findings

### Current Math/UX State

Euclid is already fail-closed in several important places: the live workbench run preserved `publishable=false`, point publication `abstained`, no benchmark-local winner, and non-predictive language for a 19-row SPY `daily_return` reconstruction. The strongest risk is not a single hype statement; it is a mismatch between evidential status and user attention. A polished retrospective equation, R2, target labels, calibration tab, and "law" wording can still make a user believe more than the contracts support.

The highest-leverage improvement theme is a single fail-closed claim/evidence contract that is visible to users: what object was found, what gate it cleared, what gate it failed, what it can be used for, and what it cannot support.

### Highest-confidence Findings

1. **Diagnostic reason-code contract is incomplete.**
   - Evidence: `src/euclid/modules/evidence_contracts.py:12-95` omits `description_gain_non_positive`; `src/euclid/modules/evidence_contracts.py:119` raises on unknown reason codes; live payload `/tmp/euclid-workbench-live-analysis.json` failed all probabilistic lanes with `unknown_evidence_reason_code ... description_gain_non_positive`; workbench Calibration rendered raw contract-error text.
   - Primary axis: both mathematical truthfulness and UX.
   - Why it matters: a meaningful "no descriptive gain" result becomes a system-looking failure instead of an interpretable abstention.
   - Confidence: high.

2. **Floor-failed retrospective reconstructions can still look like accepted descriptive structure.**
   - Evidence: live payload had `target_floor_cleared=false`, `claim_class=descriptive_reconstruction`, `publishable=false`, and Evidence Studio `claim_ceiling=descriptive_structure`; `src/euclid/workbench/service.py:370-405` selects a best fallback reconstruction even when none clear the target floor; screenshots show the Fourier equation prominently.
   - Primary axis: both.
   - Why it matters: "we drew a retrospective curve" can be read as "we found structure".
   - Confidence: high.

3. **Effective sample size can outrun the raw paired evidence.**
   - Evidence: `src/euclid/modules/predictive_tests.py:211` resolves caller-supplied `effective_sample_size`; `:337` gates on that value. The Statistical Modeling Juror ran a read-only probe where promotion passed with `raw_pair_count=2` and supplied `effective_sample_size=80`.
   - Primary axis: mathematical.
   - Why it matters: predictive support must be bounded by actual validation evidence.
   - Confidence: high.

4. **MDL/code-length meaning needs a canonical policy and derived model-code cost.**
   - Evidence: `docs/math.md:18-79` defines description gain; `src/euclid/search/descriptive_coding.py:295-300` admits candidates by gain; `:547-556` ranks by bits/gain; `src/euclid/cir/models.py:243`, `src/euclid/cir/normalize.py:62`, and `src/euclid/search/descriptive_coding.py:560` show model-code decomposition can be stored, normalized, and read from candidates.
   - Primary axis: mathematical.
   - Why it matters: compactness is only meaningful if coding costs are comparable, target-scale-aware, and not fixture-supplied.
   - Confidence: medium-high.

5. **Representation expressivity is close but not yet law-rich enough.**
   - Evidence: reviewers found scalar-heavy CIR/AST/reducer assumptions and limited first-class support for basis expansions, indexed sums, arrays, vector laws, recurrence, and composition bindings. Jury dissent sharpened this: expression-CIR integration is not absent because `src/euclid/search/orchestration.py:247` accepts `proposed_cir`, and reducer composition has typed payload validation in `src/euclid/reducers/composition.py:610`; the narrower gap is rendering, ranking, component lookup, and planted recovery tests.
   - Primary axis: mathematical.
   - Why it matters: Euclid needs to express compact laws without relying on fixture-specific strings or overfit reconstructions.
   - Confidence: medium.

6. **Probabilistic interpretation needs production stochastic evidence.**
   - Evidence: `schemas/contracts/stochastic-law.yaml:8` says heuristic Gaussian support cannot satisfy the stochastic contract; `src/euclid/modules/probabilistic_evaluation.py:97` defaults compatibility mode; `:672` emits `heuristic_gaussian_support_not_production`; `src/euclid/modules/replay.py:716` returns early unless production; `src/euclid/modules/claims.py:355` can add probabilistic interpretation for non-point forecasts.
   - Primary axis: mathematical.
   - Why it matters: probabilistic-looking output must not become user-facing forecast meaning without production support, calibration, replay, and validation scope.
   - Confidence: medium-high.

7. **Mechanistic evidence independence is too easy to satisfy.**
   - Evidence: `schemas/contracts/mechanistic-evidence.yaml:30` requires domain-bound external evidence and independence attestation; `src/euclid/modules/mechanistic_evidence.py:315` can pass independence when there are no overlap refs, including underspecified records; `:370` aggregates green dossier checks; positive test `tests/unit/modules/test_mechanistic_evidence.py:47` is thin.
   - Primary axis: mathematical.
   - Why it matters: a mechanism claim should mean independent external evidence, not just absence of detected overlap.
   - Confidence: medium-high.

8. **Workbench visual hierarchy leads with equation/R2 instead of belief verdict.**
   - Evidence: loaded overview screenshot; snapshot `.playwright-mcp/page-2026-05-26-loaded-overview-depth6.yml`; `src/euclid/_assets/workbench/app.js` renders a strong equation/hero surface before all users understand abstention, claim ceiling, blocker reasons, and no-winner status.
   - Primary axis: UX.
   - Why it matters: non-author users read visual prominence as evidential strength.
   - Confidence: high.

9. **Target, tab, and claim labels blur scientific question, analysis type, and forecast meaning.**
   - Evidence: `src/euclid/workbench/service.py:101` uses target phrasing that starts with "Predict"; UI exposes Evidence/Forecast/Calibration/Search while docs mention older Workspace/Point/Probabilistic/Benchmark labels; `src/euclid/_assets/workbench/app.js:23` exposes `Predictive symbolic law`; `schemas/contracts/claim-lanes.yaml:7` says scoped prediction is not universal law.
   - Primary axis: UX.
   - Why it matters: users can infer forecast intent before the evidence gates speak.
   - Confidence: high.

10. **Explainers and artifacts are not yet an audit packet.**
    - Evidence: `src/euclid/workbench/explainer.py` snapshots omit or underweight the current strongest surfaces such as descriptive reconstruction, gap report, and Evidence Studio; Artifacts mostly expose paths/copy affordances rather than a belief checklist with replay, comparator, robustness, calibration, benchmark, publication record, and readiness state.
    - Primary axis: UX.
    - Why it matters: users need a compact reasoned answer: what is safe to cite, replay, trust, or not trust.
    - Confidence: medium-high.

### Dissent and Uncertainty

- Expression-CIR and composition gaps should not be stated as "missing" wholesale. They exist in parts of the system; the more precise backlog item is to connect accepted representation payloads to ranking, rendering, component lookup, code-length accounting, and planted recovery tests.
- `description_gain_non_positive` may belong in a descriptive/benchmark-specific reason-code namespace rather than the global evidence list. The important invariant is typed normalization and user-comprehensible downgrade, not necessarily one flat enum.
- Some non-point claim paths may already be blocked elsewhere from publishing compatibility stochastic lanes. The claim/replay boundary should still encode the production-stochastic requirement directly.
- Mobile workbench QA was not freshly run. Treat mobile/dense-layout findings as required validation, not a fully proven defect.

### Additional Commands After Wave 2

```bash
tail -n 120 docs/progress/2026-05-26-euclid-math-ux-improvement-swarm.md
git status --branch --short
git rev-parse HEAD
ps -p 8694
sed -n '1,220p' /Users/danielbloom/.codex/superpowers/skills/verification-before-completion/SKILL.md
wc -l docs/progress/2026-05-26-euclid-math-ux-improvement-swarm.md
tail -n 30 docs/progress/2026-05-26-euclid-math-ux-improvement-swarm.md
rg -n "Pending|pending|APPROVE BACKLOG|Actual count|Final Jury Verdict|Recommended Next Development Wave" docs/progress/2026-05-26-euclid-math-ux-improvement-swarm.md
rg -n "^(Pending\.|- Actual count: pending|Actual count: pending)" docs/progress/2026-05-26-euclid-math-ux-improvement-swarm.md
git diff -- docs/progress/2026-05-26-euclid-math-ux-improvement-swarm.md
nl -ba /Users/danielbloom/.codex/memories/MEMORY.md | sed -n '33,46p'
nl -ba /Users/danielbloom/.codex/memories/MEMORY.md | sed -n '177,181p'
```

Final verification observations:

- Broad `rg` found final vote/count sections; the narrower anchored placeholder check exited with no matches for standalone `Pending.` or `Actual count: pending`.
- `git status --branch --short` after the ledger update showed only untracked generated evidence: this ledger and four workbench screenshots.
- `ps -p 8694` returned no running process for the second workbench server session.
- `git diff -- docs/progress/...` printed no diff because the ledger is untracked.

## Proposed Improvements

### P0 Backlog

1. **Typed admissibility/reason-code normalization**
   - Files: `src/euclid/modules/evidence_contracts.py`, `src/euclid/workbench/service.py`, `src/euclid/search/descriptive_coding.py`, workbench tests, evidence-contract tests.
   - Fix: make `description_gain_non_positive` and related descriptive-admissibility outcomes typed, normalized, and rendered as abstention/ineligibility, not probabilistic crashes.
   - Expected impact: users see "no admissible structure" rather than a broken calibration lane.
   - Verification: unit test live-like payload normalization; evidence-contract test for known/admissible reason codes; workbench fixture asserting no raw `unknown_evidence_reason_code` text.

2. **Verdict-first workbench overview**
   - Files: `src/euclid/_assets/workbench/app.js`, `app.css`, `index.html`, frontend snapshots/tests.
   - Fix: first viewport starts with "What can I believe?" including claim ceiling, publication status, blocker reasons, sample/evidence counts, and next evidence action. Equation/R2 moves below verdict and is labeled as reconstruction/candidate/published claim.
   - Expected impact: non-author users understand limits before equation quality.
   - Verification: screenshot tests for non-publishable and descriptive-only fixtures; loaded SPY screenshot comparison; mobile screenshot pass.

3. **Split floor-failed reconstruction from accepted descriptive structure**
   - Files: `src/euclid/workbench/service.py`, claim/evidence normalization, frontend rendering/tests.
   - Fix: when `target_floor_cleared=false`, expose `inspection_reconstruction` or `non_claim_reconstruction`, not `descriptive_structure`.
   - Expected impact: retrospective path fits stop looking like discovered compact structure.
   - Verification: backend unit asserting no `descriptive_structure` ceiling for floor-failed reconstruction; frontend fixture label test.

4. **Effective-sample-size invariant for predictive promotion**
   - Files: `src/euclid/modules/predictive_tests.py`, `src/euclid/modules/effective_sample.py`, `src/euclid/modules/scoring.py`, predictive tests.
   - Fix: derive and cap effective sample size from raw paired evidence or require an auditable estimator artifact; reject `n_eff > raw_pair_count` unless explicitly justified by a validated estimator.
   - Expected impact: predictive claims cannot be promoted by metadata detached from validation evidence.
   - Verification: regression where `raw_pair_count=2`, supplied `effective_sample_size=80` must abstain/fail.

5. **Public terminology cleanup for scoped prediction**
   - Files: `src/euclid/_assets/workbench/app.js`, `src/euclid/workbench/service.py`, `schemas/contracts/claim-lanes.yaml`, docs/tests.
   - Fix: replace user-facing "Predictive symbolic law" / universal-law implication with "Scoped predictive claim" or "Declared-scope predictive rule"; reserve "law" for evidence contracts that actually justify it.
   - Expected impact: reduces overclaiming without weakening legitimate predictive results.
   - Verification: frontend tests assert no public `Predictive symbolic law` for `predictive_within_declared_scope`.

### P1 Backlog

1. Canonical MDL/code-length policy manifest: quantization, target scale, residual coding, parameter/model code, sensitivity report, and candidate-cost derivation.
2. Production-stochastic gate for non-point/probabilistic interpretation: residual-history refs, stochastic model refs, calibration, replay, and validation scope.
3. Mechanistic evidence hardening: nonempty typed external evidence records, source version, lineage, independence, and fail-closed unknown lineage.
4. Calibration evidence clarity: sample/bin gates, reliability uncertainty, failed-lane summary, and no forecast-distribution language when lanes fail.
5. Target decision panel: "describe path", "test scoped predictive rule", "estimate event probability", and "replay/inspect level" with claim ceilings and cannot-support text.
6. Audit packet / belief checklist: loaded analysis, replay, comparator, robustness, calibration, benchmark, publication record, readiness, and artifact link per row.

### P2 Backlog

1. First-class law-rich representations: arrays, indexed sums, basis expansions, vector equations, recurrence, piecewise/regime forms, and typed component bindings.
2. Expression-CIR to descriptive coding/ranking/rendering integration, stated precisely around currently partial integration.
3. Operator capability matrix across registry, evaluator, SymPy conversion, units, and UI rendering.
4. Planted recovery suites for additive residual, piecewise/regime, spectral, recurrence, panel alignment, and nonstationary cases.
5. Explainer/documentation alignment with current workbench tabs and actual strongest surfaces.
6. Mobile and dense-evidence visual QA across empty, loaded, failed calibration, no-winner, abstained, long-equation, and artifact-heavy states.

### Quick Wins

- Add or namespace `description_gain_non_positive` and render it as a descriptive-admissibility abstention.
- Add a top-of-calibration failed-evidence summary for lane failures.
- Rename visible `Predictive symbolic law` copy.
- Put claim ceiling/publication status before the equation card.
- Update docs and explainer tab names to match the current UI.
- Add target helper text that distinguishes transform selection from forecast intent.

### Deeper Mathematical Research Bets

- A principled Euclid coding policy: scale-adaptive quantization, residual code, model code, and sensitivity analysis.
- Representation primitives for compact ordered-observation laws beyond scalar point formulas.
- Validation-scope algebra: calendar alignment, panel split geometry, nonstationarity segments, and evidence comparability.
- Production stochastic evidence contracts that connect residual models, calibration, replay, and publication meaning.
- Mechanistic evidence semantics that distinguish external causal/mechanistic support from predictive fit adjacency.

### Screenshots and Demos Needed Next

- Desktop and mobile workbench screenshots for: empty state, loaded descriptive-only abstention, failed probabilistic lanes, no benchmark winner, successful scoped predictive claim, artifact-heavy replay view, stale or rejected holistic claim, long equation.
- Demo: live run where `description_gain_non_positive` becomes a typed abstention and Calibration no longer shows raw contract errors.
- Demo: floor-failed reconstruction rendered as inspection-only with verdict-first hierarchy.
- Demo: predictive promotion regression with raw pair count smaller than supplied effective sample size.

## Final Jury Verdict

### Vote

- Pure/Applied Mathematics Juror: **APPROVE BACKLOG**. Strongest concern: failed or non-admissible situations can look mathematically richer than they are. Highest leverage: one fail-closed claim/evidence contract across backend and workbench.
- Statistical Modeling Juror: **APPROVE BACKLOG**. Strongest concern: predictive/probabilistic support can exceed raw validation evidence through effective-sample metadata. Highest leverage: statistical evidence invariant before any predictive/probabilistic pass.
- Symbolic Regression / Scientific Discovery Juror: **APPROVE BACKLOG**. Strongest concern: representation findings need sharper wording because some CIR/composition integration already exists. Highest leverage: fix descriptive reconstruction claim surface first, then canonicalize code costs.
- UX Research Juror: **APPROVE BACKLOG**. Strongest concern: target labels and visual hierarchy make forecast intent too easy to infer. Highest leverage: persistent "What can I believe?" block before the equation.
- Scientific Communication Juror: **APPROVE BACKLOG**. Strongest concern: careful docs are undercut by first-read UI wording/order. Highest leverage: target/scientific question, claim ceiling, belief checklist, then equation.

### Final Jury Verdict

**APPROVE BACKLOG** with two implementation cautions:

1. Keep the representation backlog precise. Do not claim CIR or composition integration is absent where partial integration exists.
2. Preserve fail-closed truth. The next development wave should not make the workbench more persuasive unless it also makes claim ceilings, failed gates, and evidence incompleteness more visible.

### Recommended Next Development Wave

Run a focused **Claim Truth + Verdict-First Workbench Wave**:

1. Normalize descriptive-admissibility reason codes and remove raw contract errors from calibration.
2. Split floor-failed reconstruction from accepted descriptive structure.
3. Add the effective-sample-size promotion invariant.
4. Move workbench hierarchy to verdict first, equation second.
5. Clean public terminology around scoped prediction and law.
6. Add regression fixtures and browser screenshots proving non-author users see what is safe and unsafe to believe before seeing the equation.

## 2026-05-27 Final SPY Price-Close Pass

Final live run:

- Request: SPY `price_close`, `2026-04-27` through `2026-05-27`, probabilistic lanes enabled, benchmark enabled.
- Analysis path: `build/workbench/readme-live/20260527T040846Z-spy-price-close/analysis.json`.
- Returned data: 21 trading-day close observations from `2026-04-27T00:00:00Z` through `2026-05-26T00:00:00Z`.
- Exact descriptive lane: `completed` as an expanded real inverse-DFT equation over row index `n`, sample size 21, max absolute error `2.2737367544323206e-13`, effective exact tolerance `1e-10`, exact tolerance cleared.
- Predictive law search: `no_publishable_law` with `descriptive_structure`, `predictive_support_failed`, and `predictive_scorecard_failed`.
- Benchmark-local winner: `algorithmic_search_backend / algorithmic_last_observation`.
- Benchmark descriptive fit status: `absent_reconstruction_floor_failed`.
- LLM explanations: `completed`.

README evidence updated:

- `README.md` now documents the May 27 SPY price-close run and keeps the expanded exact descriptive equation separate from predictive-law search.
- Fresh README screenshots captured through Playwright from the new analysis artifact:
  - `docs/assets/readme/workbench/live-spy-30d/price-close-overview.png`
  - `docs/assets/readme/workbench/live-spy-30d/price-close-evidence.png`
  - `docs/assets/readme/workbench/live-spy-30d/price-close-calibration.png`
  - `docs/assets/readme/workbench/live-spy-30d/price-close-artifacts.png`
- `docs/assets/readme/workbench/live-spy-30d/manifest.json` records the run summary, screenshot dimensions, and SHA-256 hashes.

Verification after the final pass:

```bash
PYTHONPATH=src python3.11 -m pytest -q tests/unit/workbench/test_service.py tests/integration/test_workbench_analysis.py
# 114 passed in 18.39s

PYTHONPATH=src python3.11 -m pytest -q tests/spec_compiler/test_math_documentation_truthfulness.py
# 8 passed in 0.63s

npm run test:frontend -- --run tests/frontend/workbench/app.test.js tests/frontend/workbench-ui.test.js
# 48 passed in 454.39s

jq . docs/assets/readme/workbench/live-spy-30d/manifest.json
# pass

git diff --check
# pass

rg -n "<forbidden exact-lane shorthand patterns>" .
# only negative assertions in regression tests

shasum -a 256 docs/assets/readme/workbench/live-spy-30d/price-close-overview.png docs/assets/readme/workbench/live-spy-30d/price-close-evidence.png docs/assets/readme/workbench/live-spy-30d/price-close-calibration.png docs/assets/readme/workbench/live-spy-30d/price-close-artifacts.png
# all four hashes match docs/assets/readme/workbench/live-spy-30d/manifest.json

PYTHONPATH=src python3.11 -c '<normalize saved SPY analysis and check the descriptive exact label against the forbidden shorthand set>'
# normalized saved analysis begins with \hat{y}(t_n)=... expanded inverse-DFT coefficients; has_old=False; has_formula_latex=True; predictive_status=no_publishable_law
```

Frontend note: the full jsdom frontend harness initially exposed saved-analysis load waits that were too short under full-suite load. The app behavior was not changed for that failure; `tests/frontend/workbench/app.test.js` now uses a 15s Vitest timeout for the integration-style file and a 3s default polling window for async load assertions.
