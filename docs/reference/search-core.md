---
title: Search Core
status: active_reference
related:
- system.md
- modeling-pipeline.md
- contracts-manifests.md
- ../../schemas/core/math-source-map.yaml
---
# Search Core

Euclid's search core provides bounded proposal generation, candidate normalization, replay-aware backend disclosures, and MDL-style description-gain accounting. It searches declared spaces of reducers and compositions, converts candidates into a common intermediate representation, and compares them only when the resulting code-length accounting is semantically comparable.

## Search classes

The main backends live in `src/euclid/search/backends.py`.

Supported search classes:

- `exact_finite_enumeration`
- `bounded_heuristic`
- `equality_saturation_heuristic`
- `stochastic_heuristic`

Each backend makes a different promise:

- exact enumeration only claims exactness over the declared finite program space
- bounded heuristic declares incompleteness and proposal omission
- equality saturation declares rewrite-neighborhood search with cost extraction
- stochastic search declares seeded, restart-aware heuristic exploration

The backend label is therefore part of the evidence contract: it bounds what completeness, omission, and reproducibility claims the search result may carry.

## Candidate Intermediate Representation

The CIR lives in:

- `src/euclid/cir/models.py`
- `src/euclid/cir/normalize.py`

The candidate intermediate representation separates:

- structural identity: family, form, inputs, state, literals, parameters, composition
- execution semantics: history access, update law, forecast operator, observation model
- evidence and provenance: backend origin, model-code decomposition, replay hooks, transient diagnostics

Normalization only canonicalizes the structural and execution terms plus model-code decomposition. Provenance and transient diagnostics do not affect canonical bytes or hashes.

## Reducer families and compositions

Retained primitive families include:

- `analytic`
- `recursive`
- `spectral`
- `algorithmic`

Composition families include:

- `piecewise`
- `additive_residual`
- `regime_conditioned`
- `shared_plus_local_decomposition`

Composition semantics live in:

- `src/euclid/reducers/models.py`
- `src/euclid/reducers/composition.py`

The search layer can represent more structures than later publication code will treat as law-eligible. Search breadth and publication eligibility are intentionally separate policies.

## Description gain and comparison classes

Descriptive coding is implemented in `src/euclid/search/descriptive_coding.py` and the support code in `src/euclid/math`.

The main quantity is:

- `description_gain = reference_bits - total_code_bits`

Total code bits include:

- family bits
- structure bits
- literal and parameter bits
- state bits
- quantized residual data bits

Cross-candidate codelength comparison is only allowed inside a single `CodelengthComparisonKey`. The `strict_single_class` law requires matching quantizer, reference policy, data-code family, support, horizon geometry, `row_set_id`, residual-history construction, parameter/state lattice, and runtime signature. The opt-in `prequential_laplace_residual_bin_v1` data-code family is legal only when each row is encoded from prefix-only evidence.

These comparability constraints are normative, not cosmetic. Euclid does not allow code-length rankings between candidates whose observation models, row sets, quantizers, reference descriptions, or runtime signatures make the accounting incommensurate.

## Multi-horizon fitting strategies

`FitStrategySpec` is the explicit identity for fitting geometry. The default remains `legacy_one_step`; it preserves the historical one-step objective when no rollout fitting is configured. Staged strategies include `recursive_rollout`, `direct_analytic`, `joint_analytic`, and `rectify_analytic`. Each strategy identity includes the strategy id, horizon set, horizon weights, point loss, and entity aggregation mode.

Horizon sets do not have to be contiguous. A non-contiguous set such as `(1, 3)` is valid when the legal training-origin panel declares rows for those horizons. Rollout objective rows are computed over that legal origin/horizon panel and mirror scoring aggregation over the same panel.

## Adapters

Adapters turn heterogeneous sources into one normalized candidate universe.

Key files:

- `src/euclid/adapters/algorithmic_dsl.py`
- `src/euclid/adapters/decomposition.py`
- `src/euclid/adapters/sparse_library.py`
- `src/euclid/adapters/portfolio.py`

These adapters preserve source provenance while rebinding candidates into common CIR and finalist-selection structures.

## Algorithmic DSL

The DSL lives in:

- `src/euclid/search/dsl`
- `src/euclid/modules/algorithmic_dsl.py`

It defines a bounded program fragment that can be parsed, canonicalized, enumerated, and translated into algorithmic CIR candidates. Stable aliases such as `algorithmic_last_observation` are preserved in adapter normalization.

## Prototype workflow

`src/euclid/prototype/workflow.py` is a simpler end-to-end pipeline over a small toy family set:

- `constant`
- `drift`
- `linear_trend`
- `seasonal_naive`

It still proves replay, robustness, and release behavior, but it is a smaller workflow than the full runtime search surface.

## Key source anchors

- `src/euclid/search/backends.py`
- `src/euclid/search/frontier.py`
- `src/euclid/search/portfolio.py`
- `src/euclid/search/descriptive_coding.py`
- `src/euclid/search/dsl/parser.py`
- `src/euclid/search/dsl/enumerator.py`
- `src/euclid/cir/models.py`
- `src/euclid/cir/normalize.py`
- `src/euclid/reducers/models.py`
- `src/euclid/reducers/composition.py`
- `src/euclid/adapters/portfolio.py`
- `src/euclid/math/codelength.py`
- `src/euclid/math/quantization.py`
- `src/euclid/math/reference_descriptions.py`


## Engine provenance and fitting diagnostics

Search engines disclose their provenance, resource budget, timeout behavior, and replay seed before their candidates can support a claim lane. PySINDy and PySR adapters enter through the same candidate intermediate representation as native reducers. External engine output is proposal evidence only: it cannot publish a claim directly and must still pass Euclid-owned CIR closure, fitting, scoring, replay, falsification, and publication gates. Equality-saturation and rewrite traces are evidence about the explored neighborhood, not proof that every algebraic form was explored. Stochastic restarts report seeds and restart counts, and bounded engines report proposal limits and graceful degradation.

Law-eligible publication is not the same as candidate discovery. A backend may find a candidate that remains descriptive only, downgraded, or abstained because score, calibration, robustness, replay, or evidence contracts did not close.
