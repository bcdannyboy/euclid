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

Cross-candidate codelength comparison is only allowed when the observation-model family, forecast type, support, quantization step, and composition runtime signature are comparable.

These comparability constraints are normative, not cosmetic. Euclid does not allow code-length rankings between candidates whose observation models or runtime signatures make the accounting incommensurate.

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
