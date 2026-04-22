# Euclid Enhancement Master Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Euclid's current narrow retained-scope discovery core with a robust scientific law-discovery platform for ordered observations, while preserving Euclid's core evidence discipline: CIR identity, replay, validation geometry, comparability, falsification, and scoped publication.

**Architecture:** The enhanced system uses mature numerical and symbolic libraries for computation, but keeps Euclid as the governing layer. Search engines propose candidate laws, Euclid lowers them into a typed symbolic representation and CIR, Euclid refits constants under frozen evaluation geometry, Euclid scores and falsifies them, and Euclid publishes only the strongest claim justified by evidence.

**Tech Stack:** Python 3.11+, NumPy, pandas, SciPy, SymPy, Pint, statsmodels, scikit-learn, PySINDy, PySR, egglog, joblib/Ray, SQLAlchemy, Pydantic, PyYAML, Typer, PyArrow, httpx, python-dotenv, vcrpy, responses/respx, pytest, pytest-timeout, pytest-xdist, Hypothesis, and a rebuilt evidence-oriented workbench.

---

## 0. Core Policy For This Plan

This plan intentionally does not preserve backward compatibility with functionality that is being replaced. The existing behavior may remain temporarily while work is in progress, but the final target should remove or demote old retained-slice implementations when the enhanced replacement is complete.

The design priority is:

1. Correct scientific and mathematical behavior.
2. Honest claim scope.
3. Reproducibility and replay.
4. Robust discovery power.
5. Operational convenience.

The design priority is not:

1. Preserving every existing public shape.
2. Keeping dependency count low.
3. Avoiding major module boundaries.
4. Maintaining old shim labels that implied capabilities not implemented.

The new Euclid should be allowed to replace existing modules, schemas, docs, fixtures, and workbench surfaces when the enhanced functionality is clearer and more scientifically correct.

## 1. Non-Negotiable System Invariants

All future implementation work must preserve these invariants:

- External engines can propose equations, but they cannot publish claims.
- Every candidate must normalize into Euclid-owned CIR.
- Backend provenance must never affect canonical structural identity.
- Fitting must be fold-local and time-safe.
- Confirmatory data must never be used for search, fitting, symbolic simplification decisions, or hyperparameter selection.
- Every optimizer, symbolic simplifier, stochastic model, and external engine must write replay metadata.
- Every strong claim must have an explicit claim lane.
- "Universal law" must mean invariance and transport evidence, not merely high fit or high compression on one series.
- Probabilistic claims must arise from explicit stochastic models, not a hardcoded Gaussian wrapper around point forecasts.
- Benchmark success must be semantic: exact artifact existence is not enough.

## 2. High-Level Replacement Map

| Current Surface | Replace With | Remove Or Demote |
|---|---|---|
| Narrow `search/backends.py` proposal pool | Modular engine portfolio under `search/engines`, `search/orchestration`, `expr`, and `fit` | Remove current default proposal pool as the main discovery mechanism |
| Small algorithmic DSL as primary expressive substrate | Typed symbolic expression IR backed by SymPy and Euclid metadata | Keep old DSL only as an import/lowering compatibility reader until fully deleted |
| `adapters/sparse_library.py` as SINDy label shim | Real PySINDy backend with sparse dynamics traces | Remove relabel-only `sindy-sparse-library` behavior |
| `adapters/decomposition.py` as AI-Feynman label shim | Real decomposition orchestrator using symmetry, separability, residual decomposition, and subproblem dispatch | Remove relabel-only `ai_feynman-decomposition` behavior |
| Equality saturation as proposal sorting | Real rewrite/e-graph simplification and extraction layer | Remove current cost-sort-only equality saturation implementation |
| Family-specific candidate fitting switch | Unified SciPy-backed fitting layer with family specializations as fast paths | Remove duplicate fitting logic from search and candidate-fitting modules |
| Gaussian-only observation model | Explicit observation/process model registry using SciPy distributions | Remove hardcoded Gaussian-only publication semantics |
| Heuristic probabilistic support path | Fitted stochastic models with proper scoring and calibration | Remove family-specific Gaussian scale inflation as production path |
| Boolean `candidate_beats_baseline` promotion | Statistical and practical promotion evidence using SciPy/statsmodels | Remove pure boolean promotion as sufficient for predictive publication |
| `predictively_supported` as strongest ordinary lane | Separate `invariant_predictive_law`, `stochastic_law`, and `transport_supported_law` lanes | Stop using "law" language for scoped predictive claims without invariance |
| In-sample shared-plus-local panel optimizer | Shared symbolic skeleton discovery with entity/local parameters and holdout entities | Remove panel memorization as support for shared law claims |
| Benchmark artifact existence checks | Benchmark semantic threshold gates | Remove surface pass rules that only check file existence |
| Vanilla formula-display workbench | Evidence studio showing lineage, invariance, diagnostics, falsification, and claim ceilings | Replace UI sections that imply claim strength without evidence |

## 3. Library Usage Policy

Use mature libraries as the default source of mathematical behavior. Do not reimplement numerical optimization, statistical distributions, sparse regression, symbolic simplification, or time-series diagnostics unless Euclid needs a specific governance wrapper.

### 3.1 NumPy

Use NumPy for:

- Dense numeric arrays.
- Vectorized candidate evaluation.
- Residual computation.
- Design matrices.
- Stable sorting and deterministic numeric transforms.
- Low-level data interchange with SciPy, scikit-learn, PySINDy, and PySR.

Do not use NumPy for:

- Statistical tests that SciPy or statsmodels already implements.
- Symbolic manipulation.
- Units or dimension checking.

### 3.2 pandas

Use pandas for:

- External tabular ingestion.
- Benchmark dataset loading.
- Friendly CSV/Parquet IO.
- Wide-to-long and long-to-panel transformations before Euclid canonical ingestion.

Do not let pandas objects become the internal identity model. Convert to Euclid-owned dataclasses/manifests before search, fitting, replay, and publication.

### 3.3 SciPy

SciPy is a core dependency for enhanced Euclid.

Use `scipy.optimize` for:

- `least_squares` for nonlinear constant fitting.
- `minimize` for constrained objectives and likelihood fitting.
- `differential_evolution`, `dual_annealing`, or `basinhopping` for difficult nonconvex constant initialization.
- Bound-constrained and penalty-based parameter fitting.
- Solver diagnostics and convergence reporting.

Use `scipy.stats` for:

- Gaussian, Student-t, Laplace, Poisson, negative binomial, Bernoulli, beta, lognormal, and mixture support calculations.
- CDF, PPF, log likelihood, CRPS approximations where needed, probability integral transform diagnostics, and event probabilities.
- Bootstrap intervals using `bootstrap`.
- Distributional goodness-of-fit checks where appropriate.

Use `scipy.signal` for:

- Periodogram, Welch spectra, peak finding, smoothing, Savitzky-Golay preprocessing, and signal diagnostics.
- Spectral feature generation when declared in the feature policy.

Use `scipy.integrate` for:

- `solve_ivp` based rollout validation of discovered ODEs.
- Integral-form or weak-form features.
- Numerical integration checks for conservation-law residuals.

Use `scipy.special` for:

- `gammaln`, `logsumexp`, `expit`, `logit`, stable log probabilities, and special functions needed by observation models.

Do not hand-roll any of the above in Euclid unless there is a documented reason and a property test comparing against SciPy.

### 3.4 SymPy

SymPy is a core dependency for symbolic law representation.

Use SymPy for:

- Expression parsing and normalization.
- Algebraic simplification under declared assumptions.
- Symbolic derivatives.
- Common subexpression elimination.
- Canonical expression serialization.
- Safe expression equivalence checks.
- Lowering PySR/PySINDy/native expressions into Euclid expression IR.
- Constructing human-readable formulas.

Use SymPy assumptions explicitly. Expressions that differ by domain assumptions must not collapse into the same CIR identity unless the assumptions prove equivalence.

Do not use string manipulation for formula identity after this replacement is complete.

### 3.5 Pint

Pint is the primary units and dimensional-analysis dependency.

Use Pint for:

- Unit registries.
- Dimensional compatibility checks.
- Unit-aware operator admissibility.
- Data column unit declarations.
- Mechanistic term unit validation.

SymPy may still assist with symbolic dimensions, but Pint should own runtime unit checking because it is designed for practical unit operations.

### 3.6 statsmodels

Use statsmodels for:

- Time-series residual diagnostics.
- Autocorrelation checks such as Ljung-Box style tests.
- Heteroskedasticity checks.
- HAC/Newey-West style covariance estimates.
- Regression diagnostics.
- Paired predictive comparison support when serial dependence matters.

Do not make a predictive claim depend only on raw score difference when statsmodels can provide a more honest uncertainty estimate.

### 3.7 scikit-learn

Use scikit-learn for:

- LASSO, ElasticNet, Ridge, and cross-validated sparse regression baselines.
- Standardization and feature pipelines for sparse libraries.
- Grouping/clustering support for environment or regime proposals when appropriate.
- Model-selection helpers for development-only tuning.

Do not use scikit-learn estimators directly as publishable Euclid laws. Use them as proposal or support-selection tools, then lower the selected symbolic structure into Euclid CIR and refit through Euclid.

### 3.8 PySINDy

PySINDy is a first-class discovery backend.

Use PySINDy for:

- Sparse identification of dynamical systems.
- ODE and discrete-time sparse dynamics.
- Polynomial, Fourier, custom, and weak-form libraries.
- STLSQ and SR3 optimizers.
- Ensemble SINDy for support stability.
- Implicit or rational discovery variants when suitable.
- Derivative estimation and smoothing utilities when declared in the plan.

Euclid must record:

- PySINDy version.
- Feature library configuration.
- Differentiation method.
- Optimizer and thresholds.
- Coefficient path.
- Support masks.
- Ensemble inclusion probabilities when used.
- Development rows used for fitting.
- Frozen seeds and random states.

PySINDy output must be lowered into Euclid expression IR and CIR. PySINDy model ranking must not bypass Euclid's MDL, validation, or claim gates.

### 3.9 PySR

PySR is a first-class symbolic-regression proposer, with Julia runtime metadata captured.

Use PySR for:

- Broad closed-form expression search.
- Hall-of-fame expression generation.
- Operator-constrained symbolic regression.
- Pareto candidate generation.
- Constant initialization before Euclid refit.

Euclid must record:

- PySR version.
- SymbolicRegression.jl version if available.
- Julia version.
- Operator set.
- Operator constraints and nested constraints.
- Population size, iterations, timeout, random seed.
- Hall-of-fame entries.
- Complexity function.
- Loss function.

PySR output must be parsed/lowered through SymPy and Euclid expression IR, then refit through Euclid.

### 3.10 egglog

Use egglog for real equality saturation and rewrite extraction when SymPy simplification is insufficient.

Use egglog for:

- Large rewrite neighborhoods.
- E-class duplicate collapse.
- Extracting lowest-cost equivalent expressions.
- Applying domain-specific rewrite systems.

Every rewrite rule must declare:

- Algebraic law.
- Required domain assumptions.
- Unit compatibility assumptions.
- Branch cut or singularity constraints.
- Whether the rewrite is exact, approximate, or prohibited for publication.

Do not use e-graphs to claim universal semantic identity without assumptions.

### 3.11 joblib and Ray

Use joblib for local parallel candidate evaluation.

Use Ray for distributed engine execution if benchmark and search workloads need larger scale.

Parallelism must remain deterministic:

- Candidate IDs define aggregation order.
- Seeds derive from Euclid seed scopes.
- Parallel failures become typed diagnostics.
- Results are stable across worker counts unless the search class explicitly declares stochastic behavior.

### 3.12 Hypothesis

Use Hypothesis for property-based tests.

Targets:

- Parser round trips.
- Expression equivalence.
- Unit compatibility.
- Rewrite soundness under assumptions.
- Optimizer no-confirmatory-access properties.
- Replay determinism.
- Observation-model support checks.

## 4. Target Package Structure

Create and migrate toward this structure:

```text
src/euclid/
  expr/
    ast.py
    operators.py
    domains.py
    units.py
    sympy_bridge.py
    evaluators.py
    serialization.py
  fit/
    objectives.py
    scipy_optimizers.py
    parameterization.py
    diagnostics.py
    refit.py
  stochastic/
    observation_models.py
    process_models.py
    scoring_rules.py
    calibration.py
    diagnostics.py
  search/
    orchestration.py
    engine_contracts.py
    engines/
      native.py
      pysindy_engine.py
      pysr_engine.py
      sparse_regression.py
      decomposition.py
      latent_state.py
      egraph.py
    proposals/
      scientific_templates.py
      lag_libraries.py
      seasonal.py
      recurrence.py
  invariance/
    environments.py
    scoring.py
    gates.py
    manifests.py
  panel/
    shared_structure.py
    partial_pooling.py
    leave_one_entity_out.py
  rewrites/
    rules.py
    sympy_simplifier.py
    egglog_runner.py
    extraction.py
  falsification/
    residuals.py
    surrogate_tests.py
    perturbations.py
    counterexamples.py
  benchmarks/
    existing files plus richer semantic gating
```

Existing files may remain during transition, but the final enhanced architecture should move major behavior out of monolithic modules.

## 5. Phase 0: Evidence Spine Reset

### Objective

Make release/readiness/benchmark evidence trustworthy before the large math replacement lands.

### Replace Or Remove

Remove any release-readiness behavior that infers semantic support from weak signals such as file existence alone.

Replace any completion report path that depends on stale temp files or ambiguous asset lookup.

### Files

- Modify: `src/euclid/release.py`
- Modify: `src/euclid/benchmarks/reporting.py`
- Modify: `src/euclid/benchmarks/runtime.py`
- Modify: `src/euclid/operator_runtime/resources.py`
- Modify: `src/euclid/cli/run.py`
- Modify: `src/euclid/cli/replay.py`
- Modify: `tests/unit/test_dev_scripts_smoke.py`
- Modify: `tests/unit/test_runtime_package_layout.py`
- Modify: `tests/integration/test_release_candidate_workflow.py`
- Create or update: `.github/workflows/ci.yml`

### Steps

1. Define one canonical asset policy.
2. Ensure scripts and tests use that policy.
3. Add structured semantic IDs to run evidence reports.
4. Add structured semantic IDs to replay evidence reports.
5. Make release completion consume explicit IDs first.
6. Make SQLite/artifact inspection a fallback only.
7. Add fail-closed reason codes for missing semantic evidence.
8. Update docs to state the asset and release evidence policy.

### Verification

Run:

```bash
pytest -q tests/unit/test_dev_scripts_smoke.py tests/unit/test_runtime_package_layout.py
pytest -q tests/integration/test_release_candidate_workflow.py
python -m euclid smoke
python -m euclid release status
python -m euclid release verify-completion
```

Expected:

- No release row passes without explicit supporting evidence.
- Missing assets produce precise errors.
- CI contract is represented in the repository.

## 6. Phase 1: Dependency And Numerical Runtime Foundation

### Objective

Add trusted math libraries as core dependencies and capture their versions in replay.

### Replace Or Remove

Remove the policy that avoids SciPy/SymPy/statsmodels/scikit-learn. The enhanced system depends on them.

Remove hand-written numerical routines when a library routine is more accurate or better tested.

### Files

- Modify: `pyproject.toml`
- Modify: `src/euclid/modules/replay.py`
- Modify: `src/euclid/runtime/profiling.py`
- Create: `src/euclid/runtime/numerical_environment.py`
- Create: `schemas/contracts/numerical-runtime.yaml`
- Create: `tests/unit/runtime/test_numerical_environment.py`

### Dependencies To Add

```toml
scipy = ">=1.13"
sympy = ">=1.13"
pint = ">=0.24"
statsmodels = ">=0.14"
scikit-learn = ">=1.5"
pysindy = ">=2.1"
pysr = ">=1.5"
egglog = ">=7.0"
joblib = ">=1.4"
ray = ">=2.0"
hypothesis = ">=6.0"
```

Exact version bounds should be adjusted to available package compatibility during implementation.

### Steps

1. Add dependencies.
2. Add numerical environment capture.
3. Include package versions in replay bundles.
4. Include Julia/PySR runtime metadata when available.
5. Include BLAS and platform metadata where practical.
6. Add a numerical policy manifest for tolerances and solver defaults.
7. Update clean-install and release checks to install the enhanced dependency set.

### Verification

Run:

```bash
pytest -q tests/unit/runtime/test_numerical_environment.py
python -m euclid smoke
python -m euclid replay --run-id <known-run-id>
```

Expected:

- Replay metadata includes numerical package versions.
- Missing numerical dependencies fail early and clearly.

## 7. Phase 2: Expression IR Replacement

### Objective

Replace the narrow algorithmic DSL as the main law representation with a typed symbolic expression IR.

### Replace Or Remove

Replace `src/euclid/search/dsl` as the primary expression model.

Remove string-based formula identity as a production mechanism.

Keep old DSL only as a temporary importer that lowers old programs into the new expression IR. Delete it once fixtures and benchmarks are migrated.

### Files

- Create: `src/euclid/expr/ast.py`
- Create: `src/euclid/expr/operators.py`
- Create: `src/euclid/expr/domains.py`
- Create: `src/euclid/expr/units.py`
- Create: `src/euclid/expr/sympy_bridge.py`
- Create: `src/euclid/expr/evaluators.py`
- Create: `src/euclid/expr/serialization.py`
- Modify: `src/euclid/cir/models.py`
- Modify: `src/euclid/cir/normalize.py`
- Modify: `schemas/contracts/algorithmic-dsl.yaml`
- Create: `schemas/contracts/expression-ir.yaml`
- Create: `tests/unit/expr/test_ast.py`
- Create: `tests/unit/expr/test_sympy_bridge.py`
- Create: `tests/unit/expr/test_units.py`
- Create: `tests/unit/expr/test_serialization.py`

### Expression Node Types

Implement at least:

- `Literal`
- `Parameter`
- `Feature`
- `Lag`
- `Delay`
- `State`
- `Derivative`
- `Integral`
- `UnaryOp`
- `BinaryOp`
- `NaryOp`
- `Conditional`
- `Piecewise`
- `NoiseTerm`
- `DistributionParameter`
- `FunctionCall`

### Operator Metadata

Every operator must declare:

- Name.
- Arity.
- Input domains.
- Output domain.
- Unit rule.
- Differentiability.
- Monotonicity if known.
- Commutativity.
- Associativity.
- Identity element if known.
- Absorbing element if known.
- Singularities.
- Safe/protected evaluation behavior.
- SymPy equivalent.
- NumPy/SciPy evaluator.

### Required Operators

Core:

- `add`, `sub`, `mul`, `div`
- protected division
- `neg`, `abs`, `min`, `max`
- `pow`, `pow2`, `sqrt`
- `exp`, `log`, protected log
- `sin`, `cos`, `tan`
- `tanh`, sigmoid
- `floor`, `ceil`, `clip`
- `where` / conditional

Time-series:

- lag
- finite difference
- rolling mean
- rolling sum
- cumulative sum
- convolution
- seasonal phase
- derivative estimate
- integral estimate

Stochastic:

- location parameter
- scale parameter
- rate parameter
- probability parameter
- dispersion parameter
- process noise term

### CIR Integration

CIR must store:

- Expression tree canonical bytes.
- Assumption set.
- Domain constraints.
- Unit constraints.
- Parameter declarations.
- Feature dependencies.
- State dependencies.
- Stochastic dependencies.
- Backend origin.
- Replay hooks.

### Verification

Run:

```bash
pytest -q tests/unit/expr
pytest -q tests/unit/cir
```

Expected:

- Equivalent expressions normalize together under valid assumptions.
- Non-equivalent expressions remain distinct.
- Unit-invalid expressions are rejected before fitting.

## 8. Phase 3: Unified Fitting Layer

### Objective

Replace family-specific fitting branches with a general fitting layer that uses SciPy.

### Replace Or Remove

Replace the main family switch in `src/euclid/modules/candidate_fitting.py`.

Remove separate closed-form search-time parameter estimation from `search/backends.py` where it duplicates fit-time behavior.

Keep simple analytic closed forms only as fast paths inside the new fitting layer.

### Files

- Create: `src/euclid/fit/objectives.py`
- Create: `src/euclid/fit/scipy_optimizers.py`
- Create: `src/euclid/fit/parameterization.py`
- Create: `src/euclid/fit/diagnostics.py`
- Create: `src/euclid/fit/refit.py`
- Modify: `src/euclid/modules/candidate_fitting.py`
- Modify: `src/euclid/search/descriptive_coding.py`
- Create: `schemas/contracts/optimizer-diagnostics.yaml`
- Create: `tests/unit/fit/test_scipy_optimizers.py`
- Create: `tests/unit/fit/test_parameterization.py`
- Create: `tests/unit/modules/test_candidate_fitting_enhanced.py`

### Fitting Modes

Implement:

- Closed-form linear least squares.
- Nonlinear least squares with `scipy.optimize.least_squares`.
- Constrained minimization with `scipy.optimize.minimize`.
- Robust least squares using SciPy loss options.
- Likelihood fitting for supported distributions.
- Global initialization with `differential_evolution` or `basinhopping` when configured.
- Multi-start deterministic local optimization.

### Parameter Model

Parameters must support:

- Initial value.
- Bounds.
- Transform, such as positive via log transform.
- Fixed flag.
- Shared/global flag.
- Entity-local flag.
- Regime-local flag.
- Prior or penalty.
- Unit.
- Description-length encoding cost.

### Optimizer Diagnostics

Every fit must record:

- Optimizer backend.
- Objective ID.
- Library versions.
- Seed.
- Initial parameters.
- Final parameters.
- Bounds.
- Loss value.
- Gradient/Jacobian summary where available.
- Iterations.
- Function evaluations.
- Convergence status.
- Failure reason.
- Rows used.
- Segment IDs used.

### Verification

Run:

```bash
pytest -q tests/unit/fit
pytest -q tests/unit/modules/test_candidate_fitting_enhanced.py
```

Expected:

- Known parameters are recovered on planted structures.
- Confirmatory rows are not accessed.
- Failed fits produce typed diagnostics, not silent rejection.

## 9. Phase 4: Search Engine Orchestration

### Objective

Replace monolithic search behavior with an engine portfolio.

### Replace Or Remove

Replace `search/backends.py` as the central implementation of all search classes.

Keep it temporarily as a thin facade while migrating tests and callers.

Remove equality saturation behavior that is only a sort key.

### Files

- Create: `src/euclid/search/engine_contracts.py`
- Create: `src/euclid/search/orchestration.py`
- Create: `src/euclid/search/engines/native.py`
- Create: `src/euclid/search/engines/pysindy_engine.py`
- Create: `src/euclid/search/engines/pysr_engine.py`
- Create: `src/euclid/search/engines/sparse_regression.py`
- Create: `src/euclid/search/engines/decomposition.py`
- Create: `src/euclid/search/engines/latent_state.py`
- Create: `src/euclid/search/engines/egraph.py`
- Modify: `src/euclid/search/backends.py`
- Modify: `src/euclid/search/portfolio.py`
- Modify: `src/euclid/search/frontier.py`
- Create: `tests/unit/search/test_engine_contracts.py`
- Create: `tests/unit/search/test_orchestration.py`
- Create: `tests/unit/search/engines/test_native_engine.py`

### Engine Contract

Each engine must return:

- Candidate expression IR.
- Proposed CIR or lowerable structure.
- Engine ID.
- Engine version.
- Search class.
- Search space declaration.
- Budget declaration.
- Rows and features used.
- Random seeds.
- Candidate trace.
- Omission disclosure.
- Failure diagnostics.

### Engine Types

Implement:

- Exact finite native engine.
- Bounded native engine.
- Stochastic native engine.
- PySINDy engine.
- PySR engine.
- Sparse-regression engine.
- Decomposition engine.
- Latent-state engine.
- Rewrite/e-graph postprocessor.

### Frontier Axes

Expand frontier axes to include:

- Total code bits.
- Structure code bits.
- Description gain bits.
- Fit loss.
- Out-of-sample score.
- Parameter count.
- Parameter stability.
- Support stability.
- Invariance score.
- Calibration score.
- Robustness score.
- Runtime cost.
- Rewrite complexity.

### Verification

Run:

```bash
pytest -q tests/unit/search
pytest -q tests/integration/test_multi_backend_cir_pipeline.py
```

Expected:

- All engines produce comparable candidate records.
- Search class disclosures are accurate.
- Frontier selection does not use confirmatory evidence.

## 10. Phase 5: PySINDy Backend

### Objective

Replace the SINDy label shim with a real sparse dynamics backend.

### Replace Or Remove

Remove the current `normalize_sparse_library_candidate` as the main implementation.

Do not keep a backend that merely rebinds recursive/spectral proposals as `sindy-sparse-library`.

### Files

- Replace: `src/euclid/adapters/sparse_library.py`
- Create: `src/euclid/search/engines/pysindy_engine.py`
- Create: `src/euclid/search/engines/pysindy_lowering.py`
- Create: `schemas/contracts/pysindy-engine-trace.yaml`
- Create: `tests/unit/search/engines/test_pysindy_engine.py`
- Create: `tests/integration/test_pysindy_pipeline.py`

### PySINDy Usage

Use:

- `pysindy.SINDy`
- `pysindy.PolynomialLibrary`
- `pysindy.FourierLibrary`
- `pysindy.CustomLibrary`
- `pysindy.STLSQ`
- `pysindy.SR3`
- PySINDy differentiation methods.
- Ensemble support where available.

### Steps

1. Build legal library terms from Euclid feature view and expression registry.
2. Estimate derivatives or discrete transitions under declared policy.
3. Fit PySINDy model on development rows only.
4. Extract active support.
5. Extract coefficients.
6. Lower terms to Euclid expression IR.
7. Build CIR.
8. Refit constants through Euclid fitting layer.
9. Score through Euclid MDL and validation gates.
10. Record PySINDy trace.

### Verification

Benchmarks:

- Logistic growth.
- Linear ODE.
- Damped oscillator.
- Lotka-Volterra.
- Noisy oscillator with weak-form option.

Expected:

- Correct support recovery on clean tasks.
- Honest abstention or weaker claim under high noise.
- Support stability recorded for ensembles.

## 11. Phase 6: PySR Backend

### Objective

Use PySR as a broad symbolic regression proposer, not as a publisher.

### Replace Or Remove

Do not preserve any future hand-written broad GP search if PySR handles it better.

Remove native duplicated evolutionary search unless it exists only for deterministic toy fragments.

### Files

- Create: `src/euclid/search/engines/pysr_engine.py`
- Create: `src/euclid/search/engines/pysr_lowering.py`
- Create: `schemas/contracts/pysr-engine-trace.yaml`
- Create: `tests/unit/search/engines/test_pysr_lowering.py`
- Create: `tests/integration/test_pysr_pipeline.py`

### PySR Usage

Use:

- `pysr.PySRRegressor`.
- Operator constraints.
- Nested constraints.
- Hall-of-fame output.
- Custom complexity mapping.
- Declared loss functions.

### Steps

1. Build PySR input matrix from legal feature view.
2. Restrict operators to Euclid-approved operator registry.
3. Fit PySR only on development rows.
4. Extract hall-of-fame formulas.
5. Parse formulas through SymPy.
6. Lower formulas into Euclid expression IR.
7. Refit constants through Euclid.
8. Normalize into CIR.
9. Score, falsify, and gate through Euclid.
10. Record Julia/PySR runtime metadata.

### Verification

Benchmarks:

- Feynman-style formulas.
- Nguyen-style formulas.
- Damped oscillator scalar observation.
- Saturating kinetics.
- Power law.
- Adversarial interpolation bait.

Expected:

- PySR candidate quality improves discovery breadth.
- Euclid still blocks overfit formulas that fail validation.

## 12. Phase 7: Real Rewrite And Equality Saturation

### Objective

Replace equality-saturation heuristic sorting with real rewrite expansion, equivalence, and extraction.

### Replace Or Remove

Remove `_equality_extractor_sort_key` as the implementation of equality saturation.

Remove any documentation that implies current equality saturation is a real e-graph engine until this phase lands.

### Files

- Create: `src/euclid/rewrites/rules.py`
- Create: `src/euclid/rewrites/sympy_simplifier.py`
- Create: `src/euclid/rewrites/egglog_runner.py`
- Create: `src/euclid/rewrites/extraction.py`
- Modify: `src/euclid/search/engines/egraph.py`
- Modify: `src/euclid/search/backends.py`
- Create: `schemas/contracts/rewrite-trace.yaml`
- Create: `tests/unit/rewrites/test_rules.py`
- Create: `tests/unit/rewrites/test_egglog_runner.py`

### Rewrite Rule Categories

Implement:

- Associative and commutative normalization.
- Constant folding.
- Neutral element removal.
- Distributive factoring where cost improves.
- Trigonometric identities under declared assumptions.
- Log/exp identities under positive-domain assumptions.
- Rational simplification with nonzero denominator assumptions.
- Piecewise simplification.
- Unit-preserving simplification.

### Verification

Run:

```bash
pytest -q tests/unit/rewrites
pytest -q tests/unit/search/test_equality_saturation_backend.py
```

Expected:

- Rewrites have side-condition evidence.
- Unsafe identities are rejected.
- Extracted forms have lower declared cost without changing semantics under assumptions.

## 13. Phase 8: Universal And Transport Claim Lanes

### Objective

Make universal-law claims first-class and separate from scoped predictive claims.

### Replace Or Remove

Remove any user-facing implication that `predictively_supported` means universal.

Replace `predictive_law` terminology where it does not include invariance evidence.

### Files

- Modify: `schemas/contracts/claim-lanes.yaml`
- Modify: `schemas/contracts/candidate-state-machine.yaml`
- Modify: `src/euclid/modules/claims.py`
- Modify: `src/euclid/modules/gate_lifecycle.py`
- Modify: `src/euclid/modules/catalog_publishing.py`
- Modify: `src/euclid/modules/replay.py`
- Modify: `src/euclid/workbench/service.py`
- Create: `src/euclid/invariance/environments.py`
- Create: `src/euclid/invariance/scoring.py`
- Create: `src/euclid/invariance/gates.py`
- Create: `schemas/contracts/invariance-evaluation.yaml`
- Create: `tests/unit/modules/test_claims_universal.py`
- Create: `tests/unit/invariance/test_gates.py`

### Claim Lanes

Implement:

- `descriptive_structure`
- `predictive_within_declared_scope`
- `invariant_predictive_law`
- `stochastic_law`
- `mechanistically_compatible_law`
- `transport_supported_law`

### Universal Claim Requirements

Require:

- At least two environments, entities, trajectories, regimes, or pseudo-environments.
- Shared symbolic structure.
- Parameter stability or declared parameter-variation law.
- Residual invariance.
- Out-of-environment validation.
- Falsification dossier.
- No unresolved time-safety issue.
- Replay-verified publication bundle.

### Verification

Expected:

- A single-series fit cannot publish universal unless pseudo-environments and invariance gates are explicitly satisfied.
- Cross-entity claims remain blocked unless transport evidence exists.
- Mechanistic evidence can raise only after predictive and invariance floors pass.

## 14. Phase 9: Shared-Structure Panel Discovery

### Objective

Replace in-sample shared-plus-local fitting with true shared-law discovery.

### Replace Or Remove

Remove use of in-sample panel mean offsets as evidence for shared law.

Remove hardcoded diagnostics such as `sharing_map: ["intercept"]` as final behavior.

### Files

- Replace: `src/euclid/modules/shared_plus_local_decomposition.py`
- Create: `src/euclid/panel/shared_structure.py`
- Create: `src/euclid/panel/partial_pooling.py`
- Create: `src/euclid/panel/leave_one_entity_out.py`
- Modify: `schemas/contracts/shared-plus-local-evaluation.yaml`
- Modify: `schemas/contracts/unseen-entity-prediction-policy.yaml`
- Create: `tests/unit/panel/test_shared_structure.py`
- Create: `tests/integration/test_shared_structure_law_pipeline.py`

### Model

A shared-structure candidate has:

- Shared symbolic skeleton.
- Global parameters.
- Entity-local parameters.
- Optional regime-local parameters.
- Parameter dispersion penalty.
- Local refit policy.
- Unseen-entity policy.
- Group support mask.

### Validation

Use:

- Leave-one-entity-out.
- Leave-one-trajectory-out.
- Parameter dispersion tests.
- Residual invariance across entities.
- Shared skeleton support stability.

### Verification

Expected:

- Within-panel memorization does not publish shared law.
- Same-skeleton/different-constant systems publish scoped shared-structure claims.
- Unseen-entity claims require explicit unseen-entity validation.

## 15. Phase 10: Stochastic Model Replacement

### Objective

Replace heuristic Gaussian probabilistic support with explicit stochastic law modeling.

### Replace Or Remove

Remove current production reliance on family-specific Gaussian scale inflation.

Remove hardcoded event probability threshold behavior.

### Files

- Replace: `src/euclid/math/observation_models.py`
- Replace or heavily modify: `src/euclid/modules/probabilistic_evaluation.py`
- Create: `src/euclid/stochastic/observation_models.py`
- Create: `src/euclid/stochastic/process_models.py`
- Create: `src/euclid/stochastic/scoring_rules.py`
- Create: `src/euclid/stochastic/calibration.py`
- Create: `src/euclid/stochastic/diagnostics.py`
- Modify: `src/euclid/modules/scoring.py`
- Create: `schemas/contracts/stochastic-law.yaml`
- Create: `schemas/contracts/event-definition.yaml`
- Create: `tests/unit/stochastic/test_observation_models.py`
- Create: `tests/unit/stochastic/test_scoring_rules.py`
- Create: `tests/integration/test_stochastic_law_pipeline.py`

### Observation Families

Support:

- Gaussian.
- Student-t.
- Laplace.
- Poisson.
- Negative binomial.
- Bernoulli.
- Beta.
- Lognormal.
- Mixture models.
- Bounded continuous distributions.

### Scale And Dispersion

Support:

- Global scale.
- Horizon-dependent scale.
- Feature-dependent scale.
- State-dependent scale.
- Symbolic scale law.
- Regime-specific scale.

### Event Definitions

Events must be manifests:

- Event ID.
- Variable.
- Operator.
- Threshold.
- Threshold source.
- Units.
- Scope.
- Calibration requirement.

### Verification

Expected:

- Event probabilities are not hardcoded.
- Distribution claims include likelihood evidence and calibration.
- Heavy-tail data can prefer Student-t over Gaussian under comparable scoring.

## 16. Phase 11: Statistical Promotion Gates

### Objective

Replace pure score improvement with statistically and practically justified promotion.

### Replace Or Remove

Remove `candidate_primary_score < baseline_primary_score` as sufficient promotion evidence.

### Files

- Modify: `src/euclid/modules/evaluation_governance.py`
- Modify: `src/euclid/modules/scoring.py`
- Create: `src/euclid/modules/predictive_tests.py`
- Create: `schemas/contracts/paired-predictive-test-result.yaml`
- Create: `schemas/contracts/prequential-score-stream.yaml`
- Create: `tests/unit/modules/test_predictive_tests.py`
- Create: `tests/integration/test_statistical_promotion_gate.py`

### Tests To Implement

Use SciPy/statsmodels for:

- Paired loss-difference tests.
- Block bootstrap confidence intervals.
- HAC/Newey-West adjusted intervals.
- Diebold-Mariano-style predictive comparison where applicable.
- Rolling/prequential degradation checks.
- Practical significance thresholds.

### Gate Logic

Promotion requires:

- Comparable score artifacts.
- Candidate beats baseline by practical margin.
- Confidence interval excludes unacceptable degradation.
- Many-model/search adjustment satisfied.
- No calibration blocker for probabilistic objects.
- No time-safety issue.

### Verification

Expected:

- Small noisy wins do not promote.
- Large consistent wins promote.
- Serial dependence is handled by declared correction.

## 17. Phase 12: Falsification And Residual Diagnostics

### Objective

Make strong claims survive active attempts to break them.

### Replace Or Remove

Remove robustness status that passes strong claims while null tests are missing or irrelevant.

Keep permissive robustness only for low-level descriptive claims if explicitly declared.

### Files

- Modify: `src/euclid/modules/robustness.py`
- Create: `src/euclid/falsification/residuals.py`
- Create: `src/euclid/falsification/surrogate_tests.py`
- Create: `src/euclid/falsification/perturbations.py`
- Create: `src/euclid/falsification/counterexamples.py`
- Create: `schemas/contracts/falsification-dossier.yaml`
- Create: `tests/unit/falsification/test_residuals.py`
- Create: `tests/integration/test_falsification_dossier.py`

### Diagnostics

Implement:

- Autocorrelation diagnostics.
- Heteroskedasticity diagnostics.
- Heavy-tail diagnostics.
- Regime residual drift.
- Calibration residual diagnostics.
- Influence/outlier sensitivity.

### Surrogates And Perturbations

Implement:

- IID permutation.
- Block permutation.
- Phase randomization.
- Missingness injection.
- Noise injection.
- Structural break insertion.
- Quantization perturbation.
- Window truncation.
- Environment swap.
- Entity holdout.

### Verification

Expected:

- Random-walk and interpolation bait tasks do not publish strong laws.
- Falsification failures lower claim ceilings.

## 18. Phase 13: Benchmark Universe Replacement

### Objective

Replace coverage-style benchmarks with law-discovery benchmarks.

### Replace Or Remove

Remove benchmark pass rules that only check artifact existence.

Remove benchmark tasks that imply capability without metric thresholds.

### Files

- Modify: `src/euclid/benchmarks/manifests.py`
- Modify: `src/euclid/benchmarks/runtime.py`
- Modify: `src/euclid/benchmarks/reporting.py`
- Modify: `src/euclid/readiness/judgment.py`
- Create many new manifests under `benchmarks/tasks`
- Create new suites under `benchmarks/suites`
- Create tests under `tests/benchmarks`

### Benchmark Families

Add:

- Classic symbolic regression.
- Feynman-style equations.
- Nguyen-style equations.
- Sparse ODE systems.
- Discrete-time recurrence systems.
- Multi-trajectory dynamics.
- Multi-entity shared laws.
- Hidden-state systems.
- Delay systems.
- Stochastic systems.
- Heavy-tail systems.
- Count systems.
- Bounded-response systems.
- Adversarial false-law systems.
- Transport and environment-shift systems.

### Metrics

Track:

- Exact symbolic recovery.
- Equivalent symbolic recovery.
- Parameter error.
- Support recovery.
- Rollout error.
- Extrapolation error.
- Out-of-environment error.
- Invariance score.
- Parameter stability.
- Calibration score.
- Robustness score.
- False-claim rate.
- Runtime.
- Seed sensitivity.

### Verification

Expected:

- Readiness rows require semantic thresholds.
- False-law benchmarks directly penalize overclaiming.
- Universal-law readiness requires multi-environment success.

## 19. Phase 14: Performance And Scaling Replacement

### Objective

Make expanded discovery computationally realistic.

### Replace Or Remove

Remove sequential-only candidate evaluation as the default.

Remove repeated evaluation of shared subexpressions.

### Files

- Modify: `src/euclid/performance.py`
- Modify: `src/euclid/runtime/profiling.py`
- Create: `src/euclid/runtime/parallel.py`
- Create: `src/euclid/runtime/cache.py`
- Create: `tests/perf/test_candidate_throughput.py`
- Create: `tests/perf/test_engine_runtime_budgets.py`

### Caching

Cache:

- Feature matrices.
- Expression evaluations.
- Subtree evaluations.
- Fitted parameter results.
- Residual vectors.
- SymPy simplification results.
- E-graph extracted forms.
- PySINDy/PySR candidate outputs.

### Parallelism

Use:

- joblib for local multiprocessing.
- Ray for distributed workloads.
- Deterministic candidate ordering for aggregation.

### Verification

Expected:

- Results are stable across worker counts.
- Runtime budgets appear in telemetry.
- Engine failures are isolated diagnostics, not run-level crashes unless required engine coverage is mandatory.

## 20. Phase 15: Workbench Replacement

### Objective

Replace formula-display UI with a scientific evidence studio.

### Replace Or Remove

Remove UI language that suggests a formula is a law without the claim lane to support it.

Replace opaque saved-analysis displays with evidence-oriented panels.

### Files

- Modify or replace: `src/euclid/workbench/service.py`
- Modify or replace: `src/euclid/workbench/server.py`
- Replace or heavily modify: `src/euclid/_assets/workbench/app.js`
- Replace or heavily modify: `src/euclid/_assets/workbench/app.css`
- Modify: `tests/unit/workbench/test_service.py`
- Modify: `tests/frontend/workbench/app.test.js`

### Views

Add:

- Candidate lineage.
- Engine provenance.
- CIR normalization view.
- Formula simplification trace.
- Optimizer diagnostics.
- Pareto frontier.
- Residual diagnostics.
- Calibration diagnostics.
- Invariance matrix.
- Entity/trajectory transfer matrix.
- Parameter stability plots.
- Falsification dossier.
- Claim ceiling explanation.
- Replay bundle browser.

### Verification

Expected:

- Workbench never creates claims.
- Workbench displays claim evidence and ceilings from backend artifacts.
- Descriptive, predictive, invariant, stochastic, mechanistic, and transport-supported claims are visually and textually distinct.

## 21. Phase 16: Documentation Replacement

### Objective

Make docs match the enhanced scientific system.

### Replace Or Remove

Remove or rewrite docs that describe retained-slice limitations as if they are the desired final architecture.

Remove ambiguous "law" terminology when the claim is only scoped predictive.

### Files

- Replace or heavily modify: `README.md`
- Replace or heavily modify: `docs/modeling-pipeline.md`
- Replace or heavily modify: `docs/search-core.md`
- Replace or heavily modify: `docs/benchmarks-readiness.md`
- Replace or heavily modify: `docs/workbench.md`
- Modify: `docs/contracts-manifests.md`
- Modify: `docs/testing-truthfulness.md`

### Required Docs

Add sections for:

- Expression IR.
- Library usage and numerical replay.
- Discovery engines.
- PySINDy integration.
- PySR integration.
- Universal claim lane.
- Stochastic law modeling.
- Invariance evidence.
- Falsification dossiers.
- Benchmark semantics.
- Workbench evidence inspection.

### Verification

Expected:

- Every documented claim lane maps to schema, code, tests, and benchmarks.
- Docs state what has been replaced and removed.

## 22. Global Test Matrix

The final enhanced system should pass:

```bash
./scripts/lint.sh
./scripts/test.sh
npm run test:frontend
pytest -q tests/unit/expr
pytest -q tests/unit/fit
pytest -q tests/unit/stochastic
pytest -q tests/unit/invariance
pytest -q tests/unit/panel
pytest -q tests/unit/rewrites
pytest -q tests/unit/falsification
pytest -q tests/unit/search
pytest -q tests/unit/cir
pytest -q tests/unit/modules
pytest -q tests/integration
pytest -q tests/benchmarks
pytest -q tests/perf
python -m euclid benchmarks run --suite current-release.yaml --no-resume
python -m euclid benchmarks run --suite full-vision.yaml --no-resume
python -m euclid release status
python -m euclid release verify-completion
python -m euclid release certify-research-readiness
```

## 23. Completion Criteria For The Whole Program

The enhancement program is complete only when:

1. Old shim backends are removed or demoted.
2. Search engines emit rich CIR candidates with real traces.
3. SciPy/SymPy-backed fitting and symbolic handling are the default.
4. PySINDy and PySR work as first-class engines.
5. Equality saturation is real or clearly not advertised.
6. Universal claims require invariance evidence.
7. Stochastic claims require explicit stochastic models.
8. Shared-structure claims require panel/entity validation.
9. Statistical promotion gates replace boolean score improvement.
10. Falsification dossiers exist for strong claims.
11. Benchmark readiness uses semantic thresholds.
12. Workbench displays evidence and claim ceilings.
13. Replay captures numerical library versions and solver settings.
14. Documentation states the enhanced architecture, not the retained-slice legacy.

## 24. Implementation Order

Implement in this order:

1. Evidence spine reset.
2. Dependency and numerical runtime foundation.
3. Expression IR replacement.
4. Unified fitting layer.
5. Search engine orchestration.
6. PySINDy backend.
7. PySR backend.
8. Real rewrite/equality saturation.
9. Universal and transport claim lanes.
10. Shared-structure panel discovery.
11. Stochastic model replacement.
12. Statistical promotion gates.
13. Falsification and residual diagnostics.
14. Benchmark universe replacement.
15. Performance and scaling replacement.
16. Workbench replacement.
17. Documentation replacement.

This order keeps Euclid honest at every step: first make the evidence spine reliable, then replace the math substrate, then add powerful engines, then raise claim standards, then expand benchmarks and UI around the stronger system.

## 25. Execution Control Rules

This section converts the roadmap into an execution plan. Future implementation work must use these phase and task IDs. If an implementation discovers that a task is wrong, update this plan first in the same change that justifies the deviation.

### 25.1 Task ID Format

Use:

```text
P<phase-number>-T<task-number>-S<subtask-number>
```

Examples:

- `P02-T03-S04`: Phase 2, Task 3, Subtask 4.
- `P10-T01`: Phase 10, Task 1.

Every commit message or PR description for this program should reference at least one task ID.

### 25.2 Execution Rules

- Execute phases in order unless this plan is updated.
- Do not implement new engine capability before the expression IR and unified fitting layer exist.
- Do not publish new claim lanes before the gates and benchmark semantics exist.
- Do not preserve deprecated behavior just because tests depend on it; update or delete the tests when the replacement is intentional.
- Do not leave shim labels that imply a real backend if the backend is still synthetic.
- Do not add a feature without replay metadata.
- Do not add a claim path without tests proving downgrade and abstention behavior.
- Do not add benchmarks that only prove successful execution; every benchmark must have semantic pass thresholds.
- Do not add UI copy that strengthens a claim beyond backend evidence.

### 25.3 Replacement Completion Rule

For every replaced subsystem, completion requires:

1. New implementation exists.
2. New tests prove the replacement.
3. Old tests are updated or deleted.
4. Old code path is removed or demoted behind an explicitly named legacy adapter.
5. Docs state the new behavior.
6. Release/readiness semantics no longer depend on the old behavior.

### 25.4 Phase Gate Template

Each phase has these gates:

- **Entry gate:** what must already be true before the phase begins.
- **Implementation gate:** what code/schema/docs must exist.
- **Replacement gate:** what old behavior must be deleted, disabled, or demoted.
- **Evidence gate:** what artifacts and replay metadata must prove.
- **Fixture test gate:** exact unit, integration, regression, golden, property-based, and edge-case tests expected to pass without network access.
- **Live API gate:** exact `.env`-backed live validation expected to pass against real external APIs without exposing secrets.
- **Exit gate:** what allows the next phase to begin.

### 25.5 Mandatory Dual-Gate Rule

Every phase, task, and subtask in this plan is gated by two independent proof channels:

1. **Fixture channel:** deterministic tests that run offline from checked-in fixtures, generated synthetic data, frozen benchmark manifests, and sanitized golden artifacts.
2. **Live API channel:** explicit live validation that loads credentials from `.env`, calls the real provider surface for the feature being proven, records sanitized semantic evidence, and fails closed when the provider response violates Euclid's contracts.

No implementation item may be marked complete unless both channels exist and have passed for that item. This is intentionally stricter than normal development practice. The enhanced Euclid must prove both reproducibility and live operational correctness.

For tasks whose code is not itself an API client, the live API gate still applies: the task must be exercised inside the nearest phase-level live pipeline that uses real `.env` credentials and proves the new code does not break ingestion, candidate generation, scoring, replay, evidence publication, or workbench presentation on live data. Example: a pure CIR canonicalization subtask is live-gated by running the live FMP ingestion pipeline through canonicalization, fitting, scoring, replay, and claim abstention/promotion semantics.

### 25.6 Required Test Gate Shape For Every Task

Every task and subtask must add or update a test-gate note with this shape:

```text
Fixture Unit Gate:
- Targeted deterministic tests for local logic.
- Hypothesis/property tests where input space matters.
- Edge cases proving typed failure modes, not generic exceptions.

Fixture Integration Gate:
- Offline end-to-end path using checked-in fixtures.
- Fixture includes normal data and at least one malformed or adversarial case.
- Artifact hashes and replay metadata are asserted.

Fixture Regression Gate:
- Golden artifact or benchmark row proving behavior remains stable.
- Previous bug/adversarial canary is preserved when relevant.
- Intentional replacement updates the golden with documented reason code.

Live API Gate:
- Loads `.env` through the approved env loader.
- Requires `EUCLID_LIVE_API_TESTS=1`.
- Uses provider keys such as `FMP_API_KEY` and `OPENAI_API_KEY` only through environment variables.
- Writes only sanitized evidence: provider name, endpoint class, timestamp, schema version, row counts, contract checks, latency bucket, and semantic pass/fail reason codes.
- Never writes keys, request authorization headers, raw secret-bearing URLs, prompt bodies containing secrets, or provider responses that licensing forbids storing.
- Exercises the implemented feature through the public runtime/CLI/workbench path, not by reaching into private helpers.

Edge-Case Gate:
- Covers empty input, single-row input, duplicated timestamps, out-of-order timestamps, missing values, nonfinite values, extreme magnitudes, timezone boundaries, rate limits, timeouts, malformed payloads, invalid credentials, provider schema drift, and replay mismatch where applicable.
```

The phase exit gate is invalid unless every task in that phase has this note and all corresponding commands pass.

### 25.7 Live API Secret Handling Policy

Live gates must use `.env` but must not commit `.env`.

The inline `EUCLID_LIVE_API_TESTS=1` and `EUCLID_LIVE_API_STRICT=1` prefixes shown in commands are execution switches. Provider credentials such as `FMP_API_KEY` and `OPENAI_API_KEY` must be read from `.env` through `src/euclid/runtime/env.py`, with process environment overrides allowed only for CI secret injection. No plan task should instruct developers to paste keys into commands.

Required files:

- Create or maintain `.env.example` with variable names only.
- Create or maintain `docs/reference/live-api-test-policy.md`.
- Create or maintain `tests/live/README.md`.
- Create or maintain `scripts/live_api_smoke.sh`.
- Create or maintain `src/euclid/runtime/env.py` as the only approved `.env` loading path.

Required environment variables:

- `EUCLID_LIVE_API_TESTS`: must be `1`, `true`, or `yes` for live tests to run.
- `EUCLID_LIVE_API_STRICT`: when enabled, missing or invalid keys fail the suite instead of skipping.
- `FMP_API_KEY`: required for live financial time-series ingestion and workbench FMP checks.
- `OPENAI_API_KEY`: required for live workbench explanation checks and any LLM-assisted explanation surface.
- `EUCLID_OPENAI_EXPLAINER_MODEL`: optional model override for live explanation checks.
- `EUCLID_LIVE_TEST_TIMEOUT_SECONDS`: optional per-test timeout override, bounded by the numerical policy.
- `EUCLID_LIVE_ARTIFACT_DIR`: optional output directory for sanitized live evidence.

Secret logging rules:

- Never print raw environment values.
- Never include query strings containing API keys in artifacts.
- Never write `.env` into release bundles.
- Redaction must be tested with fixtures containing fake secret-looking values.
- If a provider SDK logs secrets by default, wrap or configure it before it is allowed into live tests.

### 25.8 Live API Stability Rules

Live tests must account for volatility without becoming weak.

Allowed live assertions:

- Response shape matches schema.
- Required fields are present.
- Timestamps are parseable, ordered after normalization, and within expected freshness bounds.
- Numeric values are finite and inside broad sanity bounds for the domain.
- Split geometry is time-safe and does not leak future rows.
- Pipeline artifacts include provider identity, query window, row counts, and sanitized request metadata.
- Replay can reproduce the live run from the sanitized fixture snapshot where licensing allows recording.
- Claim gates abstain when live data is too short, too stale, too noisy, structurally inconsistent, or outside supported observation models.

Disallowed live assertions:

- Exact latest prices, volumes, model text, or stochastic values unless the provider response has been converted into a fixture.
- Assertions that depend on wall-clock timing narrower than the provider SLA.
- Assertions that silently skip malformed live responses.
- Assertions that promote a strong claim solely because a live API request succeeded.

### 25.9 Fixture Quality Rules

Fixture tests must be first-class, not placeholders.

Each feature must include:

- A minimal valid fixture.
- A realistic medium fixture.
- A pathological fixture.
- At least one adversarial honesty fixture when the feature can affect claims.
- A golden artifact fixture when the feature emits replay, benchmark, dossier, workbench, or publication output.

Fixture storage rules:

- Fixtures must be small enough for normal CI unless they are explicitly placed under a large-data optional suite.
- Fixture provenance must identify whether it is synthetic, sanitized live, public benchmark, or hand-authored adversarial data.
- Sanitized live fixtures must preserve schema and edge conditions while removing secrets and disallowed raw provider content.
- Regression fixtures must include the reason they exist and the behavior they protect.

### 25.10 Edge-Case Minimum Coverage

Every phase must cover the relevant subset of these edge cases:

- Data shape: empty series, one observation, two observations, very short train window, wide panel, single entity, many entities, missing entity, duplicate entity IDs.
- Time: duplicate timestamps, out-of-order timestamps, timezone-aware and timezone-naive input, DST transition, leap day, month end, market holiday, irregular cadence, late arriving revisions.
- Values: NaN, positive infinity, negative infinity, signed zero, subnormal floats, extreme magnitudes, constant series, near-constant series, spikes, heavy tails, negative values for positive-only laws, zero values for log/division laws.
- Search: no candidates, only invalid candidates, duplicate canonical candidates, engine timeout, engine crash, external engine unavailable, partial engine result, non-deterministic candidate ordering.
- Fitting: singular design matrix, underdetermined constants, failed convergence, parameter at bound, invalid domain during optimization, derivative estimation failure, stiff trajectory, high-noise trajectory.
- Stochastic: all residuals zero, extreme outlier, skewed residuals, count overdispersion, bounded response at 0 or 1, interval miscalibration, mixture component collapse.
- Claims: insufficient environments, unstable parameters, train-only improvement, failed transport, failed falsification, incomparable observation models, unsupported claim lane, stale evidence.
- Live API: missing key, blank key, invalid key, expired key, rate limit, timeout, 5xx, 429, malformed JSON, empty payload, provider schema drift, stale data, partial rows, duplicated rows, changed symbol semantics.
- Workbench: missing live key, invalid live key, slow provider, explanation unavailable, huge artifact, malformed artifact, browser refresh, secret redaction.

## 26. Phase Task Ledger

The following tasks are the canonical implementation ledger.

## 26.1 P00: Evidence Spine Reset

### Entry Gate

- Current checkout is clean except this plan.
- Existing release and benchmark behavior has been inspected.
- No new discovery engine work has started.

### P00-T01: Choose And Enforce Canonical Asset Policy

**Objective:** Make asset lookup deterministic and remove ambiguity between checkout assets and packaged assets.

**Files:**

- Modify: `src/euclid/operator_runtime/resources.py`
- Modify: `scripts/release_smoke.sh`
- Modify: `tests/unit/test_runtime_package_layout.py`
- Modify: `tests/unit/test_dev_scripts_smoke.py`
- Modify: `README.md`
- Modify: `docs/runtime-cli.md`

**Subtasks:**

- `P00-T01-S01`: Inventory every asset lookup path in runtime, release, benchmark, scripts, and tests.
- `P00-T01-S02`: Decide that packaged assets under `src/euclid/_assets` are canonical unless a root-level public mirror is explicitly documented.
- `P00-T01-S03`: Replace ad hoc root-relative lookups with a single resource resolver.
- `P00-T01-S04`: Make missing assets fail with typed error codes.
- `P00-T01-S05`: Delete tests that require obsolete root asset locations, or rewrite them to assert the canonical resolver.
- `P00-T01-S06`: Document the policy.

**Acceptance:**

- There is exactly one runtime resource-resolution policy.
- Tests no longer assume ambiguous duplicate asset roots.

### P00-T02: Add CI Contract Surface

**Objective:** If CI is part of release readiness, materialize it rather than letting tests expect a missing path.

**Files:**

- Create: `.github/workflows/ci.yml`
- Modify: `tests/unit/test_completion_regression_ci.py`
- Modify: `docs/testing-truthfulness.md`

**Subtasks:**

- `P00-T02-S01`: Decide the CI workflow is part of release evidence.
- `P00-T02-S02`: Add a CI workflow with lint, Python unit/integration tests, frontend tests, benchmark smoke, and release status smoke.
- `P00-T02-S03`: Ensure test expectations match the actual CI jobs.
- `P00-T02-S04`: Make CI workflow names stable because readiness may cite them.

**Acceptance:**

- CI file exists.
- Tests inspect real CI jobs, not fictional paths.

### P00-T03: Emit Explicit Operator Runtime Semantic Evidence

**Objective:** Completion reports must use explicit evidence, not inference.

**Files:**

- Modify: `src/euclid/cli/run.py`
- Modify: `src/euclid/operator_runtime/models.py`
- Modify: `src/euclid/release.py`
- Modify: `tests/unit/test_completion_report_logic.py`
- Modify: `tests/integration/test_cli_end_to_end.py`

**Subtasks:**

- `P00-T03-S01`: Define fields: `run_support_object_ids`, `admissibility_rule_ids`, `lifecycle_artifact_ids`, `operator_runtime_surface_ids`.
- `P00-T03-S02`: Add these fields to run evidence report models.
- `P00-T03-S03`: Populate fields from actual registered manifests and runtime decisions.
- `P00-T03-S04`: Make release completion consume fields directly.
- `P00-T03-S05`: Keep artifact inspection only as diagnostic fallback.
- `P00-T03-S06`: Add fail-closed reason codes for every missing semantic field.

**Acceptance:**

- Completion report rows cite explicit semantic evidence.
- Rows do not pass merely because a run completed.

### P00-T04: Emit Explicit Replay Semantic Evidence

**Objective:** Replay readiness must prove replay surfaces directly.

**Files:**

- Modify: `src/euclid/cli/replay.py`
- Modify: `src/euclid/modules/replay.py`
- Modify: `src/euclid/release.py`
- Modify: `tests/integration/test_operator_replay_pipeline.py`
- Modify: `tests/integration/test_completion_report_generation.py`

**Subtasks:**

- `P00-T04-S01`: Define `replay_surface_ids`, `replay_verified_artifact_ids`, and `replay_environment_ids`.
- `P00-T04-S02`: Include replay verification status and hash checks.
- `P00-T04-S03`: Include replay failure reason codes.
- `P00-T04-S04`: Connect replay evidence to readiness rows.

**Acceptance:**

- Replay evidence can independently close replay rows.
- Replay mismatch blocks publication/readiness.

### P00-T05: Convert Benchmark Surface Evidence From File Existence To Semantics

**Objective:** Benchmarks must prove what capability they exercised.

**Files:**

- Modify: `src/euclid/benchmarks/reporting.py`
- Modify: `src/euclid/benchmarks/runtime.py`
- Modify: `src/euclid/readiness/judgment.py`
- Modify: `tests/unit/benchmarks/test_reporting.py`
- Modify: `tests/benchmarks/test_full_vision_suite.py`

**Subtasks:**

- `P00-T05-S01`: Add `semantic_summary` to task reports.
- `P00-T05-S02`: Include support object IDs, claim lane IDs, replay IDs, engine IDs, score policy IDs, and threshold IDs.
- `P00-T05-S03`: Make suite status evaluate semantic summaries.
- `P00-T05-S04`: Remove pass logic based only on artifact presence.

**Acceptance:**

- A benchmark task with output files but failed semantic thresholds fails readiness.

### P00-T06: Establish Mandatory Dual Test Gate Harness

**Objective:** Create the shared harness that makes fixture tests and `.env`-backed live API tests mandatory for every later phase, task, and subtask.

**Files:**

- Create: `src/euclid/runtime/env.py`
- Create: `src/euclid/testing/live_api.py`
- Create: `src/euclid/testing/fixtures.py`
- Create: `src/euclid/testing/redaction.py`
- Create: `schemas/contracts/live-api-evidence.yaml`
- Create: `schemas/contracts/fixture-provenance.yaml`
- Create: `tests/unit/runtime/test_env_loading.py`
- Create: `tests/unit/testing/test_live_api_gate.py`
- Create: `tests/unit/testing/test_fixture_provenance.py`
- Create: `tests/unit/testing/test_secret_redaction.py`
- Create: `tests/integration/test_live_api_gate_fixture_mode.py`
- Create: `tests/regression/test_live_api_evidence_redaction.py`
- Create: `tests/live/README.md`
- Create: `tests/live/test_fmp_live_ingestion_smoke.py`
- Create: `tests/live/test_openai_live_explainer_smoke.py`
- Create: `scripts/live_api_smoke.sh`
- Create: `.env.example`
- Modify: `pyproject.toml`
- Modify: `docs/reference/runtime-cli.md`

**Subtasks:**

- `P00-T06-S01`: Define `EuclidEnv` that loads `.env` through `python-dotenv`, overlays process environment values, strips whitespace, validates required names, and exposes only presence metadata to artifacts.
- `P00-T06-S02`: Define `LiveApiGate` with provider name, credential variable names, endpoint class, timeout, strict/skip policy, semantic checks, and sanitized evidence output.
- `P00-T06-S03`: Define `FixtureGate` with fixture provenance, expected schema, edge-case class, golden artifact policy, and regression reason codes.
- `P00-T06-S04`: Define redaction utilities that remove API keys from headers, URLs, payloads, exception messages, tracebacks, and workbench artifacts.
- `P00-T06-S05`: Add `.env.example` containing `EUCLID_LIVE_API_TESTS`, `EUCLID_LIVE_API_STRICT`, `FMP_API_KEY`, `OPENAI_API_KEY`, `EUCLID_OPENAI_EXPLAINER_MODEL`, `EUCLID_LIVE_TEST_TIMEOUT_SECONDS`, and `EUCLID_LIVE_ARTIFACT_DIR` with blank values only.
- `P00-T06-S06`: Add pytest markers `unit`, `integration`, `regression`, `benchmark`, `perf`, and `live_api`.
- `P00-T06-S07`: Add `pytest-timeout` defaults so live and fixture tests cannot hang indefinitely.
- `P00-T06-S08`: Add `scripts/live_api_smoke.sh` that loads `.env`, verifies required variables according to strict mode, runs `tests/live`, and writes sanitized evidence to the configured artifact directory.
- `P00-T06-S09`: Add fixture-mode tests for the live harness using `responses` or `respx` so rate limits, timeouts, malformed JSON, invalid keys, and schema drift are deterministic.
- `P00-T06-S10`: Add regression tests proving fake secret-looking values never appear in logs, artifacts, pytest failure text, workbench responses, or release bundles.
- `P00-T06-S11`: Add live FMP smoke test using `FMP_API_KEY` that exercises public ingestion, frozen split geometry, candidate evaluation, replay metadata, and claim abstention/promotion semantics on a tiny bounded query.
- `P00-T06-S12`: Add live OpenAI smoke test using `OPENAI_API_KEY` that exercises only the workbench explanation path, verifies redaction and schema, and does not treat model prose as deterministic.
- `P00-T06-S13`: Document skip behavior: local live tests skip unless `EUCLID_LIVE_API_TESTS=1`; release certification runs strict mode and fails when required live keys are missing.
- `P00-T06-S14`: Add plan-lint/spec-compiler validation that every `PXX-TYY` and `PXX-TYY-SZZ` entry has fixture, integration/regression, live API, and edge-case gates before the phase may close.

**Acceptance:**

- Offline test harness can simulate live API success and failure modes deterministically.
- Live test harness can call real FMP and OpenAI paths when `.env` provides keys.
- Sanitized live evidence contains no secrets.
- Future phases have a reusable, mandatory gate template.

**Fixture Unit Gate:**

Run:

```bash
pytest -q tests/unit/runtime/test_env_loading.py tests/unit/testing/test_live_api_gate.py tests/unit/testing/test_fixture_provenance.py tests/unit/testing/test_secret_redaction.py
```

Required edge cases:

- Missing `.env`.
- Blank `.env` values.
- Process env overriding `.env`.
- Invalid boolean flag strings.
- Fake key in URL query string.
- Fake key in `Authorization` header.
- Fake key embedded in exception message.
- Fake key in nested JSON payload.

**Fixture Integration Gate:**

Run:

```bash
pytest -q tests/integration/test_live_api_gate_fixture_mode.py
```

Required edge cases:

- FMP returns HTTP 401.
- FMP returns HTTP 429.
- FMP returns HTTP 500.
- FMP returns malformed JSON.
- FMP returns empty history.
- FMP returns duplicate dates.
- FMP returns out-of-order rows.
- OpenAI response endpoint returns refusal-like content.
- OpenAI response endpoint returns tool-free text.
- OpenAI response endpoint times out.

**Fixture Regression Gate:**

Run:

```bash
pytest -q tests/regression/test_live_api_evidence_redaction.py
```

Required edge cases:

- Redaction remains stable when providers add new header names.
- Sanitized evidence schema remains backward-readable.
- Release bundle excludes `.env`.
- Golden live-evidence fixture contains provider metadata but no raw credentials.

**Live API Gate:**

Run with `.env` populated:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 ./scripts/live_api_smoke.sh
```

Required semantic checks:

- FMP live response validates against the ingestion schema.
- FMP live rows normalize into ordered, finite, time-safe observations.
- Live FMP run emits replay-visible provider, endpoint class, query window, row count, and schema version.
- Live FMP run does not publish a universal claim unless invariance gates exist and pass.
- OpenAI live explanation path returns a structured workbench explanation artifact or a typed abstention reason.
- OpenAI live evidence records model name and endpoint class but not prompt secrets or authorization headers.

**Edge-Case Gate:**

- Live gates must fail closed on invalid credentials in strict mode.
- Live gates must skip with explicit reason when disabled locally.
- Live gates must downgrade to abstention, not claim publication, when live data is too short, stale, malformed, or outside supported domain.

### P00 Exit Gate

Run:

```bash
pytest -q tests/unit/test_dev_scripts_smoke.py tests/unit/test_runtime_package_layout.py
pytest -q tests/unit/test_completion_report_logic.py tests/unit/benchmarks/test_reporting.py
pytest -q tests/integration/test_cli_end_to_end.py tests/integration/test_completion_report_generation.py
pytest -q tests/unit/runtime/test_env_loading.py tests/unit/testing/test_live_api_gate.py tests/unit/testing/test_fixture_provenance.py tests/unit/testing/test_secret_redaction.py
pytest -q tests/integration/test_live_api_gate_fixture_mode.py
pytest -q tests/regression/test_live_api_evidence_redaction.py
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 ./scripts/live_api_smoke.sh
python -m euclid smoke
python -m euclid release status
python -m euclid release verify-completion
```

P00 is complete when release and benchmark evidence are explicit, deterministic, and fail closed.

## 26.2 P01: Dependency And Numerical Runtime Foundation

### Entry Gate

- P00 is complete.
- Release evidence can record new runtime dependencies.

### P01-T01: Add Scientific Dependencies

**Objective:** Add trusted libraries as first-class dependencies.

**Files:**

- Modify: `pyproject.toml`
- Modify: `package-lock` only if frontend tooling requires no change should be untouched
- Modify: `scripts/install.sh`
- Modify: `scripts/install_smoke.sh`
- Modify: `docs/runtime-cli.md`

**Subtasks:**

- `P01-T01-S01`: Add SciPy.
- `P01-T01-S02`: Add SymPy.
- `P01-T01-S03`: Add Pint.
- `P01-T01-S04`: Add statsmodels.
- `P01-T01-S05`: Add scikit-learn.
- `P01-T01-S06`: Add PySINDy.
- `P01-T01-S07`: Add PySR and document Julia runtime requirements.
- `P01-T01-S08`: Add egglog.
- `P01-T01-S09`: Add joblib.
- `P01-T01-S10`: Add Hypothesis to dev dependencies.
- `P01-T01-S11`: Add `python-dotenv` as the only approved `.env` loading dependency.
- `P01-T01-S12`: Add `httpx` for live API clients and fixture-mode HTTP transport tests.
- `P01-T01-S13`: Add `responses` and/or `respx` for deterministic fixture-mode API failure tests.
- `P01-T01-S14`: Add `vcrpy` only for sanitized regression cassettes where provider licensing permits response recording.
- `P01-T01-S15`: Add `pytest-timeout` so slow live providers and optimizer failures cannot hang CI.
- `P01-T01-S16`: Add `pytest-xdist` for parallel fixture suites, while keeping live API tests serialized by provider and key.
- `P01-T01-S17`: Decide whether Ray is core or optional based on install stability; if optional, document the distributed extra.

**Acceptance:**

- Clean install includes scientific stack.
- Import smoke verifies all core packages.
- Test import smoke verifies `python-dotenv`, `httpx`, `responses` or `respx`, `vcrpy`, `pytest-timeout`, and `pytest-xdist`.
- Dependency policy explains which libraries are runtime dependencies and which are test-only dependencies.

### P01-T02: Numerical Environment Capture

**Objective:** Replay must know which numerical libraries produced artifacts.

**Files:**

- Create: `src/euclid/runtime/numerical_environment.py`
- Modify: `src/euclid/modules/replay.py`
- Modify: `src/euclid/runtime/profiling.py`
- Create: `schemas/contracts/numerical-runtime.yaml`
- Create: `tests/unit/runtime/test_numerical_environment.py`

**Subtasks:**

- `P01-T02-S01`: Capture Python version.
- `P01-T02-S02`: Capture NumPy, pandas, SciPy, SymPy, Pint, statsmodels, scikit-learn, PySINDy, PySR, egglog versions.
- `P01-T02-S03`: Capture Julia version when PySR is initialized.
- `P01-T02-S04`: Capture BLAS/LAPACK information where available.
- `P01-T02-S05`: Include platform and CPU metadata.
- `P01-T02-S06`: Add numerical environment to replay bundles.
- `P01-T02-S07`: Add mismatch diagnostics during replay.

**Acceptance:**

- Replay bundle can explain numerical runtime provenance.

### P01-T03: Numerical Policy Manifest

**Objective:** Standardize numerical tolerances and solver policies.

**Files:**

- Create: `schemas/contracts/numerical-policy.yaml`
- Create: `src/euclid/manifests/numerical.py`
- Create: `tests/unit/manifests/test_numerical_policy.py`

**Subtasks:**

- `P01-T03-S01`: Define default absolute/relative tolerances.
- `P01-T03-S02`: Define optimizer iteration defaults.
- `P01-T03-S03`: Define deterministic seed derivation.
- `P01-T03-S04`: Define failure thresholds.
- `P01-T03-S05`: Define allowed numerical instability downgrades.

**Acceptance:**

- Fitting and search can cite one numerical policy.

### P01 Exit Gate

Run:

```bash
pytest -q tests/unit/runtime/test_numerical_environment.py tests/unit/manifests/test_numerical_policy.py
python -m euclid smoke
python -m euclid release certify-clean-install
```

P01 is complete when scientific dependencies are installed, importable, documented, and replay-visible.

## 26.3 P02: Expression IR Replacement

### Entry Gate

- P01 is complete.
- Numerical and symbolic library versions are replay-visible.

### P02-T01: Define Expression AST

**Objective:** Create Euclid-owned typed symbolic expression nodes.

**Files:**

- Create: `src/euclid/expr/ast.py`
- Create: `tests/unit/expr/test_ast.py`

**Subtasks:**

- `P02-T01-S01`: Implement immutable dataclasses or Pydantic models for expression nodes.
- `P02-T01-S02`: Add `Literal`, `Parameter`, `Feature`, `Lag`, `Delay`, `State`.
- `P02-T01-S03`: Add `Derivative`, `Integral`, `UnaryOp`, `BinaryOp`, `NaryOp`.
- `P02-T01-S04`: Add `Conditional`, `Piecewise`, `NoiseTerm`, `DistributionParameter`.
- `P02-T01-S05`: Add canonical child ordering for commutative nodes.
- `P02-T01-S06`: Add stable serialization.

**Acceptance:**

- Expression nodes are immutable and hash-stable.

### P02-T02: Define Operator Registry

**Objective:** Make all math operators explicit and metadata-backed.

**Files:**

- Create: `src/euclid/expr/operators.py`
- Create: `tests/unit/expr/test_operators.py`

**Subtasks:**

- `P02-T02-S01`: Define operator metadata schema.
- `P02-T02-S02`: Add arithmetic operators.
- `P02-T02-S03`: Add protected math operators.
- `P02-T02-S04`: Add trigonometric and exponential operators.
- `P02-T02-S05`: Add bounded transforms such as sigmoid/logit.
- `P02-T02-S06`: Add time-series operators.
- `P02-T02-S07`: Add stochastic parameter operators.
- `P02-T02-S08`: Add domain, unit, differentiability, singularity, and SymPy mapping metadata.

**Acceptance:**

- No operator can be evaluated or serialized without metadata.

### P02-T03: Add SymPy Bridge

**Objective:** Use SymPy for parsing, simplification, and symbolic equivalence.

**Files:**

- Create: `src/euclid/expr/sympy_bridge.py`
- Create: `tests/unit/expr/test_sympy_bridge.py`

**Subtasks:**

- `P02-T03-S01`: Convert expression IR to SymPy.
- `P02-T03-S02`: Convert supported SymPy expressions to expression IR.
- `P02-T03-S03`: Implement simplification under assumptions.
- `P02-T03-S04`: Implement symbolic derivative support.
- `P02-T03-S05`: Implement safe equivalence checks.
- `P02-T03-S06`: Reject unsupported SymPy functions with typed diagnostics.

**Acceptance:**

- PySR/SymPy formulas can lower into Euclid IR.
- Domain-sensitive expressions do not collapse incorrectly.

### P02-T04: Add Units And Domains

**Objective:** Make dimensional consistency part of expression admissibility.

**Files:**

- Create: `src/euclid/expr/domains.py`
- Create: `src/euclid/expr/units.py`
- Create: `tests/unit/expr/test_units.py`

**Subtasks:**

- `P02-T04-S01`: Add domain types: real, positive real, nonnegative real, integer, probability, bounded interval.
- `P02-T04-S02`: Add Pint unit registry wrapper.
- `P02-T04-S03`: Add operator unit propagation rules.
- `P02-T04-S04`: Add unit compatibility diagnostics.
- `P02-T04-S05`: Add unknown-unit handling policy.

**Acceptance:**

- Unit-invalid expressions are blocked before fitting.

### P02-T05: Integrate Expression IR With CIR

**Objective:** CIR identity must include expression structure and assumptions.

**Files:**

- Modify: `src/euclid/cir/models.py`
- Modify: `src/euclid/cir/normalize.py`
- Modify: `tests/unit/cir/test_cir_models.py`
- Modify: `tests/unit/cir/test_cir_normalize.py`

**Subtasks:**

- `P02-T05-S01`: Add expression payload to structural layer.
- `P02-T05-S02`: Add assumption set to canonical bytes.
- `P02-T05-S03`: Add unit/domain constraints to execution/admissibility layers.
- `P02-T05-S04`: Ensure backend provenance remains outside canonical identity.
- `P02-T05-S05`: Add tests for same expression from different engines normalizing together.

**Acceptance:**

- CIR can represent expressive symbolic laws without string identity.

### P02-T06: Retire Old DSL As Primary Runtime

**Objective:** Demote old algorithmic DSL to a reader/lowering path.

**Files:**

- Modify: `src/euclid/search/dsl/*`
- Modify: `src/euclid/modules/algorithmic_dsl.py`
- Modify: `tests/unit/search/dsl/*`

**Subtasks:**

- `P02-T06-S01`: Add lowering from old DSL programs to new expression IR.
- `P02-T06-S02`: Mark old DSL enumerator as legacy in docs/tests.
- `P02-T06-S03`: Remove old DSL as the default proposal generator after native engine replacement.

**Acceptance:**

- No new code depends on old DSL as the main expression model.

### P02 Exit Gate

Run:

```bash
pytest -q tests/unit/expr
pytest -q tests/unit/cir
pytest -q tests/unit/search/dsl
```

P02 is complete when expression IR is the canonical law structure and old DSL is no longer the primary substrate.

## 26.4 P03: Unified Fitting Layer

### Entry Gate

- P02 is complete.
- CIR can carry expression IR.

### P03-T01: Define Parameterization Model

**Files:**

- Create: `src/euclid/fit/parameterization.py`
- Create: `tests/unit/fit/test_parameterization.py`

**Subtasks:**

- `P03-T01-S01`: Define parameter declaration model.
- `P03-T01-S02`: Support bounds, transforms, units, fixed/shared/local flags.
- `P03-T01-S03`: Support parameter penalties and priors.
- `P03-T01-S04`: Add serialization into candidate fit artifacts.

**Acceptance:**

- Every fitted constant has a declaration and replay-visible constraints.

### P03-T02: Define Objective Registry

**Files:**

- Create: `src/euclid/fit/objectives.py`
- Create: `tests/unit/fit/test_objectives.py`

**Subtasks:**

- `P03-T02-S01`: Implement squared error.
- `P03-T02-S02`: Implement absolute error.
- `P03-T02-S03`: Implement Huber and robust SciPy-compatible residuals.
- `P03-T02-S04`: Implement negative log likelihood objectives for stochastic families.
- `P03-T02-S05`: Add regularization penalties.

**Acceptance:**

- Objectives expose residual and scalar-loss interfaces.

### P03-T03: Implement SciPy Optimizers

**Files:**

- Create: `src/euclid/fit/scipy_optimizers.py`
- Create: `tests/unit/fit/test_scipy_optimizers.py`

**Subtasks:**

- `P03-T03-S01`: Wrap `scipy.optimize.least_squares`.
- `P03-T03-S02`: Wrap `scipy.optimize.minimize`.
- `P03-T03-S03`: Add multi-start deterministic initialization.
- `P03-T03-S04`: Add optional global initialization with `differential_evolution`.
- `P03-T03-S05`: Capture convergence diagnostics.
- `P03-T03-S06`: Convert numerical failures to typed diagnostics.

**Acceptance:**

- Optimizer results are deterministic under declared seed policy.

### P03-T04: Replace Candidate Fitting

**Files:**

- Modify: `src/euclid/modules/candidate_fitting.py`
- Modify: `src/euclid/search/descriptive_coding.py`
- Modify: `tests/unit/modules/test_candidate_fitting.py`
- Create: `tests/unit/modules/test_candidate_fitting_enhanced.py`

**Subtasks:**

- `P03-T04-S01`: Route expression candidates through unified fitter.
- `P03-T04-S02`: Preserve analytic linear closed form as internal fast path.
- `P03-T04-S03`: Preserve spectral least squares as internal fast path.
- `P03-T04-S04`: Remove search-time ad hoc parameter estimation.
- `P03-T04-S05`: Ensure fit rows derive only from declared segment.
- `P03-T04-S06`: Include optimizer diagnostics in candidate artifacts.

**Acceptance:**

- Candidate fitting no longer relies on hardcoded family switch for all new candidates.

### P03-T05: Remove Duplicate Fitting Logic From Search

**Files:**

- Modify: `src/euclid/search/backends.py`
- Modify: `src/euclid/search/engines/native.py`

**Subtasks:**

- `P03-T05-S01`: Remove `_lag_1_affine_fit` as search proposal fitting.
- `P03-T05-S02`: Search proposes structure and initial guesses only.
- `P03-T05-S03`: Fitting layer owns final parameter values.

**Acceptance:**

- Search and fitting have clean ownership boundaries.

### P03 Exit Gate

Run:

```bash
pytest -q tests/unit/fit
pytest -q tests/unit/modules/test_candidate_fitting.py tests/unit/modules/test_candidate_fitting_enhanced.py
pytest -q tests/integration/test_phase05_candidate_fitting.py
```

P03 is complete when parameter fitting is general, SciPy-backed, and replay-visible.

## 26.5 P04: Search Engine Orchestration

### Entry Gate

- P03 is complete.
- Search can emit expression skeletons and use unified fitting.

### P04-T01: Define Engine Contract

**Files:**

- Create: `src/euclid/search/engine_contracts.py`
- Create: `tests/unit/search/test_engine_contracts.py`

**Subtasks:**

- `P04-T01-S01`: Define engine input context.
- `P04-T01-S02`: Define engine output candidate record.
- `P04-T01-S03`: Define engine trace and omission disclosure.
- `P04-T01-S04`: Define engine failure diagnostics.
- `P04-T01-S05`: Define rows/features access contract.

**Acceptance:**

- Every engine implements the same proposal interface.

### P04-T02: Create Orchestrator

**Files:**

- Create: `src/euclid/search/orchestration.py`
- Create: `tests/unit/search/test_orchestration.py`

**Subtasks:**

- `P04-T02-S01`: Run configured engines.
- `P04-T02-S02`: Collect engine traces.
- `P04-T02-S03`: Lower candidates into CIR.
- `P04-T02-S04`: Refit candidates.
- `P04-T02-S05`: Deduplicate by CIR hash.
- `P04-T02-S06`: Build frontier metrics.
- `P04-T02-S07`: Emit search evidence.

**Acceptance:**

- Multiple engines can run in one governed search plan.

### P04-T03: Replace Native Search Backend

**Files:**

- Create: `src/euclid/search/engines/native.py`
- Modify: `src/euclid/search/backends.py`
- Modify: `tests/unit/search/test_backends.py`

**Subtasks:**

- `P04-T03-S01`: Implement exact native fragment engine.
- `P04-T03-S02`: Implement bounded native fragment engine.
- `P04-T03-S03`: Implement stochastic native fragment engine.
- `P04-T03-S04`: Move proposal logic out of monolithic backend.
- `P04-T03-S05`: Keep `backends.py` only as facade during migration.

**Acceptance:**

- Old monolithic proposal pool is not the core engine.

### P04-T04: Expand Frontier Axes

**Files:**

- Modify: `src/euclid/search/frontier.py`
- Modify: `src/euclid/search/policies.py`
- Modify: `tests/unit/search/test_frontier.py`

**Subtasks:**

- `P04-T04-S01`: Add fit loss.
- `P04-T04-S02`: Add out-of-sample score.
- `P04-T04-S03`: Add support stability.
- `P04-T04-S04`: Add parameter stability.
- `P04-T04-S05`: Add invariance score placeholder.
- `P04-T04-S06`: Add robustness and calibration placeholders.
- `P04-T04-S07`: Enforce comparability for every active axis.

**Acceptance:**

- Frontier can rank scientific evidence, not only description bits.

### P04 Exit Gate

Run:

```bash
pytest -q tests/unit/search
pytest -q tests/integration/test_multi_backend_cir_pipeline.py
pytest -q tests/integration/test_search_honesty_pipeline.py
```

P04 is complete when engine orchestration replaces the monolithic search implementation.

## 26.6 P05: PySINDy Backend

### Entry Gate

- P04 is complete.
- Engine contract and expression IR are stable.

### P05-T01: Implement PySINDy Engine Configuration

**Files:**

- Create: `src/euclid/search/engines/pysindy_engine.py`
- Create: `schemas/contracts/pysindy-engine-trace.yaml`
- Create: `tests/unit/search/engines/test_pysindy_engine.py`

**Subtasks:**

- `P05-T01-S01`: Define PySINDy engine config.
- `P05-T01-S02`: Map Euclid feature policies to PySINDy libraries.
- `P05-T01-S03`: Support polynomial libraries.
- `P05-T01-S04`: Support Fourier libraries.
- `P05-T01-S05`: Support custom libraries from expression operator registry.
- `P05-T01-S06`: Record differentiation and optimizer config.

**Acceptance:**

- PySINDy engine can be configured from Euclid search plan.

### P05-T02: Implement PySINDy Fitting And Trace Capture

**Subtasks:**

- `P05-T02-S01`: Build development-only matrices.
- `P05-T02-S02`: Fit PySINDy model.
- `P05-T02-S03`: Capture coefficients.
- `P05-T02-S04`: Capture support mask.
- `P05-T02-S05`: Capture optimizer diagnostics.
- `P05-T02-S06`: Capture differentiation diagnostics.
- `P05-T02-S07`: Capture ensemble inclusion probabilities when used.

**Acceptance:**

- PySINDy trace is complete enough for replay and audit.

### P05-T03: Lower PySINDy Model To Euclid IR

**Files:**

- Create: `src/euclid/search/engines/pysindy_lowering.py`
- Create: `tests/unit/search/engines/test_pysindy_lowering.py`

**Subtasks:**

- `P05-T03-S01`: Convert active terms to expression IR.
- `P05-T03-S02`: Convert coefficients to Euclid parameters.
- `P05-T03-S03`: Build state update law.
- `P05-T03-S04`: Build CIR candidate.
- `P05-T03-S05`: Refit through Euclid unified fitter.

**Acceptance:**

- PySINDy output is a Euclid candidate, not an external model blob.

### P05-T04: Remove SINDy Label Shim

**Files:**

- Replace: `src/euclid/adapters/sparse_library.py`
- Modify: `tests/unit/cir/test_backend_adapters.py`

**Subtasks:**

- `P05-T04-S01`: Delete relabel-only sparse-library behavior.
- `P05-T04-S02`: Redirect adapter imports to real PySINDy backend where needed.
- `P05-T04-S03`: Update tests that expected wrapper behavior.

**Acceptance:**

- No code path claims SINDy provenance without running or importing a real SINDy trace.

### P05 Exit Gate

Run:

```bash
pytest -q tests/unit/search/engines/test_pysindy_engine.py tests/unit/search/engines/test_pysindy_lowering.py
pytest -q tests/integration/test_pysindy_pipeline.py
```

P05 is complete when PySINDy can discover, lower, refit, score, and replay sparse dynamics.

## 26.7 P06: PySR Backend

### Entry Gate

- P05 is complete.
- External engine traces are supported.

### P06-T01: Implement PySR Engine Configuration

**Files:**

- Create: `src/euclid/search/engines/pysr_engine.py`
- Create: `schemas/contracts/pysr-engine-trace.yaml`
- Create: `tests/unit/search/engines/test_pysr_engine.py`

**Subtasks:**

- `P06-T01-S01`: Define operator mapping from Euclid registry to PySR.
- `P06-T01-S02`: Define complexity mapping.
- `P06-T01-S03`: Define nested operator constraints.
- `P06-T01-S04`: Define loss mapping.
- `P06-T01-S05`: Capture PySR and Julia runtime metadata.

**Acceptance:**

- PySR receives only Euclid-approved operators and features.

### P06-T02: Parse Hall Of Fame

**Files:**

- Create: `src/euclid/search/engines/pysr_lowering.py`
- Create: `tests/unit/search/engines/test_pysr_lowering.py`

**Subtasks:**

- `P06-T02-S01`: Extract hall-of-fame expressions.
- `P06-T02-S02`: Parse expressions through SymPy.
- `P06-T02-S03`: Lower expressions to Euclid IR.
- `P06-T02-S04`: Convert PySR constants to Euclid parameters.
- `P06-T02-S05`: Refit constants through Euclid.

**Acceptance:**

- PySR formulas become normal Euclid CIR candidates.

### P06-T03: Remove Any Future Native GP Duplication

**Subtasks:**

- `P06-T03-S01`: Audit native stochastic search for duplicated PySR behavior.
- `P06-T03-S02`: Keep native stochastic search only for bounded deterministic fragments or testing.
- `P06-T03-S03`: Update docs to state PySR owns broad symbolic regression.

**Acceptance:**

- Euclid does not maintain an inferior duplicate GP engine.

### P06 Exit Gate

Run:

```bash
pytest -q tests/unit/search/engines/test_pysr_engine.py tests/unit/search/engines/test_pysr_lowering.py
pytest -q tests/integration/test_pysr_pipeline.py
```

P06 is complete when PySR proposes candidates that Euclid can normalize, refit, validate, and replay.

## 26.8 P07: Rewrite And Equality Saturation

### Entry Gate

- P06 is complete.
- SymPy bridge and expression assumptions are stable.

### P07-T01: Implement Rewrite Rule Registry

**Files:**

- Create: `src/euclid/rewrites/rules.py`
- Create: `tests/unit/rewrites/test_rules.py`

**Subtasks:**

- `P07-T01-S01`: Define rewrite rule schema.
- `P07-T01-S02`: Add algebraic rules.
- `P07-T01-S03`: Add trigonometric rules.
- `P07-T01-S04`: Add log/exp rules.
- `P07-T01-S05`: Add rational simplification rules.
- `P07-T01-S06`: Add unit-preserving constraints.

**Acceptance:**

- Every rule has assumptions and side conditions.

### P07-T02: Implement SymPy Simplification Backend

**Files:**

- Create: `src/euclid/rewrites/sympy_simplifier.py`
- Create: `tests/unit/rewrites/test_sympy_simplifier.py`

**Subtasks:**

- `P07-T02-S01`: Run SymPy simplification under assumptions.
- `P07-T02-S02`: Verify equivalence where possible.
- `P07-T02-S03`: Emit simplification trace.

**Acceptance:**

- Simplification is traceable and assumption-aware.

### P07-T03: Implement egglog Backend

**Files:**

- Create: `src/euclid/rewrites/egglog_runner.py`
- Create: `src/euclid/rewrites/extraction.py`
- Create: `tests/unit/rewrites/test_egglog_runner.py`

**Subtasks:**

- `P07-T03-S01`: Translate Euclid IR to egglog terms.
- `P07-T03-S02`: Run saturation with limits.
- `P07-T03-S03`: Extract lowest-cost expression.
- `P07-T03-S04`: Record e-class counts and saturation status.
- `P07-T03-S05`: Convert extracted term back to Euclid IR.

**Acceptance:**

- E-graph output is replayable and bounded.

### P07-T04: Replace Equality Sort Heuristic

**Files:**

- Modify: `src/euclid/search/backends.py`
- Modify: `src/euclid/search/engines/egraph.py`
- Modify: `tests/unit/search/test_equality_saturation_backend.py`

**Subtasks:**

- `P07-T04-S01`: Delete `_equality_extractor_sort_key` as equality implementation.
- `P07-T04-S02`: Route equality-saturation search class to rewrite/e-graph engine.
- `P07-T04-S03`: Update coverage disclosures with real rewrite metrics.

**Acceptance:**

- Equality saturation is real or unavailable; never a sort-only label.

### P07 Exit Gate

Run:

```bash
pytest -q tests/unit/rewrites
pytest -q tests/unit/search/test_equality_saturation_backend.py
```

P07 is complete when rewrite/equality behavior is mathematically traceable and no longer label-only.

## 26.9 P08: Universal And Transport Claim Lanes

### Entry Gate

- P07 is complete.
- Search can discover and simplify candidates from multiple engines.

### P08-T01: Redefine Claim Taxonomy

**Files:**

- Modify: `schemas/contracts/claim-lanes.yaml`
- Modify: `schemas/contracts/enum-registry.yaml`
- Modify: `src/euclid/modules/claims.py`
- Create: `tests/unit/modules/test_claims_universal.py`

**Subtasks:**

- `P08-T01-S01`: Add `descriptive_structure`.
- `P08-T01-S02`: Add `predictive_within_declared_scope`.
- `P08-T01-S03`: Add `invariant_predictive_law`.
- `P08-T01-S04`: Add `stochastic_law`.
- `P08-T01-S05`: Add `mechanistically_compatible_law`.
- `P08-T01-S06`: Add `transport_supported_law`.
- `P08-T01-S07`: Remove or rename ambiguous `predictive_law` wording.

**Acceptance:**

- Scoped prediction and universal law are separate claim ceilings.

### P08-T02: Implement Environment Construction

**Files:**

- Create: `src/euclid/invariance/environments.py`
- Create: `tests/unit/invariance/test_environments.py`

**Subtasks:**

- `P08-T02-S01`: Support explicit environment labels.
- `P08-T02-S02`: Support entity-based environments.
- `P08-T02-S03`: Support rolling-era pseudo-environments.
- `P08-T02-S04`: Support volatility/regime pseudo-environments.
- `P08-T02-S05`: Support known intervention windows.
- `P08-T02-S06`: Record environment construction policy.

**Acceptance:**

- Invariance tests know exactly what environments mean.

### P08-T03: Implement Invariance Scoring

**Files:**

- Create: `src/euclid/invariance/scoring.py`
- Create: `src/euclid/invariance/gates.py`
- Create: `schemas/contracts/invariance-evaluation.yaml`
- Create: `tests/unit/invariance/test_gates.py`

**Subtasks:**

- `P08-T03-S01`: Compute residual invariance.
- `P08-T03-S02`: Compute parameter stability.
- `P08-T03-S03`: Compute support stability.
- `P08-T03-S04`: Compute environment holdout scores.
- `P08-T03-S05`: Apply thresholds.
- `P08-T03-S06`: Emit invariance evaluation manifest.

**Acceptance:**

- Universal claims require passed invariance evaluation.

### P08-T04: Wire Claim Gates

**Files:**

- Modify: `src/euclid/modules/gate_lifecycle.py`
- Modify: `src/euclid/modules/catalog_publishing.py`
- Modify: `src/euclid/modules/replay.py`
- Modify: `src/euclid/operator_runtime/workflow.py`

**Subtasks:**

- `P08-T04-S01`: Add invariance status to scorecard.
- `P08-T04-S02`: Add transport status to scorecard.
- `P08-T04-S03`: Block universal lane without invariance.
- `P08-T04-S04`: Block transport lane without out-of-environment evidence.
- `P08-T04-S05`: Preserve downgrade/abstention semantics.

**Acceptance:**

- Claims cannot overstate scope.

### P08 Exit Gate

Run:

```bash
pytest -q tests/unit/invariance tests/unit/modules/test_claims_universal.py
pytest -q tests/integration/test_publication_pipeline.py
```

P08 is complete when universal and transport claims are explicit, gated, and replayable.

## 26.10 P09: Shared-Structure Panel Discovery

### Entry Gate

- P08 is complete.
- Universal claims can consume invariance evidence.

### P09-T01: Replace Shared-Plus-Local Fitter

**Files:**

- Replace: `src/euclid/modules/shared_plus_local_decomposition.py`
- Create: `src/euclid/panel/shared_structure.py`
- Create: `tests/unit/panel/test_shared_structure.py`

**Subtasks:**

- `P09-T01-S01`: Define shared symbolic skeleton model.
- `P09-T01-S02`: Define global parameters.
- `P09-T01-S03`: Define entity-local parameters.
- `P09-T01-S04`: Define regime-local parameters.
- `P09-T01-S05`: Add dispersion penalties.
- `P09-T01-S06`: Remove mean-offset baseline as law evidence.

**Acceptance:**

- Panel law evidence is based on shared structure, not in-sample offsets.

### P09-T02: Implement Partial Pooling

**Files:**

- Create: `src/euclid/panel/partial_pooling.py`
- Create: `tests/unit/panel/test_partial_pooling.py`

**Subtasks:**

- `P09-T02-S01`: Implement ridge/shrinkage parameter fitting.
- `P09-T02-S02`: Implement group support penalties.
- `P09-T02-S03`: Record pooling strength.
- `P09-T02-S04`: Record parameter dispersion.

**Acceptance:**

- Shared/local decomposition has statistical structure.

### P09-T03: Implement Entity Holdout

**Files:**

- Create: `src/euclid/panel/leave_one_entity_out.py`
- Create: `tests/unit/panel/test_leave_one_entity_out.py`

**Subtasks:**

- `P09-T03-S01`: Add leave-one-entity-out splits.
- `P09-T03-S02`: Add leave-one-trajectory-out splits.
- `P09-T03-S03`: Score shared skeleton transfer.
- `P09-T03-S04`: Emit unseen-entity evidence.

**Acceptance:**

- Unseen-entity claims require holdout evidence.

### P09 Exit Gate

Run:

```bash
pytest -q tests/unit/panel
pytest -q tests/integration/test_shared_structure_law_pipeline.py
```

P09 is complete when shared-structure claims are backed by entity/trajectory validation.

## 26.11 P10: Stochastic Model Replacement

### Entry Gate

- P09 is complete.
- Claim lanes can distinguish stochastic laws.

### P10-T01: Replace Observation Model Registry

**Files:**

- Replace: `src/euclid/math/observation_models.py`
- Create: `src/euclid/stochastic/observation_models.py`
- Create: `tests/unit/stochastic/test_observation_models.py`

**Subtasks:**

- `P10-T01-S01`: Add Gaussian.
- `P10-T01-S02`: Add Student-t.
- `P10-T01-S03`: Add Laplace.
- `P10-T01-S04`: Add Poisson.
- `P10-T01-S05`: Add negative binomial.
- `P10-T01-S06`: Add Bernoulli.
- `P10-T01-S07`: Add beta.
- `P10-T01-S08`: Add lognormal.
- `P10-T01-S09`: Add mixture model interface.
- `P10-T01-S10`: Add bounded support checks.

**Acceptance:**

- Observation family is explicit and comparable.

### P10-T02: Replace Probabilistic Support Generation

**Files:**

- Replace or heavily modify: `src/euclid/modules/probabilistic_evaluation.py`
- Create: `src/euclid/stochastic/process_models.py`
- Create: `tests/unit/stochastic/test_process_models.py`

**Subtasks:**

- `P10-T02-S01`: Remove heuristic Gaussian scale inflation as production path.
- `P10-T02-S02`: Fit distribution parameters.
- `P10-T02-S03`: Fit scale/dispersion laws.
- `P10-T02-S04`: Support state-dependent scale.
- `P10-T02-S05`: Support horizon-dependent scale.
- `P10-T02-S06`: Emit stochastic model manifest.

**Acceptance:**

- Distribution predictions come from fitted stochastic models.

### P10-T03: Replace Event Probability Hardcoding

**Files:**

- Create: `schemas/contracts/event-definition.yaml`
- Modify: `src/euclid/modules/probabilistic_evaluation.py`
- Create: `tests/unit/stochastic/test_event_definitions.py`

**Subtasks:**

- `P10-T03-S01`: Define event manifest.
- `P10-T03-S02`: Add threshold source.
- `P10-T03-S03`: Add operator.
- `P10-T03-S04`: Add units.
- `P10-T03-S05`: Calculate event probability from fitted distribution.

**Acceptance:**

- Event probabilities are never hardcoded to origin target unless declared.

### P10-T04: Add Proper Scoring Rules

**Files:**

- Create: `src/euclid/stochastic/scoring_rules.py`
- Modify: `src/euclid/modules/scoring.py`
- Create: `tests/unit/stochastic/test_scoring_rules.py`

**Subtasks:**

- `P10-T04-S01`: Implement log score.
- `P10-T04-S02`: Implement CRPS where available or approximated.
- `P10-T04-S03`: Implement interval score.
- `P10-T04-S04`: Implement pinball loss.
- `P10-T04-S05`: Implement Brier score.

**Acceptance:**

- Probabilistic forecasts use proper scoring rules.

### P10 Exit Gate

Run:

```bash
pytest -q tests/unit/stochastic
pytest -q tests/integration/test_stochastic_law_pipeline.py
```

P10 is complete when stochastic claims are explicit, fitted, scored, calibrated, and replayable.

## 26.12 P11: Statistical Promotion Gates

### Entry Gate

- P10 is complete.
- Point and probabilistic scores are comparable under stronger models.

### P11-T01: Add Predictive Test Module

**Files:**

- Create: `src/euclid/modules/predictive_tests.py`
- Create: `tests/unit/modules/test_predictive_tests.py`

**Subtasks:**

- `P11-T01-S01`: Implement paired loss difference summary.
- `P11-T01-S02`: Implement bootstrap confidence intervals.
- `P11-T01-S03`: Implement block bootstrap.
- `P11-T01-S04`: Implement HAC-adjusted intervals using statsmodels.
- `P11-T01-S05`: Implement Diebold-Mariano-style test where applicable.

**Acceptance:**

- Predictive tests produce typed manifests and diagnostics.

### P11-T02: Add Prequential Score Streams

**Files:**

- Create: `schemas/contracts/prequential-score-stream.yaml`
- Modify: `src/euclid/modules/scoring.py`
- Create: `tests/unit/modules/test_prequential_scores.py`

**Subtasks:**

- `P11-T02-S01`: Emit per-origin losses.
- `P11-T02-S02`: Emit per-horizon losses.
- `P11-T02-S03`: Emit per-entity losses.
- `P11-T02-S04`: Emit per-regime losses.
- `P11-T02-S05`: Add rolling degradation diagnostics.

**Acceptance:**

- Aggregated scores can be audited against time-local failures.

### P11-T03: Replace Boolean Promotion

**Files:**

- Modify: `src/euclid/modules/evaluation_governance.py`
- Modify: `src/euclid/modules/gate_lifecycle.py`
- Create: `schemas/contracts/paired-predictive-test-result.yaml`
- Create: `tests/integration/test_statistical_promotion_gate.py`

**Subtasks:**

- `P11-T03-S01`: Remove pure `candidate_beats_baseline` sufficiency.
- `P11-T03-S02`: Require practical margin evidence.
- `P11-T03-S03`: Require uncertainty evidence.
- `P11-T03-S04`: Require many-model correction evidence when applicable.
- `P11-T03-S05`: Preserve blocking and downgrade behavior.

**Acceptance:**

- No predictive claim promotes from a tiny noisy score improvement alone.

### P11 Exit Gate

Run:

```bash
pytest -q tests/unit/modules/test_predictive_tests.py tests/unit/modules/test_prequential_scores.py
pytest -q tests/integration/test_statistical_promotion_gate.py
```

P11 is complete when predictive promotion is inferentially justified.

## 26.13 P12: Falsification And Residual Diagnostics

### Entry Gate

- P11 is complete.
- Promotion gates can consume additional falsification status.

### P12-T01: Add Residual Diagnostics

**Files:**

- Create: `src/euclid/falsification/residuals.py`
- Create: `tests/unit/falsification/test_residuals.py`

**Subtasks:**

- `P12-T01-S01`: Autocorrelation diagnostics.
- `P12-T01-S02`: Heteroskedasticity diagnostics.
- `P12-T01-S03`: Heavy-tail diagnostics.
- `P12-T01-S04`: Regime residual drift.
- `P12-T01-S05`: Entity residual diagnostics.

**Acceptance:**

- Residual failures can lower claim ceilings.

### P12-T02: Add Surrogate Tests

**Files:**

- Create: `src/euclid/falsification/surrogate_tests.py`
- Create: `tests/unit/falsification/test_surrogate_tests.py`

**Subtasks:**

- `P12-T02-S01`: IID permutation.
- `P12-T02-S02`: Block permutation.
- `P12-T02-S03`: Phase randomization.
- `P12-T02-S04`: Seasonal-preserving surrogates.
- `P12-T02-S05`: Null p-value summaries.

**Acceptance:**

- Structure claims compare against appropriate nulls.

### P12-T03: Add Perturbation And Counterexample Search

**Files:**

- Create: `src/euclid/falsification/perturbations.py`
- Create: `src/euclid/falsification/counterexamples.py`
- Create: `tests/unit/falsification/test_counterexamples.py`

**Subtasks:**

- `P12-T03-S01`: Missingness perturbation.
- `P12-T03-S02`: Noise perturbation.
- `P12-T03-S03`: Window truncation.
- `P12-T03-S04`: Environment swap.
- `P12-T03-S05`: Counterexample horizon search.

**Acceptance:**

- Strong claims survive declared falsification attempts.

### P12-T04: Emit Falsification Dossier

**Files:**

- Create: `schemas/contracts/falsification-dossier.yaml`
- Modify: `src/euclid/modules/robustness.py`
- Modify: `src/euclid/modules/gate_lifecycle.py`
- Create: `tests/integration/test_falsification_dossier.py`

**Subtasks:**

- `P12-T04-S01`: Aggregate residual diagnostics.
- `P12-T04-S02`: Aggregate surrogate results.
- `P12-T04-S03`: Aggregate perturbation results.
- `P12-T04-S04`: Decide claim ceiling impact.
- `P12-T04-S05`: Include dossier in replay/publication.

**Acceptance:**

- Universal and stochastic claims require falsification dossiers.

### P12 Exit Gate

Run:

```bash
pytest -q tests/unit/falsification
pytest -q tests/integration/test_falsification_dossier.py
```

P12 is complete when strong claims include falsification evidence.

## 26.14 P13: Benchmark Universe Replacement

### Entry Gate

- P12 is complete.
- Claim gates have final semantics.

### P13-T01: Redefine Benchmark Manifest Semantics

**Files:**

- Modify: `src/euclid/benchmarks/manifests.py`
- Modify: `schemas/readiness/full-vision-matrix.yaml`
- Modify: `tests/unit/benchmarks/test_manifests.py`

**Subtasks:**

- `P13-T01-S01`: Add metric thresholds.
- `P13-T01-S02`: Add expected claim ceilings.
- `P13-T01-S03`: Add false-claim expectations.
- `P13-T01-S04`: Add engine requirements.
- `P13-T01-S05`: Add semantic readiness row IDs.

**Acceptance:**

- Benchmark manifests define pass/fail semantics.

### P13-T02: Add Discovery Benchmark Families

**Files:**

- Create/modify: `benchmarks/tasks/**`
- Create/modify: `src/euclid/_assets/benchmarks/tasks/**`
- Create: `tests/benchmarks/test_law_discovery_suites.py`

**Subtasks:**

- `P13-T02-S01`: Classic symbolic regression tasks.
- `P13-T02-S02`: Sparse ODE tasks.
- `P13-T02-S03`: Multi-trajectory tasks.
- `P13-T02-S04`: Multi-entity shared-law tasks.
- `P13-T02-S05`: Hidden-state tasks.
- `P13-T02-S06`: Delay-system tasks.
- `P13-T02-S07`: Stochastic tasks.
- `P13-T02-S08`: Adversarial false-law tasks.

**Acceptance:**

- Benchmark coverage maps to all enhanced claim lanes.

### P13-T03: Add Semantic Result Evaluation

**Files:**

- Modify: `src/euclid/benchmarks/runtime.py`
- Modify: `src/euclid/benchmarks/reporting.py`
- Modify: `src/euclid/readiness/judgment.py`

**Subtasks:**

- `P13-T03-S01`: Evaluate exact symbolic recovery.
- `P13-T03-S02`: Evaluate equivalent recovery.
- `P13-T03-S03`: Evaluate parameter error.
- `P13-T03-S04`: Evaluate rollout error.
- `P13-T03-S05`: Evaluate invariance score.
- `P13-T03-S06`: Evaluate false-publication rate.

**Acceptance:**

- Readiness uses metrics, not artifact existence.

### P13 Exit Gate

Run:

```bash
pytest -q tests/benchmarks
python -m euclid benchmarks run --suite full-vision.yaml --no-resume
python -m euclid release certify-research-readiness
```

P13 is complete when benchmarks prove enhanced law-discovery semantics.

## 26.15 P14: Performance And Scaling

### Entry Gate

- P13 is complete.
- Benchmark workloads expose realistic runtime pressure.

### P14-T01: Add Evaluation Cache

**Files:**

- Create: `src/euclid/runtime/cache.py`
- Create: `tests/unit/runtime/test_cache.py`

**Subtasks:**

- `P14-T01-S01`: Cache feature matrices.
- `P14-T01-S02`: Cache expression evaluations.
- `P14-T01-S03`: Cache subtree evaluations.
- `P14-T01-S04`: Cache fitted constants.
- `P14-T01-S05`: Cache simplification results.
- `P14-T01-S06`: Include cache keys in replay diagnostics.

**Acceptance:**

- Cache hits do not change results.

### P14-T02: Add Parallel Runtime

**Files:**

- Create: `src/euclid/runtime/parallel.py`
- Create: `tests/unit/runtime/test_parallel.py`

**Subtasks:**

- `P14-T02-S01`: Add joblib local parallelism.
- `P14-T02-S02`: Add deterministic aggregation.
- `P14-T02-S03`: Add worker failure diagnostics.
- `P14-T02-S04`: Add Ray adapter if enabled.

**Acceptance:**

- Results are stable across worker counts.

### P14-T03: Add Performance Gates

**Files:**

- Modify: `src/euclid/performance.py`
- Create: `tests/perf/test_candidate_throughput.py`
- Create: `tests/perf/test_engine_runtime_budgets.py`

**Subtasks:**

- `P14-T03-S01`: Candidate throughput budget.
- `P14-T03-S02`: Optimizer iteration budget.
- `P14-T03-S03`: PySINDy runtime budget.
- `P14-T03-S04`: PySR runtime budget.
- `P14-T03-S05`: E-graph saturation budget.

**Acceptance:**

- Expanded discovery is bounded and measurable.

### P14 Exit Gate

Run:

```bash
pytest -q tests/unit/runtime/test_cache.py tests/unit/runtime/test_parallel.py
pytest -q tests/perf
```

P14 is complete when performance is deterministic, cached, parallel, and budgeted.

## 26.16 P15: Workbench Evidence Studio

### Entry Gate

- P14 is complete.
- Backend evidence surfaces are stable.

### P15-T01: Redesign Workbench Data Service

**Files:**

- Modify or replace: `src/euclid/workbench/service.py`
- Modify: `tests/unit/workbench/test_service.py`

**Subtasks:**

- `P15-T01-S01`: Expose candidate lineage.
- `P15-T01-S02`: Expose engine provenance.
- `P15-T01-S03`: Expose optimizer diagnostics.
- `P15-T01-S04`: Expose invariance evidence.
- `P15-T01-S05`: Expose stochastic diagnostics.
- `P15-T01-S06`: Expose falsification dossier.
- `P15-T01-S07`: Expose claim ceiling explanation.

**Acceptance:**

- Workbench API is evidence-oriented, not formula-only.

### P15-T02: Replace Workbench UI Panels

**Files:**

- Replace or heavily modify: `src/euclid/_assets/workbench/app.js`
- Replace or heavily modify: `src/euclid/_assets/workbench/app.css`
- Modify: `tests/frontend/workbench/app.test.js`

**Subtasks:**

- `P15-T02-S01`: Add Pareto frontier panel.
- `P15-T02-S02`: Add lineage panel.
- `P15-T02-S03`: Add residual diagnostics panel.
- `P15-T02-S04`: Add calibration panel.
- `P15-T02-S05`: Add invariance matrix.
- `P15-T02-S06`: Add transfer matrix.
- `P15-T02-S07`: Add falsification panel.
- `P15-T02-S08`: Add replay browser.

**Acceptance:**

- UI makes claim evidence and limitations visible.

### P15-T03: Remove Overclaiming UI Copy

**Subtasks:**

- `P15-T03-S01`: Audit every use of "law" in UI.
- `P15-T03-S02`: Replace unsupported "law" language with claim-lane-specific labels.
- `P15-T03-S03`: Add tests for claim terminology.

**Acceptance:**

- Workbench cannot imply universal support without a universal claim artifact.

### P15 Exit Gate

Run:

```bash
pytest -q tests/unit/workbench
npm run test:frontend
python scripts/workbench_ui_smoke.py
```

P15 is complete when the workbench is an evidence studio.

## 26.17 P16: Documentation Replacement

### Entry Gate

- P15 is complete.
- Enhanced behavior is stable.

### P16-T01: Replace System Overview Docs

**Files:**

- Replace or heavily modify: `README.md`
- Replace or heavily modify: `docs/system.md`
- Replace or heavily modify: `docs/modeling-pipeline.md`

**Subtasks:**

- `P16-T01-S01`: Document enhanced architecture.
- `P16-T01-S02`: Document replacement of retained-slice search.
- `P16-T01-S03`: Document claim lanes.
- `P16-T01-S04`: Document evidence flow.

**Acceptance:**

- Docs no longer describe legacy retained slice as final system.

### P16-T02: Replace Search And Engine Docs

**Files:**

- Replace or heavily modify: `docs/search-core.md`

**Subtasks:**

- `P16-T02-S01`: Document expression IR.
- `P16-T02-S02`: Document engine contract.
- `P16-T02-S03`: Document PySINDy.
- `P16-T02-S04`: Document PySR.
- `P16-T02-S05`: Document rewrite/e-graph behavior.
- `P16-T02-S06`: Document external library provenance.

**Acceptance:**

- Search docs map to real engines.

### P16-T03: Replace Benchmark And Readiness Docs

**Files:**

- Replace or heavily modify: `docs/benchmarks-readiness.md`
- Modify: `docs/testing-truthfulness.md`

**Subtasks:**

- `P16-T03-S01`: Document benchmark semantics.
- `P16-T03-S02`: Document false-claim benchmarks.
- `P16-T03-S03`: Document readiness thresholds.
- `P16-T03-S04`: Document semantic evidence requirements.

**Acceptance:**

- Readiness docs explain what is scientifically proven.

### P16-T04: Replace Workbench Docs

**Files:**

- Replace or heavily modify: `docs/workbench.md`

**Subtasks:**

- `P16-T04-S01`: Document evidence panels.
- `P16-T04-S02`: Document claim ceilings.
- `P16-T04-S03`: Document replay browser.
- `P16-T04-S04`: Document that UI does not create claims.

**Acceptance:**

- Workbench docs match evidence studio behavior.

### P16 Exit Gate

Run:

```bash
pytest -q tests/spec_compiler || true
pytest -q tests/unit tests/integration tests/benchmarks
python -m euclid release status
python -m euclid release verify-completion
python -m euclid release certify-research-readiness
```

P16 is complete when documentation, code, schemas, tests, benchmarks, and readiness tell the same story.

## 27. Final 100% Completion Checklist

Use this checklist at the end of the enhancement program.

- [ ] P00 evidence spine reset complete.
- [ ] P01 scientific dependencies and numerical replay complete.
- [ ] P02 expression IR replacement complete.
- [ ] P03 unified fitting layer complete.
- [ ] P04 engine orchestration complete.
- [ ] P05 real PySINDy backend complete.
- [ ] P06 real PySR backend complete.
- [ ] P07 rewrite/equality saturation complete.
- [ ] P08 universal and transport claim lanes complete.
- [ ] P09 shared-structure panel discovery complete.
- [ ] P10 stochastic model replacement complete.
- [ ] P11 statistical promotion gates complete.
- [ ] P12 falsification dossier complete.
- [ ] P13 benchmark universe replacement complete.
- [ ] P14 performance and scaling complete.
- [ ] P15 workbench evidence studio complete.
- [ ] P16 documentation replacement complete.
- [ ] Old SINDy shim removed.
- [ ] Old decomposition shim removed.
- [ ] Sort-only equality saturation removed.
- [ ] Gaussian heuristic support path removed as production path.
- [ ] Boolean-only predictive promotion removed.
- [ ] Panel mean-offset law evidence removed.
- [ ] Ambiguous "predictive law" language removed where unsupported.
- [ ] Mandatory dual-gate harness exists and is used by every phase.
- [ ] `.env.example`, live API test policy, and live test README exist without secrets.
- [ ] Every phase, task, and subtask has fixture unit, fixture integration, fixture regression, live API, and edge-case gates.
- [ ] All fixture unit tests pass.
- [ ] All fixture integration tests pass.
- [ ] All fixture regression/golden tests pass.
- [ ] All benchmark fixture tests pass.
- [ ] Strict live API smoke passes with keys loaded from `.env`.
- [ ] Live FMP ingestion path passes semantic checks and sanitized evidence checks.
- [ ] Live OpenAI workbench explanation path passes schema, timeout, redaction, and typed-abstention checks.
- [ ] Secret redaction tests prove API keys cannot leak through logs, artifacts, exceptions, URLs, headers, workbench payloads, or release bundles.
- [ ] Every strong claim has explicit gate evidence.
- [ ] Every external library result has replay-visible version and settings.
- [ ] Every benchmark pass is semantic.
- [ ] Workbench displays evidence without creating claims.

When all items are checked, Euclid has reached the target enhanced architecture defined by this plan.

## 28. Mandatory Phase-Level Dual Gates

This section is an overlay on every phase in Section 26. It is not optional and it is not advisory. If any phase-specific gate below conflicts with an earlier lighter test command, use the stricter command and update the earlier section.

Every phase must create a machine-readable gate manifest:

```text
tests/gates/PXX.yaml
```

Each manifest must list:

- Phase ID.
- Task IDs covered.
- Subtask IDs covered.
- Fixture unit commands.
- Fixture integration commands.
- Fixture regression/golden commands.
- Live API commands.
- Required `.env` variables.
- Edge-case classes covered.
- Artifact outputs.
- Redaction assertions.
- Replay assertions.
- Claim-scope assertions.
- Known external-provider volatility assumptions.

The spec compiler or release verifier must reject a phase as incomplete when its `tests/gates/PXX.yaml` is missing any task or subtask ID from this plan.

### 28.1 P00 Evidence Spine Reset Gates

**Fixture Unit Gate:**

- Asset resolver tests.
- Release semantic field tests.
- Replay semantic field tests.
- Benchmark semantic summary tests.
- Env loading tests.
- Redaction tests.
- Live harness fixture tests.

**Fixture Integration Gate:**

- End-to-end CLI smoke using offline fixtures.
- Release completion report generation using offline fixtures.
- Replay verification using fixture artifacts.
- Fixture-mode FMP and OpenAI HTTP simulations.

**Fixture Regression Gate:**

- Golden release report proving semantic evidence is required.
- Golden sanitized live evidence proving no secret leakage.
- Golden readiness report proving file-existence-only evidence fails.

**Live API Gate:**

Run:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 ./scripts/live_api_smoke.sh
```

Live checks:

- `FMP_API_KEY` loads from `.env`.
- Live FMP response creates a sanitized ingestion evidence artifact.
- `OPENAI_API_KEY` loads from `.env`.
- Live OpenAI explanation path creates a sanitized workbench explanation or typed abstention artifact.
- No live key appears in stdout, stderr, pytest reports, release bundles, or workbench payloads.

### 28.2 P01 Dependency And Numerical Runtime Gates

**Fixture Unit Gate:**

- Import tests for NumPy, pandas, SciPy, SymPy, Pint, statsmodels, scikit-learn, PySINDy, PySR, egglog, joblib, Ray if enabled, SQLAlchemy, Pydantic, PyYAML, Typer, PyArrow, httpx, python-dotenv, vcrpy, responses/respx, pytest-timeout, pytest-xdist, and Hypothesis.
- Numerical environment capture tests.
- Numerical policy manifest tests.
- Dependency failure message tests for missing optional engines.

**Fixture Integration Gate:**

- Clean install smoke in a temporary environment.
- Replay bundle containing numerical versions.
- PySR unavailable fixture proving typed downgrade when Julia is absent.
- PySINDy unavailable fixture proving typed downgrade when import fails.

**Fixture Regression Gate:**

- Golden numerical environment artifact.
- Golden dependency policy manifest.
- Golden failure diagnostics for missing external runtime.

**Live API Gate:**

Run:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 pytest -q tests/live/test_dependency_runtime_live.py
```

Live checks:

- Live FMP ingestion artifact includes numerical environment metadata.
- Live OpenAI workbench explanation artifact includes numerical/runtime metadata where applicable.
- Live provider failures preserve dependency diagnostics and redaction.

### 28.3 P02 Typed Expression And CIR Replacement Gates

**Fixture Unit Gate:**

- Typed AST construction.
- Operator registry domain constraints.
- Unit propagation through Pint.
- SymPy simplification round trips.
- CIR serialization/deserialization.
- Structural hash invariance.
- Hypothesis-generated expression trees.

**Fixture Integration Gate:**

- Offline fixture series lowered through AST, CIR, fitting, scoring, replay, and claim gate.
- Fixture with illegal domains: log of nonpositive values, division by zero, sqrt of negative values, invalid units, incompatible binary operations.
- Fixture proving canonical equivalence for syntactically different expressions.

**Fixture Regression Gate:**

- Golden CIR for analytic, recursive, spectral, algorithmic, PySINDy, PySR, and rewrite-derived candidates.
- Golden structural hashes for canonical expressions.
- Golden rejection artifact for illegal units/domains.

**Live API Gate:**

Run:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 pytest -q tests/live/test_expression_cir_live_pipeline.py
```

Live checks:

- Live FMP observations produce typed feature variables.
- Live candidate expressions lower into CIR without leaking provider metadata into structural identity.
- Live illegal-domain candidates are rejected with typed errors.
- Live replay can recover the same CIR hash from sanitized evidence.

### 28.4 P03 Unified Fitting And Constant Optimization Gates

**Fixture Unit Gate:**

- SciPy least-squares fitting.
- Robust fitting.
- Bound-constrained fitting.
- Multi-start initialization.
- Parameter covariance and diagnostics.
- Failure classification.
- Hypothesis tests for finite residual behavior.

**Fixture Integration Gate:**

- Offline synthetic linear, nonlinear, rational, logistic, exponential, sinusoidal, and stiff fixtures.
- Fixture for singular design matrix.
- Fixture for underdetermined constants.
- Fixture for optimizer timeout.
- Fixture for invalid domain during optimization.

**Fixture Regression Gate:**

- Golden fit summaries.
- Golden optimizer traces.
- Golden parameter code-length accounting.
- Golden failed-convergence artifacts.

**Live API Gate:**

Run:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 pytest -q tests/live/test_constant_fitting_live.py
```

Live checks:

- Live FMP bounded lag/return fitting completes under timeout.
- Nonlinear constant fitting emits convergence or typed downgrade.
- Live residuals are finite after fitted-domain guards.
- Live fit evidence includes optimizer method, seed, bounds, tolerance, iteration count, and sanitized data-window metadata.

### 28.5 P04 Engine Orchestration And Search Portfolio Gates

**Fixture Unit Gate:**

- Engine registry tests.
- Engine capability declarations.
- Timeout and budget policy tests.
- Deduplication and CIR lowering tests.
- Search honesty tests for exact, bounded, stochastic, and external-engine modes.

**Fixture Integration Gate:**

- Offline portfolio run combining native, PySINDy, PySR, and rewrite candidates.
- Fixture with one crashing engine.
- Fixture with one slow engine.
- Fixture with duplicate candidates across engines.
- Fixture with no valid candidates.

**Fixture Regression Gate:**

- Golden engine portfolio evidence.
- Golden search omission statement.
- Golden candidate provenance and structural hash archive.

**Live API Gate:**

Run:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 pytest -q tests/live/test_search_portfolio_live.py
```

Live checks:

- Live FMP series flows through the orchestrator.
- External engine failures do not block other engines.
- Candidate deduplication works on live candidates.
- Search completeness and omission statements are present in live evidence.

### 28.6 P05 Real PySINDy Backend Gates

**Fixture Unit Gate:**

- PySINDy library construction.
- Derivative estimation wrappers.
- STLSQ/SR3 optimizer configuration.
- Weak-form configuration.
- SINDy-PI/implicit law lowering where supported.
- Coefficient threshold and sparsity accounting.

**Fixture Integration Gate:**

- Clean logistic growth recovery.
- Noisy logistic growth recovery.
- Lorenz or low-dimensional chaotic fixture with known support.
- Delay-system fixture where direct observed scalar must abstain or use delay embedding.
- Count/irregular-sampling fixture proving typed rejection when unsupported.

**Fixture Regression Gate:**

- Golden PySINDy candidate CIR.
- Golden support recovery metrics.
- Golden sparse solver trace.
- Golden abstention artifact for insufficient derivative quality.

**Live API Gate:**

Run:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 pytest -q tests/live/test_pysindy_live_backend.py
```

Live checks:

- Live FMP return/level transformations can be submitted to PySINDy under a bounded budget.
- Live derivative estimation diagnostics are emitted.
- Live PySINDy candidates lower into CIR or fail with typed reason codes.
- Live PySINDy results cannot publish universal claims without P08/P09 invariance evidence.

### 28.7 P06 Real PySR Backend Gates

**Fixture Unit Gate:**

- PySR configuration translation.
- Operator constraint translation.
- Constant optimization capture.
- Julia runtime diagnostics.
- Equation table parsing.
- Complexity/Pareto archive lowering.

**Fixture Integration Gate:**

- Nguyen-style symbolic recovery fixture.
- Feynman-style unit-aware fixture.
- Noisy fixture requiring robust selection.
- Domain-guard fixture for division/log/sqrt.
- Julia unavailable fixture proving typed downgrade.

**Fixture Regression Gate:**

- Golden PySR equation table.
- Golden PySR-to-CIR lowering.
- Golden operator constraint evidence.
- Golden timeout/partial-result evidence.

**Live API Gate:**

Run:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 pytest -q tests/live/test_pysr_live_backend.py
```

Live checks:

- Live FMP lagged feature table can run through a tiny PySR budget.
- PySR live output lowers to typed CIR.
- Live evidence records Julia/PySR versions, operators, constraints, random seed, budget, and timeout.
- PySR live results are scored and gated by Euclid, not published directly.

### 28.8 P07 Rewrite And Equality Saturation Gates

**Fixture Unit Gate:**

- Rewrite rule registration.
- Egglog/e-graph extraction.
- Cost model tests.
- Unit/domain-preserving rewrite tests.
- Non-terminating rewrite guard tests.

**Fixture Integration Gate:**

- Offline candidate archive with redundant algebraic forms.
- Fixture where simplification changes cost but not semantics.
- Fixture where rewrite would violate domain if applied globally.
- Fixture with rational/trigonometric identities under domain guards.

**Fixture Regression Gate:**

- Golden e-class archive.
- Golden extracted best representative.
- Golden proof trace or rewrite provenance.
- Golden rejection for unsafe rewrite.

**Live API Gate:**

Run:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 pytest -q tests/live/test_rewrite_live_pipeline.py
```

Live checks:

- Live-generated candidate duplicates collapse to canonical forms.
- Rewrite output preserves live fitted predictions within tolerance.
- Unsafe rewrites are rejected on live domain evidence.
- Live evidence records rewrite rules, e-graph budget, extraction cost, and final CIR hash.

### 28.9 P08 Universal And Transport Claim Gates

**Fixture Unit Gate:**

- Invariance score tests.
- Environment split tests.
- Claim ceiling tests.
- Transport abstention tests.
- Parameter-stability tests.
- Residual-invariance tests.

**Fixture Integration Gate:**

- Multi-environment synthetic same-law fixture.
- Multi-environment spurious-correlation fixture.
- Structural break fixture.
- Hidden intervention fixture.
- Single-series pseudo-environment fixture.

**Fixture Regression Gate:**

- Golden universal-claim publication artifact.
- Golden abstention artifact for insufficient environments.
- Golden downgrade artifact for unstable parameters.
- Golden failed-transport artifact.

**Live API Gate:**

Run:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 pytest -q tests/live/test_universal_claim_live_gates.py
```

Live checks:

- Live FMP symbols form environments without leakage.
- Live rolling-era pseudo-environments are time-safe.
- Live claim path abstains when invariance is not established.
- Live evidence includes parameter dispersion, residual diagnostics, environment labels, and claim ceiling.

### 28.10 P09 Shared-Structure Panel Discovery Gates

**Fixture Unit Gate:**

- Shared skeleton representation.
- Entity-specific parameter fitting.
- Shrinkage penalty tests.
- Group support stability tests.
- Panel missingness tests.

**Fixture Integration Gate:**

- Same law, different constants fixture.
- Different laws, similar fit fixture.
- Missing entity fixture.
- Unequal history length fixture.
- High-dispersion parameter fixture.

**Fixture Regression Gate:**

- Golden shared-structure model artifact.
- Golden local-only rejection artifact.
- Golden parameter dispersion table.
- Golden support stability report.

**Live API Gate:**

Run:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 pytest -q tests/live/test_shared_structure_panel_live.py
```

Live checks:

- Live FMP multi-symbol panel normalizes into aligned and unaligned panel views.
- Shared-structure search runs with bounded budget.
- Missing rows and market holidays do not leak future data.
- Live panel result either passes shared-structure evidence or emits a typed abstention.

### 28.11 P10 Stochastic Observation And Process Model Gates

**Fixture Unit Gate:**

- Gaussian, Student-t, Laplace, Poisson, negative binomial, beta/bounded, heteroskedastic, mixture, and state-dependent scale likelihood tests.
- Distribution support tests.
- Quantile and interval tests.
- CRPS/log-score tests.
- Calibration tests.

**Fixture Integration Gate:**

- Heavy-tail fixture.
- Count overdispersion fixture.
- Bounded-rate fixture.
- Volatility-clustering fixture.
- Mixture-regime fixture.
- Invalid distribution-support fixture.

**Fixture Regression Gate:**

- Golden stochastic fit artifact.
- Golden calibration report.
- Golden probabilistic forecast bundle.
- Golden downgrade from unsupported Gaussian heuristic.

**Live API Gate:**

Run:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 pytest -q tests/live/test_stochastic_models_live.py
```

Live checks:

- Live FMP residuals evaluate under multiple candidate observation models.
- Heavy-tail or heteroskedastic models can beat Gaussian only through comparable likelihood and complexity accounting.
- Unsupported support violations fail closed.
- Live probabilistic evidence includes calibration diagnostics and does not infer stochastic law from point fit alone.

### 28.12 P11 Statistical Promotion Gates

**Fixture Unit Gate:**

- Rolling-origin split tests.
- Prequential score tests.
- Multiple-testing control tests.
- Bootstrap/permutation tests.
- Predictive promotion threshold tests.
- Leakage rejection tests.

**Fixture Integration Gate:**

- Overfit symbolic expression fixture.
- Random walk fixture.
- True extrapolating law fixture.
- Regime-shift fixture.
- Small-sample fixture.

**Fixture Regression Gate:**

- Golden promotion artifact.
- Golden abstention artifact.
- Golden leakage failure artifact.
- Golden multiple-testing correction report.

**Live API Gate:**

Run:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 pytest -q tests/live/test_statistical_promotion_live.py
```

Live checks:

- Live FMP rolling-origin splits are frozen before fitting.
- Live candidate promotion requires out-of-sample improvement, not training compression alone.
- Live data with insufficient horizon abstains.
- Live evidence records split geometry, baseline comparison, score deltas, uncertainty, and promotion/downgrade reason.

### 28.13 P12 Falsification Dossier Gates

**Fixture Unit Gate:**

- Counterexample generation.
- Stress perturbation generation.
- Domain violation tests.
- Residual diagnostic tests.
- Dossier schema tests.

**Fixture Integration Gate:**

- Candidate that fails under extrapolation.
- Candidate that fails under perturbation.
- Candidate that fails under alternate observation model.
- Candidate that survives declared falsification set.

**Fixture Regression Gate:**

- Golden falsification dossier.
- Golden counterexample archive.
- Golden survivor report.
- Golden public-claim downgrade report.

**Live API Gate:**

Run:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 pytest -q tests/live/test_falsification_live.py
```

Live checks:

- Live FMP candidate undergoes perturbation, rolling-era, alternate baseline, and missingness stress tests.
- Live falsification can downgrade or block claim publication.
- Live dossier records counterexamples without storing secrets.
- Live workbench can display falsification results without strengthening claim language.

### 28.14 P13 Benchmark Universe Gates

**Fixture Unit Gate:**

- Benchmark registry tests.
- Dataset manifest tests.
- Metric computation tests.
- Baseline adapter tests.
- Semantic threshold tests.

**Fixture Integration Gate:**

- Feynman/Nguyen fixture.
- SRBench-style fixture.
- SINDy dynamics fixture.
- Multi-environment law fixture.
- Hidden-state/delay fixture.
- Adversarial honesty fixture.

**Fixture Regression Gate:**

- Golden benchmark leaderboard.
- Golden benchmark card.
- Golden baseline comparison report.
- Golden adversarial canary report.

**Live API Gate:**

Run:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 pytest -q tests/live/test_benchmark_live_smoke.py
```

Live checks:

- Live FMP real-series smoke can be registered as a non-ground-truth operational benchmark.
- Live benchmark metadata distinguishes live operational validation from scientific ground-truth recovery.
- Live benchmark artifacts are sanitized and replay-aware.
- Live benchmark pass requires semantic checks, not just successful execution.

### 28.15 P14 Performance And Scaling Gates

**Fixture Unit Gate:**

- Cache key tests.
- Parallel execution tests.
- Determinism under parallelism tests.
- Budget accounting tests.
- Timeout tests.

**Fixture Integration Gate:**

- Offline large candidate archive.
- Offline repeated-subexpression cache fixture.
- Offline multi-engine parallel fixture.
- Offline memory-bound fixture.

**Fixture Regression Gate:**

- Golden runtime profile.
- Golden cache hit-rate report.
- Golden parallel determinism artifact.
- Golden timeout failure artifact.

**Live API Gate:**

Run:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 pytest -q tests/live/test_performance_live_smoke.py
```

Live checks:

- Live FMP bounded run completes within declared wall-clock budget.
- Live run records cache statistics, candidate counts, engine timings, and timeout reason codes.
- Live API calls are rate-limited and serialized where provider terms require it.
- Live performance evidence never treats provider latency as model runtime.

### 28.16 P15 Workbench Evidence Studio Gates

**Fixture Unit Gate:**

- Workbench API contract tests.
- Artifact rendering tests.
- Secret redaction tests.
- Claim language tests.
- Explanation schema tests.

**Fixture Integration Gate:**

- Offline workbench run with fixture artifacts.
- Offline malformed artifact.
- Offline missing key state.
- Offline invalid key state.
- Offline OpenAI timeout/refusal state.

**Fixture Regression Gate:**

- Golden workbench JSON payload.
- Golden UI contract artifact.
- Golden explanation artifact.
- Golden claim-language downgrade artifact.

**Live API Gate:**

Run:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 pytest -q tests/live/test_workbench_live_evidence_studio.py
```

Live checks:

- Live FMP key can be used through the workbench path without exposing the key.
- Live OpenAI key can be used for explanation generation without deterministic prose assertions.
- Workbench displays claim ceilings, live evidence status, falsification status, and abstention reasons.
- Workbench never converts explanation text into scientific evidence.

### 28.17 P16 Documentation And Migration Gates

**Fixture Unit Gate:**

- Documentation link tests.
- CLI help snapshot tests.
- Schema reference tests.
- Plan/spec compiler tests.

**Fixture Integration Gate:**

- Offline tutorial commands execute with fixtures.
- Offline benchmark walkthrough executes with fixtures.
- Offline workbench walkthrough executes with fixture artifacts.

**Fixture Regression Gate:**

- Golden README snippets.
- Golden docs reference table.
- Golden migration/removal checklist.
- Golden command transcript with fixture mode.

**Live API Gate:**

Run:

```bash
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 pytest -q tests/live/test_documented_live_commands.py
```

Live checks:

- Documented FMP live smoke command works with `.env`.
- Documented OpenAI workbench explanation command works with `.env`.
- Docs describe live-provider volatility and do not assert deterministic live values.
- Docs include explicit warning that `.env` is never committed.

## 29. Task And Subtask Gate Manifest Requirements

Every `PXX-TYY` task and every `PXX-TYY-SZZ` subtask must be represented in the phase gate manifest. A task is incomplete when any of these fields is missing:

```yaml
id: PXX-TYY-SZZ
owner_module: src/euclid/...
fixture_unit:
  command: pytest -q ...
  assertions:
    - ...
fixture_integration:
  command: pytest -q ...
  assertions:
    - ...
fixture_regression:
  command: pytest -q ...
  golden_artifacts:
    - ...
live_api:
  command: EUCLID_LIVE_API_TESTS=1 ...
  env:
    required:
      - FMP_API_KEY
  semantic_assertions:
    - ...
edge_cases:
  required:
    - ...
redaction:
  assertions:
    - no_secret_in_stdout
    - no_secret_in_artifact
replay:
  assertions:
    - replay_metadata_complete
claim_scope:
  assertions:
    - no_unjustified_claim_promotion
```

Rules:

- The manifest must be reviewed alongside code.
- The manifest must be included in release evidence.
- A subtask cannot inherit an empty parent gate; it must name the exact fixture and live checks that prove its behavior.
- When a live API cannot deterministically prove a numeric value, the live gate must assert schema, freshness, ordering, redaction, and semantic downgrade behavior instead.
- When a subtask replaces old functionality, its regression gate must prove the old behavior is removed, disabled, or demoted.
- When a subtask introduces external-library behavior, its live and fixture gates must capture library version, configuration, and failure mode.

## 30. Required Live API Providers And Usage Boundaries

Euclid's enhanced plan requires live checks but must keep provider responsibility clear.

### 30.1 Financial Modeling Prep

Use `FMP_API_KEY` for:

- Live ordered-observation ingestion.
- Live workbench data retrieval.
- Live multi-symbol panel smoke tests.
- Live rolling-origin split validation.
- Live stochastic residual diagnostics.
- Live performance smoke under small bounded windows.

Do not use FMP for:

- Ground-truth scientific law recovery.
- Exact numeric regression assertions on latest market values.
- Universal-law publication by itself.

Required FMP live edge cases:

- Invalid key.
- Rate limit.
- Empty symbol.
- Delisted or unavailable symbol.
- Out-of-order rows.
- Duplicate dates.
- Missing adjusted close.
- Market holiday gaps.
- Stale data.
- Provider schema drift.

### 30.2 OpenAI

Use `OPENAI_API_KEY` for:

- Live workbench explanation generation.
- Live explanation schema and timeout checks.
- Live redaction and key-handling checks.

Do not use OpenAI for:

- Determining whether a law is true.
- Replacing Euclid's score, falsification, or claim gates.
- Exact regression assertions on generated prose.

Required OpenAI live edge cases:

- Missing key.
- Invalid key.
- Timeout.
- Rate limit.
- Empty response.
- Refusal or unavailable response.
- Malformed response shape.
- Model override through `EUCLID_OPENAI_EXPLAINER_MODEL`.
- Prompt and key redaction in logs/artifacts.

### 30.3 Additional Providers

Additional live providers may be added later only if:

- They have an explicit `.env.example` variable.
- They have fixture-mode tests for success and failures.
- They have sanitized evidence schemas.
- Their licensing permits the planned fixture and artifact behavior.
- They cannot promote scientific claims without Euclid-owned gates.

## 31. Robust Test Coverage By Enhancement Area

This section defines the minimum edge-case inventory implementers must draw from when writing the task-level gates.

### 31.1 Ingestion And State Reconstruction

Required fixture cases:

- Empty CSV/API payload.
- One row.
- Two rows.
- Non-monotonic timestamps.
- Duplicate timestamps with same value.
- Duplicate timestamps with conflicting values.
- Missing timestamp.
- Missing target value.
- Nonfinite target value.
- Timezone-naive input.
- Timezone-aware input.
- DST transition.
- Leap day.
- Irregular cadence.
- Revision/backfill row.

Required live cases:

- Live FMP symbol with enough rows.
- Live FMP symbol with sparse or unavailable rows.
- Live FMP provider timeout simulated in fixture mode and timeout-handled in live mode.

### 31.2 Expression, Units, And Domains

Required fixture cases:

- Unit-compatible addition.
- Unit-incompatible addition.
- Unit-compatible multiplication.
- Dimensionless transcendental input.
- Non-dimensionless transcendental input rejection.
- Division by zero.
- Near-zero denominator.
- Log domain violation.
- Sqrt domain violation.
- Power with invalid base/exponent combination.
- Piecewise branch with uncovered domain.

Required live cases:

- Live FMP price/return features have explicit units or dimensionless status.
- Live candidate with invalid domain is rejected before claim evaluation.

### 31.3 Search Engines

Required fixture cases:

- Native engine no candidates.
- Native engine duplicate candidates.
- PySINDy exact support recovery.
- PySINDy noisy support recovery.
- PySINDy derivative failure.
- PySR exact symbolic recovery.
- PySR timeout with partial result.
- PySR Julia unavailable.
- E-graph rewrite explosion budget.
- E-graph unsafe rewrite.

Required live cases:

- Live FMP orchestrator run with each enabled engine.
- Live engine failure does not corrupt portfolio evidence.
- Live duplicate candidates collapse by CIR hash.

### 31.4 Fitting And Scoring

Required fixture cases:

- Closed-form and SciPy fit agree on linear model.
- Nonlinear fit recovers constants within tolerance.
- Bound-constrained fit lands at bound with diagnostic.
- Singular matrix.
- Underdetermined constants.
- Non-convergence.
- Residual all zeros.
- Residual heavy tail.
- Residual heteroskedasticity.
- Incomparable observation models.

Required live cases:

- Live FMP bounded fit emits optimizer trace.
- Live FMP residuals choose or reject robust observation model through declared score policy.

### 31.5 Universality And Claims

Required fixture cases:

- Same law across environments.
- Same fit but different law across environments.
- Parameter drift.
- Support drift.
- Residual distribution drift.
- Hidden confounder.
- Structural break.
- Single environment.
- Too few samples per environment.
- Strong in-sample but weak out-of-sample result.

Required live cases:

- Live FMP multi-symbol environments.
- Live rolling-era pseudo-environments.
- Live abstention for insufficient invariance.
- Live downgrade when parameter dispersion exceeds threshold.

### 31.6 Stochastic Models

Required fixture cases:

- Gaussian residuals.
- Student-t heavy tails.
- Laplace residuals.
- Poisson counts.
- Negative-binomial overdispersion.
- Beta bounded response.
- Zero-inflated counts where unsupported.
- Mixture model with component collapse.
- Heteroskedastic scale.
- Calibration failure.

Required live cases:

- Live FMP residuals evaluated under at least Gaussian, Student-t, and Laplace where support permits.
- Live probabilistic result includes calibration diagnostics and abstains when sample size is too small.

### 31.7 Workbench And Publication

Required fixture cases:

- Artifact with no claim.
- Artifact with weak claim.
- Artifact with universal claim.
- Artifact with failed falsification.
- Artifact with missing live evidence.
- Artifact with malformed schema.
- Artifact containing fake secret-looking values.
- OpenAI explanation success.
- OpenAI explanation timeout.
- OpenAI explanation unavailable.

Required live cases:

- Live FMP workbench data path.
- Live OpenAI explanation path.
- Live missing-key display path.
- Live redaction checks through API responses and UI payloads.

## 32. Release Certification Dual-Gate Command Set

At the end of each phase and at final certification, the release verifier must run the following classes of commands. Exact file paths may grow as implementation proceeds, but the command classes may not be removed.

```bash
pytest -q tests/unit
pytest -q tests/integration
pytest -q tests/regression
pytest -q tests/golden
pytest -q tests/benchmarks
pytest -q tests/spec_compiler
python -m euclid release status
python -m euclid release verify-completion
python -m euclid release certify-research-readiness
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 ./scripts/live_api_smoke.sh
EUCLID_LIVE_API_TESTS=1 EUCLID_LIVE_API_STRICT=1 pytest -q tests/live
```

Final certification fails when:

- Any fixture suite fails.
- Any regression golden changes without an intentional replacement reason.
- Any live API test fails in strict mode.
- Any live API test leaks a secret.
- Any live API artifact lacks provider, endpoint class, query window, schema version, row count, timeout, and semantic status where applicable.
- Any phase gate manifest omits a task or subtask.
- Any strong claim lacks fixture evidence, live evidence, replay evidence, falsification evidence, and claim-scope evidence.
- Any provider volatility is asserted as an exact deterministic value.
- Any external engine output bypasses Euclid-owned CIR, fitting, scoring, replay, falsification, and publication gates.
