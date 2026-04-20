# Euclid Pipeline Mathematics

This document summarizes the mathematical objects, loss laws, and aggregation rules used by Euclid’s runtime pipeline.

## 1) Evaluation geometry and weighted horizons

Euclid builds a walk-forward evaluation plan with horizons
$\mathcal{H} = \{1,\dots,H\}$, where each scored origin contributes one target per horizon.

- Horizon weights are required to be a simplex over the declared horizons:
  - each weight $w_h \ge 0$
  - $\sum_{h\in\mathcal{H}} w_h = 1$
  - exact decimal sum is enforced (not approximate floating tolerance).
- If weights are omitted, default weights are near-uniform decimals with the final weight set as the exact remainder to force exact sum 1.

This horizon simplex is used by both point and probabilistic scoring aggregation.

## 2) Descriptive coding objective (MDL-style admission)

For descriptive-scope admissibility, Euclid computes total codelength:

$$
L_{\text{total}} = L_{\text{family}} + L_{\text{structure}} + L_{\text{literals}} + L_{\text{params}} + L_{\text{state}} + L_{\text{data}}.
$$

The data term is computed from quantized residuals.

### 2.1 Quantization

Euclid uses a fixed-step mid-tread quantizer with step $\Delta>0$:

$$
q(x) = \mathrm{round}_{\text{half-up}}\!\left(\frac{x}{\Delta}\right) \in \mathbb{Z}.
$$

Residual indices are
$r_t = q(y_t - \hat y_t)$.

### 2.2 Integer code lengths

- Zigzag integer map:
$$
\mathrm{zigzag}(z)=
\begin{cases}
2z & z\ge 0\\
-2z-1 & z<0
\end{cases}
$$
- Natural-number code length (Elias-delta-like form used in code):
$$
\ell(n)=\lfloor\log_2(n+1)\rfloor + 2\lfloor\log_2(\lfloor\log_2(n+1)\rfloor+1)\rfloor + 1.
$$

Then
$$
L_{\text{data}} = \ell(T) + \sum_{t=1}^T \ell(\mathrm{zigzag}(r_t)).
$$

### 2.3 Reference description and gain

Reference description bits are computed on the quantized observed sequence itself (same quantizer and code family):
$$
L_{\text{ref}} = \ell(T) + \sum_{t=1}^T \ell(\mathrm{zigzag}(q(y_t))).
$$

Description gain:
$$
G = L_{\text{ref}} - L_{\text{total}}.
$$

Candidate is descriptively admissible only when comparability/support checks pass and $G$ exceeds the configured minimum threshold.

## 3) Candidate fitting objectives

Euclid fits CIR families and composition operators primarily with least-squares style objectives.

### 3.1 Analytic family (weighted linear AR(1)-style form)

If parameter block contains `lag_coefficient`, Euclid fits
$$
\hat y_t = \beta_0 + \beta_1 x_t,
$$
with $x_t=\text{lag}_1$, by weighted least squares closed-form moments:

$$
\bar x_w=\frac{\sum_t w_t x_t}{\sum_t w_t},\quad
\bar y_w=\frac{\sum_t w_t y_t}{\sum_t w_t},
$$
$$
\beta_1=\frac{\sum_t w_t(x_t-\bar x_w)(y_t-\bar y_w)}{\sum_t w_t(x_t-\bar x_w)^2},\quad
\beta_0=\bar y_w-\beta_1\bar x_w.
$$
(If denominator is zero, slope is set to 0.)

Objective:
$$
\mathcal{L}=\sum_t w_t(y_t-\hat y_t)^2.
$$

If no lag coefficient is present, model collapses to constant mean (weighted intercept-only least squares).

### 3.2 Recursive family

Euclid supports recursive state semantics (e.g., running level / running mean variants), selecting parameters/state updates by minimizing one-step squared residual accumulation over training rows.

### 3.3 Spectral family

Fitted forecast basis is harmonic:
$$
\hat y = a\cos(\theta)+b\sin(\theta),\quad
\theta=\frac{2\pi\,k\,\text{phase}}{S},
$$
with harmonic index $k$, season length $S$, and fitted coefficients $(a,b)$. Training objective remains least squares.

### 3.4 Algorithmic family

Algorithmic programs are executed stepwise over integer/fractional state and lagged observation windows; objective is sum of squared errors between emitted values and observed targets:
$$
\mathcal{L}=\sum_t(\hat y_t-y_t)^2.
$$

## 4) Composition mathematics

### 4.1 Piecewise

Rows are routed to branch reducers using partition predicates. Each branch is fitted on its assigned subset.

Total objective is additive across branches:
$$
\mathcal{L}_{\text{piecewise}} = \sum_b \mathcal{L}_b.
$$

### 4.2 Additive residual

Two-stage fit:
1. Fit base reducer to get $\hat y_t^{(b)}$.
2. Fit residual reducer on
$r_t = y_t - \hat y_t^{(b)}$.

Combined prediction:
$$
\hat y_t = \hat y_t^{(b)} + \hat r_t.
$$
Objective recorded as sum of component losses.

### 4.3 Regime-conditioned

- Hard switch: one branch active per row; branchwise fitting on routed subsets; losses summed.
- Convex weighting: branch-specific weighted fits using regime weights $\pi_{t,b}$; losses summed over branches.

Inference combines branch forecasts as
$$
\hat y_{t+h}=\sum_b \pi_{t,b}\,\hat y_{t+h}^{(b)}.
$$

### 4.4 Shared-plus-local decomposition (panel)

Two implementations:

1. **Baseline mean-offset fit**
   - shared intercept $\mu$ = panel mean target
   - entity local adjustment $\alpha_e = \bar y_e - \mu$
   - prediction: $\hat y = \mu + \alpha_e$
   - objective: panel SSE.

2. **Joint panel optimizer**
   - least-squares linear model with shared intercept + shared lag + entity-specific intercept/lag offsets (design matrix with entity indicators and lag-interaction indicators).

Euclid chooses optimizer when available and strictly better than baseline (with small tolerance); otherwise baseline.

## 5) Point forecast generation equations

From fitted parameters/state, Euclid emits horizon path forecasts.

- Analytic AR-style recursion:
$$
\hat y_{h} = \beta_0 + \beta_1\hat y_{h-1},\quad \hat y_0 = y_{\text{origin}}.
$$
(If no lag term, all horizons equal intercept.)

- Recursive family: horizon path is constant at fitted level/running mean.

- Spectral family: harmonic expression evaluated per advanced phase index.

- Additive residual path:
$$
\hat y_h = \hat y_h^{(b)} + \hat y_h^{(r)}.
$$

- Regime-conditioned path:
$$
\hat y_h = \sum_b \pi_b\hat y_h^{(b)}.
$$

- Shared-local path:
$$
\hat y_h = (\beta_0^{\text{shared}} + \alpha_e) + \beta_{1,e}\hat y_{h-1},
$$
where $\beta_{1,e}$ comes from shared lag + local lag adjustment (or explicit local lag coefficient if present).

## 6) Probabilistic support construction

Probabilistic artifacts are built from point paths by attaching Gaussian location-scale support per horizon.

- Location always equals point forecast for that horizon.
- Scale uses family-specific growth from a base scale $s_0$.

### 6.1 Base scale

$$
s_0 = \max\left(\sqrt{\frac{\max(\text{final\_loss},0)}{\max(N,1)}},\;0.25 + 0.05\,\max_j |\theta_j|\right).
$$

### 6.2 Horizon scaling by family

- Analytic: $s_h=s_0\sqrt{h}$
- Recursive: $s_h=s_0(1+0.15(h-1))$
- Spectral: $s_h=s_0(1+(A/10)h)$, with $A=\max(|a|,|b|,1)$
- Algorithmic: $s_h=s_0(1+0.2(h-1))$

## 7) Forecast object derivations

Given Gaussian support $\mathcal N(\mu_h,s_h^2)$:

1. **Distribution object**: emits $(\mu_h,s_h)$.
2. **Interval object** (nominal 0.8):
   $$
   [\mu_h-z_{0.9}s_h,\;\mu_h+z_{0.9}s_h],\quad z_{0.9}\approx1.281551565545.
   $$
3. **Quantile object** at levels $0.1,0.5,0.9$:
   $$
   q_\tau = \mu_h + z_\tau s_h,
   $$
   with hard-coded z-scores $(-1.281551565545,0,1.281551565545)$.
4. **Event probability object** for event $Y\ge \text{origin target}$:
   $$
   p_h = 1-\Phi\left(\frac{\text{threshold}-\mu_h}{s_h}\right).
   $$

## 8) Scoring laws

Let $y$ be realized observation.

### 8.1 Point scoring

Supported point losses:
- Squared error: $(\hat y-y)^2$
- Absolute error: $|\hat y-y|$

### 8.2 Probabilistic scoring

1. **Gaussian CRPS** for location-scale normal:
$$
\text{CRPS}(\mu,s;y)=s\left[z(2\Phi(z)-1)+2\phi(z)-\frac{1}{\sqrt\pi}\right],
\quad z=\frac{y-\mu}{s}.
$$

2. **Gaussian log score** (negative log-likelihood):
$$
\frac12\log(2\pi s^2)+\frac{(y-\mu)^2}{2s^2}.
$$

3. **Interval score** for nominal coverage $1-\alpha$, bounds $[l,u]$:
$$
S=(u-l)+\frac{2}{\alpha}(l-y)\mathbf{1}_{y<l}+\frac{2}{\alpha}(y-u)\mathbf{1}_{y>u}.
$$

4. **Quantile pinball (average over declared quantiles)**:
$$
\rho_\tau(y-q)=
\begin{cases}
\tau(y-q), & y\ge q\\
(\tau-1)(y-q), & y<q
\end{cases}
$$
with score as mean of $\rho_\tau$ over quantiles in row.

5. **Event probability scores**:
- Brier: $(p-\mathbb{1}\{\text{event}\})^2$
- Log score:
  - event true: $-\log p$
  - event false: $-\log(1-p)$
  - code returns $+\infty$ on boundary contradictions (e.g., $p=0$ when event true).

## 9) Aggregation across origins, horizons, and entities

For each origin-horizon row, compute primary score $s_{o,h}$.

### 9.1 Single-entity mode

Per-horizon mean:
$$
\bar s_h = \frac{1}{|\mathcal O|}\sum_{o\in\mathcal O}s_{o,h}.
$$
Aggregated primary score:
$$
S = \sum_{h\in\mathcal H} w_h\bar s_h.
$$

### 9.2 Per-entity weighted mode

Let entity weights $v_e$ form simplex. Per-horizon metric:
$$
\bar s_h = \sum_e v_e\left(\frac{1}{|\mathcal O_e|}\sum_{o\in\mathcal O_e}s_{o,h}\right).
$$
Final aggregate computed equivalently as weighted sum of each entity’s horizon-weighted mean score:
$$
S=\sum_e v_e\left(\sum_h w_h\frac{1}{|\mathcal O_e|}\sum_{o\in\mathcal O_e}s_{o,h}\right).
$$

Both horizon and entity weights are validated as exact decimal simplexes.

## 10) Comparator math and practical significance

For point comparator evaluation, Euclid aligns candidate and comparator on forecast object type, score policy, horizon set, scored-origin set, and panel metadata.

Loss differential is formed per matched origin/horizon and aggregated to mean differential $\Delta$. Practical significance status is:
- candidate_better_than_margin if $\Delta > m$
- candidate_worse_than_margin if $\Delta < -m$
- within_margin otherwise.

Here $m\ge0$ is user-supplied practical significance margin.

## 11) Calibration diagnostics

### 11.1 Distribution (PIT uniformity proxy)

For each row: $u_i=\Phi((y_i-\mu_i)/s_i)$.
KS distance to Uniform(0,1):
$$
D=\max_i\left(\frac{i}{n}-u_{(i)},\;u_{(i)}-\frac{i-1}{n}\right).
$$
Pass if $D\le$ threshold (default 0.25).

### 11.2 Interval

Empirical coverage minus nominal coverage absolute gap:
$$
|\hat c - c_{nom}|,
$$
pass if gap $\le$ threshold (default 0.1).

### 11.3 Quantile

For each quantile level $\tau$, compute hit rate
$\hat h_\tau = \frac{1}{n}\sum \mathbb{1}\{y\le q_\tau\}$, then gap
$|\hat h_\tau-\tau|$. Use max gap across levels; pass if $\le$ threshold (default 0.15).

### 11.4 Event probability

Group rows by predicted probability $p$, compare empirical event frequency in each bin to $p$, and take maximum absolute reliability gap. Pass if $\le$ threshold (default 0.2).

## 12) Numerical policy notes

- Euclid frequently applies a 12-decimal rounding normalization (`_stable_float`) before emitting artifacts/scores, improving deterministic reproducibility.
- Non-finite values (NaN/Inf) cause comparability failure or missing-origin diagnostics depending on stage.
- Time-safety is mathematically enforced as equality of expected and observed `available_at` for each scored origin, preventing forward-looking leakage.
