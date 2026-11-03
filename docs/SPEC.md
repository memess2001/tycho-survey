# Tycho — implementation spec (developer contract)

Tycho is a 24×7 "sky survey" of neural-network training dynamics, built to run unattended on a Mac mini
for weeks to months. Every training run is an **object**. Every object gets cheap **photometry**:
light curves of loss, accuracy, norms, effective rank and eNTK statistics. Selected objects get expensive
**spectroscopy**: time-resolved *refined information* (RI), which tracks which subsets of variables
start interacting, at which order, at which training step. The results go into a **catalog** (SQLite plus
files). A Claude-based agent does nightly QC, classification review and proposals, and writes a weekly report.

Read `notes/origin_conversation.md` for the scientific motivation (Tycho/Kepler analogy, two-tier observation,
calibration-before-survey, transients, data releases).

This file is the **contract** between modules. Where it pins a signature, implement exactly that signature
(extra optional kwargs with defaults are fine). Where it is silent, use judgment and document it in docstrings.

---------------------------------------------------------------------------------------------------------
## 0. Conventions

* Python ≥ 3.10. Package `tycho/`, tests in `tests/` (pytest). Virtualenv: `/home/user/tycho/.venv`
  (`.venv/bin/python -m pytest ...`). Installed: numpy, scipy, torch (CPU), pyyaml, matplotlib, pandas, pyarrow,
  anthropic, pytest, pytest-timeout. **Do not install other packages.**
* Target hardware: Apple-silicon Mac mini (CPU + MPS). Code must run on Linux CPU too (tests run here).
  Never require MPS/CUDA; guard with `torch.backends.mps.is_available()`.
* RI math is numpy `float64`, natural logs (nats). Information values are in nats everywhere.
* Randomness is always explicit: `np.random.Generator` / `torch.Generator` passed in, never global state
  (except `torch.manual_seed(seed)` right before model construction, which is allowed for init determinism).
* No network access in library code paths that tests exercise (the agent runner and notifier are the only
  network users and must be mockable).
* Logging: `logging.getLogger(__name__)`; no prints in library code (CLI may print).
* Tests: each test file should run in ≲ 60 s on 4 CPU cores. Mark slower tests `@pytest.mark.slow`
  (skipped unless `--runslow`; see `tests/conftest.py`).
* Matplotlib: always `matplotlib.use("Agg")` before importing pyplot.
* File ownership: each builder owns specific files (listed in the build brief). Do not edit files you do not
  own. The contract files (§1) are owned by the orchestrator. If you need a change there, add it only if
  it is backward compatible (new optional field with default) and report it explicitly.

---------------------------------------------------------------------------------------------------------
## 1. Contract files (already written by the orchestrator — import, don't redefine)

* `tycho/config.py` — `Paths`, `DEFAULT_CONFIG`, `load_config(root)`, `deep_merge`.
* `tycho/ri/variables.py` — `VariableSet` (the discrete variables an RI analysis runs on).
* `tycho/analysis/ricurve.py` — `RICurve` (time-resolved RI of many subsets; save/load).
* `tycho/train/spec.py` — `TrainConfig`, `RIConfig`, `ObjectSpec`, `RunResult`.

---------------------------------------------------------------------------------------------------------
## 2. The instrument: refined information (`tycho/ri/`)

### 2.1 Math (implement exactly this)

Variables X = (X_1..X_d), X_i categorical with k_i states. A joint distribution is a probability table P of shape
(k_1,...,k_d). A *subset* S is a sorted tuple of axis indices. A family H of subsets is *hierarchical*
(an order ideal / down-set of the Boolean lattice) if S ∈ H and ∅ ≠ T ⊂ S imply T ∈ H. By convention every model
contains all singletons.

Hierarchical log-linear model E_H (Amari 2001; Sugiyama et al. 2016 "information decomposition on structured
space"; Enouen & Sugiyama, TMLR 2026 "A Complete Decomposition of KL Error using Refined Information and Mode
Interaction Selection"):

    log q(x) = Σ_{S ∈ H} θ_S(x_S) − ψ(θ)

E_H is e-flat. The **m-projection** of P onto E_H is P_H = argmin_{Q ∈ E_H} D(P‖Q). It is the unique member of E_H
whose S-marginals equal P's S-marginals for all S ∈ H (equivalently: the maximum-entropy distribution with those
marginals; equivalently: the MLE of the hierarchical model if P is empirical). Compute it by **iterative
proportional fitting (IPF)** over the maximal elements ("generators") of H, starting from the uniform
distribution (or from a warm start that lies in E_{H'} with H' ⊆ H).

Pythagorean relation for H ⊆ H': D(P‖P_H) = D(P‖P_{H'}) + D(P_{H'}‖P_H).

**Chain refined information.** A complete hierarchical chain is H_0 ⊂ H_1 ⊂ … ⊂ H_m with H_j = H_{j−1} ∪ {S_j},
each H_j hierarchical (S_j is *admissible*: |S_j| ≥ 2, S_j ∉ H_{j−1}, every proper nonempty subset of S_j is in
H_{j−1}). The refined information of S_j along the chain is the divergence drop

    RI_C(S_j) = D(P‖P_{H_{j−1}}) − D(P‖P_{H_j}) = D(P_{H_j} ‖ P_{H_{j−1}}) ≥ 0.

Summing along a chain from H_0 = singletons to the full power set gives a **complete decomposition** of the KL
error of the independence model, i.e. of the total correlation (multi-information):

    TC(X) = Σ_i H(X_i) − H(X) = D(P‖P_{H_0}) = Σ_j RI_C(S_j).

For d = 2 the single step is the mutual information. RI_C depends on the ordering among incomparable subsets;
the implementation must expose the chain explicitly.

**Local refined information** of a subset S (the default quantity used for event detection; chain-independent):
take the marginal P_S of the variables in S and the hierarchical family ∂S = all proper nonempty subsets of S:

    RI(S) = D(P_S ‖ (P_S)_{∂S})

RI(S) is the top-order term for S. It equals RI_C(S) for any chain in which S is added right after its
proper subsets, with nothing else involving S's variables. It is ≥ 0, it equals MI for |S| = 2, and it is 0 iff
the S-marginal has no |S|-th-order interaction. Its finite-sample estimator has a known null distribution (§2.3).
This makes it the calibrated "spectral line". (Amari's k-cut coordinates and Schneidman et al.'s connected
information of top order coincide with this quantity.)

> NOTE: The orchestrator could not access the full text of Enouen & Sugiyama (network-restricted). The chain
> definition above matches the paper's abstract ("drop in divergence between successive projected distributions
> along a chain of hierarchical collections; non-negative; extends mutual information; completely decomposes the
> KL error"). The user (who works with this framework) must cross-check notation and canonical chain choice against
> the paper; keep both `chain_decomposition` and `local_ri` so either convention can be used.

**Binary spin coding** for constructing distributions (used by calibration/tasks): for binary variables with
codes x_i ∈ {0,1}, spins s_i = 2x_i − 1 ∈ {−1,+1}; a scalar θ_S contributes θ_S·∏_{i∈S} s_i to log q.
**Möbius (Sugiyama/Amari) θ-coordinates** for binary tables use 0/1 coding:
θ_S = Σ_{T⊆S} (−1)^{|S∖T|} log p(1_T) (1_T = the configuration with ones exactly on T); η_S = P(X_i = 1 ∀ i ∈ S).

### 2.2 `tycho/ri/hypergraph.py`

```python
Subset = tuple[int, ...]
def nonempty_subsets(S: Sequence[int], proper: bool = False) -> list[Subset]   # graded-lex order
def down_closure(H: Iterable[Sequence[int]], d: int | None = None) -> frozenset[Subset]  # adds singletons of all axes < d if d given
def is_hierarchical(H: Iterable[Sequence[int]]) -> bool
def maximal_elements(H: Iterable[Sequence[int]]) -> list[Subset]                # graded-lex order
def is_admissible(H: frozenset[Subset], S: Sequence[int]) -> bool
def admissible_candidates(H: frozenset[Subset], d: int, max_order: int | None = None) -> list[Subset]
def graded_order(d: int, min_order: int = 2, max_order: int | None = None) -> list[Subset]
```
"graded-lex" = sort by (len, tuple).

### 2.3 `tycho/ri/loglinear.py`

```python
def joint_table(X: np.ndarray, cards: Sequence[int], *, weights: np.ndarray | None = None,
                soft: dict[int, np.ndarray] | None = None, normalize: bool = True,
                pseudocount: float = 0.0) -> np.ndarray
    # X (N, d) integer codes, column i in [0, cards[i]). soft: {j: (N, cards[j]) row-stochastic}; at most one
    # soft column; sample n contributes weight w_n * soft[j][n, c] to cell (x_{n,-j}, c). Returns float64 array of
    # shape tuple(cards); sums to 1 if normalize else to Σw (+ pseudocount * n_cells).
def marginal(P: np.ndarray, S: Sequence[int]) -> np.ndarray     # axes kept in sorted(S) order
@dataclass
class ProjectionResult:
    Q: np.ndarray; n_iter: int; max_err: float; converged: bool
def m_project(P: np.ndarray, H: Iterable[Sequence[int]], *, tol: float = 1e-12, max_iter: int = 10_000,
              init: np.ndarray | None = None) -> ProjectionResult
    # IPF over maximal elements of down_closure(H ∪ singletons). Fast paths: full set in H -> Q = P;
    # only singletons -> product of marginals. 0/0 := 0. Converged when, after a full sweep, the max abs deviation
    # of every generator marginal is < tol. Warn (logging) if not converged.
def kl(P: np.ndarray, Q: np.ndarray) -> float          # Σ p log(p/q), 0 log 0 = 0, inf if q=0<p
def entropy(P: np.ndarray) -> float
def mutual_information(P: np.ndarray, A: Sequence[int], B: Sequence[int]) -> float
def total_correlation(P: np.ndarray) -> float
def loglinear_table(theta: dict[Subset, float | np.ndarray], cards: Sequence[int]) -> np.ndarray
    # scalar theta_S requires all cards[i]==2 for i in S and uses spin coding; array theta_S has shape
    # tuple(cards[i] for i in S) and is added directly. Returns normalized table.
def theta_binary(P: np.ndarray) -> dict[Subset, float]   # Möbius θ (0/1 coding) for all nonempty S; P > 0 required
def eta_binary(P: np.ndarray) -> dict[Subset, float]
```

### 2.4 `tycho/ri/refined.py`

```python
@dataclass
class RIStep:
    subset: Subset; order: int; ri: float; kl_before: float; kl_after: float
@dataclass
class ChainDecomposition:
    base: tuple[Subset, ...]; steps: list[RIStep]; kl_base: float; residual: float
    @property
    def total(self) -> float   # Σ step.ri ;   invariant: kl_base ≈ total + residual
    def as_dict(self) -> dict[Subset, float]
def chain_decomposition(P, chain: Sequence[Sequence[int]], *, base=None, tol=1e-12) -> ChainDecomposition
    # raises ValueError if a chain element is not admissible at its position. RI_j computed as kl(Q_j, Q_{j-1}).
def graded_chain(d: int, max_order: int | None = None) -> list[Subset]
def greedy_selection(P, *, max_order=None, max_steps=None, min_gain=0.0) -> ChainDecomposition
    # "mode interaction selection": at each step add the admissible subset with the largest RI gain.
def local_ri(P: np.ndarray) -> float              # RI of the set of ALL axes of P (top-order term); 0 for 1-D
def local_ri_batch(tables: np.ndarray, *, tol=1e-12, max_iter=10_000) -> np.ndarray
    # tables (B, k1..km): each normalized; vectorized IPF across the batch axis; returns (B,)
def local_ri_from_table(P: np.ndarray, subsets: Sequence[Sequence[int]]) -> np.ndarray   # population values
```
Performance target: `local_ri_batch` on (20_000, 2,2,2,2) in < 2 s on one core.

### 2.5 `tycho/ri/estimators.py`

```python
def ri_df(cards: Sequence[int]) -> int        # Π (k_i − 1)
@dataclass
class RIEstimate:
    subset: Subset; order: int; ri_plugin: float; ri_corrected: float; df: int; n_eff: float
    g2: float; p_value: float; ci_low: float = nan; ci_high: float = nan; null: str = "chi2"
def estimate_local_ri(X, cards, subsets, *, soft=None, weights=None, null="chi2", n_null=200,
                      ci=False, n_boot=200, ci_level=0.95, rng=None, pseudocount=0.0,
                      n_eff=None) -> list[RIEstimate]
    # plug-in RI from empirical marginal tables (grouped by table shape, batched).
    # ri_corrected = max(0, ri_plugin − df/(2 n_eff))      (first-order null bias: E[G²] = df)
    # g2 = 2 n_eff ri_plugin ; p_value = chi2.sf(g2, df)   (LRT of nested hierarchical log-linear models)
    # null="parametric_bootstrap": draw n_null multinomial tables of size round(n_eff) from the fitted null
    #   (m-projection of the subset table onto ∂S), recompute RI, p = (1 + #{null ≥ obs}) / (n_null + 1).
    # ci=True: row-bootstrap percentile CI of ri_corrected.
    # n_eff default: N, or (Σw)²/Σw² with weights.
def bh_fdr(pvals: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]   # (rejected bool, q-values); NaN-safe
def candidate_subsets(d: int, *, max_order: int, min_order: int = 2,
                      must_include: Sequence[int] | None = None,
                      among: Sequence[int] | None = None) -> list[Subset]
@dataclass
class ScanResult:
    names: list[str]; subsets: list[Subset]; estimates: list[RIEstimate]; q_values: np.ndarray; detected: np.ndarray
    def to_records(self) -> list[dict]
def ri_scan(vs: VariableSet, *, max_order: int = 3, min_order: int = 2, must_include=None, alpha: float = 0.01,
            min_effect: float = 1e-3, null: str = "chi2", max_subsets: int = 200_000, rng=None) -> ScanResult
    # must_include defaults to vs.must_include. detected = (q <= alpha) & (ri_corrected >= min_effect).
    # Raise ValueError if the candidate count exceeds max_subsets.
```

### 2.6 `tycho/ri/discretize.py`

```python
def binarize(A: np.ndarray, method: str = "median", q: float = 0.5) -> np.ndarray   # (N,m) -> int8
    # "zero": A > 0 ; "median": A > per-column median ; "mean": A > mean ; "quantile": A > per-column q-quantile
def select_informative(B: np.ndarray, k: int, *, method: str = "entropy", rng=None,
                       exclude_constant: bool = True) -> np.ndarray   # column indices
```

### 2.7 `tycho/ri/__init__.py`
Re-export: `VariableSet, m_project, kl, local_ri, local_ri_batch, chain_decomposition, graded_chain,
greedy_selection, estimate_local_ri, ri_scan, bh_fdr, binarize`.

---------------------------------------------------------------------------------------------------------
## 3. Calibration — "build Tycho's instrument first" (`tycho/calibration/`)

`planted.py`
```python
def planted_binary(d: int, interactions: dict[Subset, float], *, biases: dict[int, float] | None = None) -> np.ndarray
def random_interactions(d: int, orders: Sequence[int], scale: float, rng) -> dict[Subset, float]
def sample_table(P: np.ndarray, n: int, rng) -> np.ndarray          # (n, d) int codes
def true_local_ri(P: np.ndarray, subsets) -> np.ndarray
def switch_on(t: np.ndarray, t_star: float, width: float, theta_max: float) -> np.ndarray   # sigmoid ramp
def gaussian_binarized(corr: np.ndarray, n: int, rng, method: str = "median") -> np.ndarray
```
`suite.py`: `@dataclass CalibrationConfig` (with `quick()` and `full()` constructors) and
`run_calibration(cfg, out_dir: Path, *, n_jobs: int = 1) -> dict` running experiments, each producing a list of
row dicts saved as `out_dir/<exp>.csv` and a summary dict:

* **E1 null / false-positive rate**: data with lower-order structure only (orders < k, strength ∈ {0, 0.5, 1}),
  N grid, k ∈ {2,3,4}. Report FPR at α (0.01, 0.05), KS statistic of p-values vs U(0,1), null quantiles of
  ri_plugin (the noise floor: 50/95/99th percentile per (k, N)).
* **E2 bias / RMSE / CI coverage**: planted θ on a k-subset (plus random lower-order background), true RI from the
  population table. Report bias, RMSE of ri_plugin and ri_corrected, and bootstrap CI coverage.
* **E3 power**: detection probability (q ≤ α & ri_corrected ≥ min_effect) vs (true RI, N, k), and the
  "minimum detectable RI at 80 % power" per (k, N).
* **E4 timing**: θ_S(t) = switch_on(t, t*, width, θ_max) observed on T checkpoints, N samples each. Run
  `tycho.analysis.events.detect_events` and report the onset error (detected − true onset, in checkpoints) vs
  (width, N). True onset := first checkpoint where the population RI ≥ min_effect.
* **E5 identity / specificity**: d = 8 binary, planted pure 3-way {0,1,2} plus pairwise {3,4} plus weak
  background; full scan to order 4. Precision/recall of detected subsets vs ground truth.
* **E6 discretization artifacts (RQ0)**: equicorrelated Gaussian latent (ρ grid), binarize (median / zero / quantile
  0.8), report local RI at orders 2, 3, 4 averaged over subsets. (Thresholding a Gaussian induces higher-order
  log-linear interactions even though the latent has only pairwise structure. This is a systematic error of the
  instrument.)
* **E7 chain dependence**: random positive tables (d = 4–5, mixed cards); RI of each subset across graded,
  reverse-graded and greedy chains (spread), and check Σ RI = TC.

`report.py`: `write_report(results: dict, out_dir: Path) -> Path` → `CALIBRATION.md` (an error-budget table and
per-experiment sections) + PNG plots + `calibration.json` (machine-readable error budget: noise floor table,
min-detectable-RI table, timing error table, artifact table). The survey reads `calibration.json` to set
`min_effect` defaults.

---------------------------------------------------------------------------------------------------------
## 4. Objects: tasks, models, training (`tycho/tasks`, `tycho/models`, `tycho/train`)

### 4.1 Task interface (`tycho/tasks/base.py`)
```python
class Task:
    name: ClassVar[str]
    def __init__(self, cfg: dict, seed: int)            # self.cfg = {**defaults(), **cfg}; unknown keys -> ValueError
    @classmethod
    def defaults(cls) -> dict
    def build_model(self) -> nn.Module                   # torch.manual_seed(self.seed) inside -> deterministic init
    def sample_batch(self, batch_size: int, gen: torch.Generator) -> tuple[Tensor, Tensor]
    def eval_splits(self) -> dict[str, tuple[Tensor, Tensor]]   # fixed; must contain "train" and "test"
    def loss(self, out, y) -> Tensor
    def accuracy(self, out, y) -> float | None
    def chance_accuracy(self) -> float | None
    def ntk_output(self, out, y) -> Tensor               # per-sample scalar for eNTK (e.g. true-class logit margin)
    def ri_variables(self, model: nn.Module, n: int, gen: torch.Generator) -> VariableSet
    def ground_truth(self) -> list[dict] | None          # [{"subset": [names...], "order": k, ...}]
    def extra_metrics(self, model) -> dict[str, float]   # task-specific light-curve metrics (keys without prefix)
    def default_ri_order(self) -> int
REGISTRY: dict[str, type[Task]]; def make_task(name: str, cfg: dict, seed: int) -> Task
```
Data randomness that defines the *problem* (e.g. the parity subset, the train/test split, Markov couplings) uses
`cfg["data_seed"]` (default 0) so it is shared across training seeds of the same config. The training seed controls
init and batch order.

### 4.2 Tasks (each in its own module, registered in REGISTRY)

1. **`sparse_parity`** (positive control: pure (k+1)-order synergy, "hidden progress").
   cfg: n=20, k=3, width=256, depth=1, act="relu", n_train=None (online fresh samples), test_size=2048,
   data_seed=0, ri_soft=True. S* = sorted(np.random.default_rng(data_seed).choice(n, k, replace=False)).
   x ∈ {±1}^n float, y = 1 if ∏_{i∈S*} x_i = −1 else 0. MLP → 2 logits, CE loss.
   RI variables: bits "x0".."x{n-1}" (1 if x_i=+1) and "yhat" (soft softmax column if ri_soft else argmax);
   must_include=[yhat]. ground_truth: [{"subset": [x_i for i in S*] + ["yhat"], "order": k+1}].
   default_ri_order = k+1.
2. **`markov_lang`** (positive control with tokens, staged orders).
   Binary tokens. Stationary order-W Markov source with conditional
   P(x_t = 1 | ctx) = σ(2·h), h = Σ_{L ∈ couplings} β_L ∏_{ℓ∈L} s_{t−ℓ}   (spins; the coupling of x_t's spin with the
   product of lagged spins is a |L|+1-order interaction). cfg: window=6, couplings={"1": 1.0, "1,3": 0.8}
   (keys = comma-separated lags), bias=0.0, n_tokens=200_000, width=128, depth=1, model="mlp" ("transformer" also
   allowed: TinyGPT over the window), data_seed=0, n_train=None (sample random positions of the long sequence).
   Input: the last `window` tokens as ±1 floats (MLP) or token ids (transformer). Target: x_t.
   RI variables: "x[t-1]".."x[t-W]" plus "xhat" (soft = model softmax); must_include=[xhat].
   ground_truth: one entry per coupling L: subset = ["x[t-ℓ]" for ℓ in L] + ["xhat"], order = |L|+1,
   "beta": β_L. Extra method `true_variables(n, gen) -> VariableSet` with soft column = the true conditional (the
   "true spectrum").
3. **`modular_addition`** (grokking object). cfg: p=23, train_frac=0.5, width=128, depth=1, data_seed=0,
   embed="onehot". Input = concat(onehot(a), onehot(b)) (2p floats); target (a+b) mod p; CE.
   eval "train" = full train set, "test" = held-out pairs. RI variables: inputs = all p² pairs; hidden units of the
   last hidden layer binarized with `binarize(method="zero")`, choose top `ri_units` (default 10) by entropy
   (`select_informative`), plus "correct" (prediction correct). must_include=None; ground_truth None;
   default_ri_order=3.
4. **`icl_markov`** (Edelman et al. 2024: in-context Markov chains; staged unigram → bigram).
   cfg: vocab=3, seq_len=48, alpha=1.0 (Dirichlet), d_model=32, n_layer=2, n_head=1, attn_only=True.
   Each sequence: transition matrix rows ~ Dirichlet(alpha), start from its stationary distribution (or uniform),
   sample seq_len tokens. Model TinyGPT; loss = mean next-token CE over positions 1..T−1.
   eval "train" and "test": fixed sets of 512 sequences (different data seeds).
   extra_metrics: "kl_to_unigram", "kl_to_bigram" = mean KL(strategy ‖ model) over positions ≥ 8, where
   unigram = in-context token frequency (add-1/2 smoothing), bigram = in-context transition counts from the current
   token (add-1/2 smoothing; Bayes-optimal under Dirichlet(1) is add-1; use alpha-consistent smoothing).
   accuracy: next-token argmax accuracy. RI variables (per sampled position t ≥ 8): "x_t" (current token),
   "u_t" (argmax of in-context unigram counts), "b_t" (argmax of in-context successor counts of x_t; fallback u_t),
   "xhat" (soft); must_include=[xhat]. ground_truth: [{"subset": ["u_t","xhat"], "order": 2, "stage": 1},
   {"subset": ["b_t","xhat"], "order": 2, "stage": 2}].
5. **`toy_superposition`** (Elhage et al. 2022). cfg: n_features=20, m_hidden=5, feature_prob=0.1,
   importance_decay=0.9. x_i = U(0,1) w.p. feature_prob else 0. Model ReLU(Wᵀ W x + b). Loss = mean_batch Σ_i I_i
   (x_i − x̂_i)², I_i = decay^i. accuracy None. extra_metrics: "n_represented" (#‖W_i‖>0.5), "superposition"
   (mean over represented i of Σ_{j≠i} (Ŵ_i·W_j)²), "dim_per_feature" (mean of D_i = ‖W_i‖⁴ / Σ_j (W_i·W_j)², i
   represented). RI variables: for the 5 most important features: "a{i}" (x_i > 0) and "r{i}" (x̂_i > 0.1);
   must_include=None; ground_truth None.

### 4.3 Models (`tycho/models/`)
* `mlp.py`: `MLP(in_dim, width, out_dim, depth=1, act="relu")`, `.forward(x)`, `.hidden(x) -> list[Tensor]`
  (post-activation of each hidden layer).
* `transformer.py`: `TinyGPT(vocab, d_model, n_layer, n_head, max_len, attn_only=False, mlp_ratio=4)`,
  `.forward(idx, return_hidden=False)` → logits (B,T,V) [and list of residual-stream states]. Causal, learned positions.
* `superposition.py`: `ToyReLU(n_features, m_hidden)`.

### 4.4 Training (`tycho/train/`)
* `spec.py` (contract; see file).
* `device.py`: `pick_device(policy: str, n_params: int) -> torch.device` ("cpu" | "mps" | "auto": mps only if
  available and n_params ≥ 2e6, else cpu), `configure_torch(threads: int, deterministic: bool = True)`.
* `schedule.py`: `eval_steps(total: int, n_log: int, n_lin: int) -> np.ndarray` (sorted unique ints, includes 0 and
  total; log-spaced from 1, plus linear) and `subsample_log(steps, n) -> np.ndarray` (≈ log-uniform subset, keeps
  first and last).
* `photometry.py`: functions `param_norm`, `effective_rank(M)` (Roy–Vetterli: exp of entropy of normalized singular
  values), `ntk_gram(model, x, y, task) -> Tensor (n,n)` (per-sample gradients of `task.ntk_output`; use
  `torch.func` with a loop fallback), `ntk_stats(K, y=None) -> dict` (trace, top_eig, erank, align = kernel–target
  alignment with centered ±1 / one-hot labels when available), `linear_cka(X, Y)`; class `Photometer` holding the
  previous state, with `measure(step, model, task, splits, grad_norm, lr, do_ntk) -> dict[str, float]`.
  **Light-curve keys** (always present, NaN when not applicable): `step, train_loss, test_loss, train_acc,
  test_acc, param_norm, grad_norm, update_norm, dist_from_init, erank_weights, erank_hidden, rep_cka_prev,
  ntk_trace, ntk_top_eig, ntk_erank, ntk_align, ntk_velocity, lr, wall_s` plus task extras as `x_<name>`.
  `ntk_velocity` = 1 − cos(K_t, K_prev) over successive NTK measurements.
* `runner.py`: `run_object(spec: ObjectSpec, *, work_dir: Path, ri: RIConfig | None = None,
  stop_flag: Callable[[], bool] | None = None, on_progress: Callable[[int, int], None] | None = None,
  resume: bool = True, ri_steps: Sequence[int] | None = None) -> RunResult`
  - Deterministic on CPU for a given spec (same light curve bit-for-bit across runs with same thread count).
  - Resume checkpoint `work_dir/resume.pt` every `train.checkpoint_every_s` seconds of wall time and when
    `stop_flag()` becomes true (then return status "interrupted"). On start with resume=True and an existing file,
    continue exactly (restore model, optimizer, data generator state, photometer state, partial light curve and
    partial RI curve).
  - Non-finite loss → stop, status "done" with summary["diverged"] = True (not "failed"); exceptions → "failed".
  - RI in situ: if `ri` and `ri.enabled`, at steps `ri_steps` (default: `subsample_log(eval steps, ri.n_points)`)
    call `task.ri_variables(model, ri.n_probe, gen_ri)` (a dedicated generator seeded from spec seed so RI probing
    never perturbs training randomness) and `ri_scan`; assemble an `RICurve` (prune via `RICurve.prune`).
  - Summary from `tycho.analysis.summarize.summarize(light_curve, chance_acc)`.
* `replay.py`: `replay_object(spec, *, work_dir, ri: RIConfig, dense: tuple[int, int] | None = None,
  n_dense: int = 30) -> RunResult` (rerun from scratch with RI points = log points ∪ n_dense points in [lo, hi]) and
  `replay_fidelity(lc_a, lc_b, keys=("train_loss","test_loss")) -> float` (max relative deviation).

---------------------------------------------------------------------------------------------------------
## 5. Analysis (`tycho/analysis/`, pure numpy)

* `ricurve.py` (contract).
* `summarize.py`: `summarize(lc: dict[str, np.ndarray], chance_acc: float | None) -> dict` with documented keys:
  `final_*` for losses/accs, `min_test_loss`, `t_train_acc{50,90,99}`, `t_test_acc{50,90,99}` (first step at which
  acc ≥ chance + level·(1 − chance)... for 50/90/99 use *absolute* thresholds 0.5/0.9/0.99 when chance ≤ 0.5, else
  the rescaled form; NaN if never), `grok_delay` = t_test_acc90 − t_train_acc99 (NaN if either missing),
  `grok_ratio` = t_test_acc90 / max(t_train_acc99, 1), `plateau_frac` (fraction of total steps before the train
  loss first falls below L0 − 0.1(L0 − Lmin)), `n_drops` (number of distinct sharp drops of smoothed log train
  loss vs log step), `loss_spikes` (evals after the first drop where train loss > 2× running min),
  `test_bump` = max_t (test_loss(t) − min_{s<t} test_loss(s)) / max(test_loss(0), eps), `diverged`,
  `max_ntk_velocity`, `step_max_ntk_velocity`, `steps_total`.
* `classify.py`: `LABELS = ["diverged", "no_learning", "memorization", "grokking", "plateau_drop", "staged",
  "double_descent", "smooth", "unknown"]`; `classify(summary, lc=None, chance_acc=None) -> Classification`
  (dataclass: label, tags: list[str], score: float in [0,1], reasons: list[str]); rule set "rules-v1",
  first-match order: diverged → no_learning → memorization → grokking → staged → plateau_drop → double_descent →
  smooth → unknown. Document every threshold as a module constant.
* `events.py`: dataclasses `RIEvent` (subset: Subset, subset_names, order, kind ∈ {"activation", "extinction",
  "reactivation"}, step, step_lo, step_hi, ri_before, ri_after, peak_ri, confidence, has_precursor: bool | None,
  matches_ground_truth: bool | None) and `Transient` (kind ∈ {"late_activation", "extinction", "reactivation",
  "pure_synergy", "order_inversion", "grokking", "loss_spike", "unknown_class"}, severity 1–3, message, event_index |
  None); `detect_events(curve: RICurve, *, alpha=0.01, on=None, off=None, persistence=2) -> list[RIEvent]`
  (on = curve.meta.get("min_effect", 1e-3), off = on/2; hysteresis + persistence; "active" at t iff q ≤ alpha and
  ri ≥ on; becomes inactive iff ri < off for `persistence` consecutive points; step_lo/step_hi bracket the onset
  between the last inactive and first active checkpoints; confidence = 1 − max q over the persistence window);
  `find_transients(events, summary, classification, *, late_frac=0.5) -> list[Transient]`
  (late_activation: activation after train acc ≥ 0.99, or after late_frac of total steps when acc is NaN;
  pure_synergy: order ≥ 3 activation with no proper subset of order ≥ 2 ever active;
  order_inversion: S activates strictly before *all* of its proper subsets that ever activate, with at least one
  activating later; extinction/reactivation: direct; grokking: classification label; unknown_class: label
  "unknown").

---------------------------------------------------------------------------------------------------------
## 6. Survey (`tycho/survey/`)

### 6.1 Catalog (`catalog.py`) — SQLite, one connection per process
Pragmas: WAL, synchronous=NORMAL, busy_timeout=30000, foreign_keys=ON. Schema version in `meta`.
Tables (exact names/columns; add indices as needed):
```
objects(object_id TEXT PRIMARY KEY, task TEXT, region TEXT, sky_version TEXT, spec_json TEXT, seed INTEGER,
        source TEXT, priority REAL, status TEXT, attempts INTEGER DEFAULT 0, worker TEXT, created_at REAL,
        started_at REAL, finished_at REAL, wall_time_s REAL, device TEXT, n_params INTEGER, error TEXT,
        code_version TEXT, random_control INTEGER DEFAULT 0)
        -- status ∈ queued | running | done | failed | interrupted
summaries(object_id TEXT PRIMARY KEY, json TEXT, final_test_acc REAL, final_test_loss REAL,
          final_train_loss REAL, t_test_acc90 REAL, grok_delay REAL, plateau_frac REAL, n_drops INTEGER,
          diverged INTEGER)
lightcurves(object_id TEXT PRIMARY KEY, path TEXT, n_points INTEGER)
classifications(id INTEGER PRIMARY KEY, object_id TEXT, label TEXT, tags_json TEXT, score REAL,
                classifier TEXT, note TEXT, created_at REAL)
followups(object_id TEXT PRIMARY KEY, reason TEXT, priority REAL, status TEXT, requested_by TEXT,
          created_at REAL, started_at REAL, finished_at REAL, worker TEXT, error TEXT)
          -- status ∈ requested | running | done | failed
ri_runs(ri_run_id INTEGER PRIMARY KEY, object_id TEXT, kind TEXT, path TEXT, n_subsets INTEGER,
        n_points INTEGER, max_order INTEGER, alpha REAL, min_effect REAL, created_at REAL)
ri_events(event_id INTEGER PRIMARY KEY, ri_run_id INTEGER, object_id TEXT, subset_json TEXT, order_k INTEGER,
          kind TEXT, step INTEGER, step_lo INTEGER, step_hi INTEGER, ri_before REAL, ri_after REAL, peak_ri REAL,
          confidence REAL, has_precursor INTEGER, matches_gt INTEGER, created_at REAL)
transients(transient_id INTEGER PRIMARY KEY, object_id TEXT, event_id INTEGER, kind TEXT, severity INTEGER,
           message TEXT, created_at REAL, notified INTEGER DEFAULT 0, acknowledged INTEGER DEFAULT 0, note TEXT)
proposals(proposal_id INTEGER PRIMARY KEY, source TEXT, region TEXT, task TEXT, point_json TEXT, seeds INTEGER,
          rationale TEXT, status TEXT, created_at REAL, decided_at REAL, n_enqueued INTEGER)
survey_log(entry_id INTEGER PRIMARY KEY, created_at REAL, author TEXT, kind TEXT, title TEXT, markdown TEXT)
agent_runs(run_id INTEGER PRIMARY KEY, kind TEXT, started_at REAL, finished_at REAL, model TEXT,
           input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER,
           est_cost_usd REAL, n_tool_calls INTEGER, status TEXT, stop_reason TEXT, error TEXT)
agent_tool_calls(id INTEGER PRIMARY KEY, run_id INTEGER, created_at REAL, tool TEXT, input_json TEXT,
                 output_preview TEXT, is_error INTEGER)
heartbeats(worker_id TEXT PRIMARY KEY, pid INTEGER, host TEXT, object_id TEXT, status TEXT, updated_at REAL)
meta(key TEXT PRIMARY KEY, value TEXT)
```
Class `Catalog(path)` with methods (all commit their own transactions):
`add_objects(specs: list[ObjectSpec], *, region: str, source: str, priority: float = 1.0,
random_control_frac: float = 0.0, rng=None) -> int` (INSERT OR IGNORE; returns #new),
`claim_next(worker_id) -> ObjectSpec | None` (BEGIN IMMEDIATE; highest priority then oldest queued or
interrupted), `heartbeat(worker_id, object_id, status)`, `requeue_stale(stale_after_s) -> int` (running objects
and running follow-ups whose worker heartbeat is older than the threshold → back to queue/requested; attempts
counted; after `max_attempts` → failed), `mark_done(result: RunResult, lc_path: str)` (objects + summaries +
lightcurves rows), `mark_failed(object_id, error)`, `mark_interrupted(object_id)`,
`add_classification(object_id, cls, classifier, note="")`, `latest_classification(object_id)`,
`request_followup(object_id, reason, priority=1.0, requested_by="scheduler") -> bool`,
`claim_followup(worker_id) -> str | None`, `finish_followup(object_id, ok: bool, error=None)`,
`add_ri_run(object_id, kind, path, curve: RICurve) -> int`, `add_events(ri_run_id, object_id, events) -> list[int]`,
`add_transients(object_id, transients, event_ids) -> list[int]`, `pending_notifications() -> list[dict]`,
`mark_notified(ids)`, `add_proposal(...) -> int`, `add_log(author, kind, title, markdown) -> int`,
`start_agent_run(kind, model) -> int`, `finish_agent_run(run_id, **fields)`,
`log_tool_call(run_id, tool, input_json, output_preview, is_error)`, `get_spec(object_id) -> ObjectSpec`,
`get_object(object_id) -> dict`, `stats() -> dict`, `query(sql: str, params=()) -> list[dict]`
(read-only: only SELECT/WITH allowed, else ValueError), `backup(dest: Path)` (sqlite backup API),
`set_meta/get_meta`.

Light curves are stored as `np.savez_compressed(paths.lightcurve_path(oid), **lc)`; RI curves via `RICurve.save`.

### 6.2 Sky (`sky.py`) — YAML; see `configs/sky_v0.yaml` for the format
```python
@dataclass
class Knob: name: str; values: list; kind: str = "ordinal"; refine: bool = False; bounds: tuple | None = None
    # kind ∈ categorical | ordinal | log | linear ; name is a task cfg key or "train.<TrainConfig field>"
@dataclass
class Region: name: str; task: str; knobs: list[Knob]; fixed: dict; train: dict; seeds: int = 3;
    ri_policy: str = "followup"; ri: dict = field(default_factory=dict); priority: float = 1.0
class Sky:
    version: str; regions: list[Region]
    @classmethod from_yaml(path) / from_dict(d)
    def region(self, name) -> Region
    def points(self, region) -> list[dict]                     # full grid (knob name -> value)
    def make_spec(self, region, point: dict, seed: int, device: str = "cpu") -> ObjectSpec
    def validate_point(self, region_name: str, point: dict) -> list[str]   # [] if OK
    def neighbors(self, region, point) -> list[dict]           # lattice neighbours (one knob ±1 grid step)
    def coords(self, region, point) -> np.ndarray              # normalized [0,1] per knob (log for log knobs)
```
Refinable numeric knobs accept any value within bounds (default [min(values), max(values)]); others must be grid
values. `ri_policy` ∈ always | followup | never.

### 6.3 Scheduler (`scheduler.py`)
`SchedulerPolicy` (target_queue=200, seeds_initial=None → region.seeds, boundary_refine=True, max_refine_depth=3,
extra_seeds=2, random_control_frac=0.05, followups_per_day=200) and `Scheduler(catalog, sky, policy, rng)` with
`top_up() -> int` (keep ≥ target_queue/2 queued: first unobserved coarse-grid points (deterministic shuffled
order per region, weighted by region priority), then refinement), `boundary_pairs(region) -> list[tuple[dict, dict,
str]]` (neighbouring points, both with ≥ 2 finished seeds, whose majority labels differ, or whose median
log10(t_test_acc90) differ by > 0.5), `refine() -> int` (extra seeds at both endpoints; midpoints on refinable knobs:
geometric for log, arithmetic for linear, rounded for ints; depth ≤ max_refine_depth), `select_followups(n) ->
list[tuple[str, str]]` (object_id, reason): priority unknown class > transient-bearing > boundary members > random
control sample; never twice).

### 6.4 Follow-up, workers, daemon, alerts, release, dashboard
* `followup.py`: `run_followup(catalog, paths, object_id, ri_cfg: RIConfig) -> int` — replay with dense RI around
  the largest train-loss drop (or uniform log points if none), save curve, detect events, find transients, record.
* `worker.py`: `process_object(catalog, paths, spec, cfg, stop_flag) -> None` (run → save light curve → summary →
  classification → if region ri_policy=always: RI events/transients) and `worker_main(worker_id, root,
  stop_event)` (heartbeat thread; prefers follow-ups; respects disk guard).
* `daemon.py`: `run_survey(root: Path, *, hours=None, max_objects=None, n_workers=None) -> dict` — supervisor:
  spawn context, SIGTERM/SIGINT graceful stop (workers checkpoint + mark interrupted), restart dead workers,
  `requeue_stale`, `Scheduler.top_up/refine`, follow-up selection, notifications for new transients, periodic
  `reports/status.json` + `reports/dashboard.html`, nightly SQLite backup (keep last 14), log rotation.
* `alerts.py`: `Notifier(cfg: dict, log_path: Path)` with `send(title, message, priority="default", tags=()) ->
  bool`: always append to log; optional channels "ntfy" ({server, topic}), "webhook" ({url}; Slack-compatible
  `{"text": ...}`), "macos" (osascript; only on darwin). Rate limit (max_per_hour) with digest of suppressed
  messages. Uses urllib only. Network failures never raise.
* `release.py`: `make_release(catalog, paths, tag: str, *, include_lightcurves=True) -> Path` →
  `releases/<tag>/` with parquet + CSV tables (objects ⋈ summaries ⋈ latest classification, ri_events,
  transients), lightcurves tarball, sky yaml, calibration.json (if present), README.md datasheet (schema, units,
  provenance, code version, license CC-BY-4.0), MANIFEST.sha256.
* `dashboard.py`: `write_dashboard(catalog, paths) -> Path` (self-contained HTML: counts, throughput 24h, label
  distribution per task, latest transients, latest RI events, worker heartbeats).

---------------------------------------------------------------------------------------------------------
## 7. Agent (`tycho/agent/`) — Claude via the official `anthropic` Python SDK

* Model `claude-opus-5-5` (configurable). Thinking is always on for this model; do NOT send `thinking` with
  `disabled` or `budget_tokens`; control depth with `output_config={"effort": "medium"}` (nightly) /
  `"high"` (weekly). Requests: `client.beta.messages.create(model=..., max_tokens=16000, system=[{"type":"text",
  "text": SYSTEM, "cache_control": {"type": "ephemeral"}}], tools=TOOL_SPECS, messages=..., output_config=...,
  betas=["server-side-fallback-2026-07-01"], extra_body={"fallbacks": "default"})` (fallbacks configurable).
  Never use forced `tool_choice` (`any`/`tool` returns 400 on this model); default auto.
* Manual agentic loop (justification: hard caps, per-iteration usage accounting, audit of every tool call into the
  catalog, deterministic tests with a scripted fake client). Handle stop reasons: `end_turn` (done), `tool_use`
  (execute all tool_use blocks, return ALL tool_result blocks in one user message; errors → `is_error: True`),
  `pause_turn` (append assistant content and continue), `refusal` (stop, record, no retry), `max_tokens` (stop,
  record). Always append the full `response.content` to history. Cap iterations (`max_iterations`, default 40) and
  estimated cost per run (`max_cost_usd`, default 3.0; prices: input $4/MTok, output $20/MTok, cache read
  $0.20/MTok, cache write $5/MTok).
* `tools.py`: `SurveyTools(catalog, sky, paths, limits)` + `TOOL_SPECS` (JSON Schema, `strict: True`,
  `additionalProperties: False`) + `dispatch(name, input) -> tuple[str, bool]`. Tools: `survey_overview`,
  `list_new_objects`, `get_object`, `list_unknown_queue`, `list_transients`, `list_ri_events`, `query_summaries`
  (group-by aggregation over summaries with whitelisted columns), `fit_power_law` (log–log OLS of a summary metric
  against a numeric knob, with slope CI), `get_sky`, `propose_configs` (validated with `Sky.validate_point`;
  per-run and per-day caps; enqueued with source="agent"), `label_object` (classifier="agent"),
  `request_followup` (capped), `write_survey_log`, `write_report` (weekly only; writes reports/weekly/<iso>.md and a
  survey_log row). All outputs JSON strings truncated to `limits.max_output_chars` (default 20000).
  No shell, no filesystem paths from the model, no network.
* `prompts.py`: stable system prompts (no timestamps) `NIGHTLY_SYSTEM`, `WEEKLY_SYSTEM`; language configurable
  (`agent.language`: "en" | "zh" | "ja").
* `runner.py`: `AgentRunner(catalog, client=None, model, effort, max_iterations, max_cost_usd, fallbacks)` with
  `run(kind, system, user_content, tools) -> AgentRunResult`; lazy `anthropic.Anthropic()`; missing credentials →
  `AgentUnavailable`; `dry_run` writes the would-be first request to `reports/agent/dryrun-<kind>.json`.
* `nightly.py` / `weekly.py`: `run_nightly(root, *, dry_run=False, client=None) -> Path`,
  `run_weekly(root, *, dry_run=False, client=None) -> Path` (context packet → runner → ensure a log/report exists
  → markdown copy under `reports/nightly/` or `reports/weekly/`).

---------------------------------------------------------------------------------------------------------
## 8. CLI (`tycho/cli.py`, console script `tycho`)

Global `--root` (default `$TYCHO_ROOT` or cwd). Subcommands:
`init`, `calibrate [--quick|--full] [--jobs N]`, `run-object --task T [--set k=v ...] [--train k=v ...] --seed S
[--ri]`, `survey enqueue [--sky PATH] [--limit N]`, `survey run [--workers N] [--hours H] [--max-objects N]`,
`survey status`, `followup run [--n N]`, `agent nightly [--dry-run]`, `agent weekly [--dry-run]`,
`release --tag TAG`, `dashboard`, `backup`.

## 9. Ops (`ops/`)
launchd templates (`com.tycho.survey.plist`, `com.tycho.nightly.plist`, `com.tycho.weekly.plist`) with
`__ROOT__`/`__PYTHON__`/`__USER__` placeholders, `install_macmini.sh`, `uninstall_macmini.sh`,
`pmset_recommendations.sh` (prints commands; does not run sudo), `healthcheck.sh`.
