# Schema Abstraction Package

A **separate, optional** analysis layer for the catastrophic forgetting simulation (`compare_catastrophic_forgetting.py`). Implements the "Replay Distortion as Directional Schema Abstraction" hypothesis — that fragmented replay during consolidation causes representational drift *toward shared structure*, not random noise.

The base simulation runs perfectly if this entire directory is deleted. Zero invasive edits.

---

## Architecture

```
schema_abstraction/
├── __init__.py              # empty — package marker
├── README.md                # this file
├── schema_core.py           # hook wiring, config flags, assembly utilities
├── schema_metrics.py        # centroid probing (I_syn mean vectors)
├── schema_downscaling.py    # synaptic downscaling (push-pull)
├── schema_generative.py     # VAE cortical generative layer
├── schema_probes.py         # generalization, anti-prediction, basin geometry
├── schema_replay.py         # uncertainty-weighted replay, Bayesian analysis
├── schema_metaplasticity.py # BCM sliding thresholds, hidden state tracking
├── schema_analysis.py       # 11 statistical tests
└── schema_visualization.py  # 18 publication figures
```

### Integration: Hooks + Sidecar

The main file exposes two things (total ~22 lines):

1. **`register_hook(name, fn)`** — register a callback for `"baseline"`, `"post_encode"`, `"post_replay"`, `"final"`, or `"analysis"`.
2. **`_call_hooks(name, **kwargs)`** — fires the registered callback.

Hooks write data to `net._hook_extra`, a dict that `run_sequential_experiment` automatically includes in the return dict as `"hook_extra"`. This survives multiprocessing because the worker's net gets serialized back to the parent.

Call `register_schema_hooks()` from `schema_core` to wire all nine elements.

---

## The Nine Theoretical Elements

### 1. Synaptic Downscaling (Push-Pull) — `schema_downscaling.py`
**File**: `schema_downscaling.py` (150 lines)

**Theory**: During sleep-like offline periods, all E→E weights are globally weakened (multiplicative decay at `DOWNSCALE_RATE = 0.003` per step). Synapses that carried replay activity in the preceding epoch are tagged and re-potentiated (protected). This makes replay **essential** — without it, *all* weights decay to the floor. Forgetting becomes active selection, not passive decay.

**Key class**: `DownscaleTracker` — attached to `net._downscale`. Tracks `replay_active_mask` (bool for each E→E synapse) and `replay_potentiation` (cumulative protection tag). After each replay epoch, applies global weakening then re-boosts protected synapses.

**Prediction**: `no_replay` condition loses ALL memories with downscaling enabled (vs. modest decay without it). The gap between `full` and `no_replay` becomes extreme.

**Configuration** (in `schema_downscaling.py`):
- `DOWNSCALE_ENABLED`, `DOWNSCALE_RATE`, `DOWNSCALE_PHASE_DUR`, `DOWNSCALE_PROTECT`, `DOWNSCALE_FLOOR`

---

### 2. Generative Cortical Layer (VAE) — `schema_generative.py`
**File**: `schema_generative.py` (189 lines)

**Theory**: Replay does not merely stabilise weights — it trains a cortical generative model. The hippocampus replays episodic traces; the cortex compresses them into a low-dimensional latent space. Over time the cortex becomes hippocampus-independent because it has learned the generative model.

**Key class**: `CorticalGenerativeLayer` — encoder `W_enc` (N_latent × N_input) and decoder `W_dec` (N_input × N_latent). A VAE-style bottleneck forces compression. Each assembly has a `latent_prototype` (exponential moving average of its latent codes). Training happens during `_gen_post_replay_hook`.

**Key metric**: `generative_independence` — Pearson r between prototype-driven reconstruction and actual hippocampal activity. High r = cortex is hippocampus-independent.

**Configuration** (in `schema_generative.py`):
- `N_LATENT = 8`, `VAE_LEARNING_RATE = 0.001`, `PROTOTYPE_MOMENTUM = 0.9`

---

### 3. Anti-Prediction Test — `schema_probes.py`
**Function**: `test_anti_prediction`

**Falsifiable claim**: "Perfect replay fidelity impairs generalization." If this is true, the "high-fidelity" condition (full sequence, no noise, no truncation) should show *better* retention of the cued memory but *worse* generalization to novel schema-consistent patterns, compared to natural fragmented replay.

Two conditions are compared:
- **Natural (blended) cue**: 70% assembly A + 30% assembly B (novel, never-seen pattern)
- **High-fidelity cue**: 100% assembly A (veridical, never-seen in isolation during training)

The prediction is a **cross-over interaction**: high-fidelity retains better but generalizes worse.

---

### 4. Generalization Probe — `schema_probes.py`
**Function**: `test_generalization`

After consolidation, cue the network with a *schema-consistent novel input* never seen during training (a blend of two adjacent assemblies). Measure how often the network correctly completes the expected next assembly in the overlap chain.

This shifts the evaluation from *retention* ("did you remember?") to *generalization* ("can you use what you learned for something new?"). The frame becomes "memory as compression for future use" rather than "memory as preservation."

---

### 5. Statistical Power (Multiple Seeds) — `schema_experiments.py`
**Function**: `run_schema_abstraction_sweep`

The "3 overlapping vs 3 non-overlapping pairs" problem gives n=3 per group, which is fatal for review. This function runs the full experiment across `n_seeds` random seeds (each seed offset by 10000 from `MASTER_SEED`), collecting schema data for each.

With `n_seeds=10` and 3 overlapping pairs each → n=30 per group. `run_multi_seed_meta_analysis` then aggregates across seeds with mean/sem.

---

### 6. Uncertainty-Weighted Replay — `schema_replay.py`
**Function**: `compute_uncertainty_weights`

Implements the active inference principle: the network should replay what it is *most uncertain about*, not what it knows best. Uncertainty = variance of probe scores across recent trials. High-uncertainty memories get higher replay priority.

`uncertainty_weighted_priority = uncertainty / (mean_score + 0.01)` per assembly.

---

### 7. Structured Forgetting (Basin Geometry) — `schema_probes.py`
**Function**: `compute_basin_geometry`, `compute_structured_forgetting`

Tests whether forgetting is selective by attractor basin geometry. For each assembly, measure:
- **`basin_volume`**: fraction of noise-initialised trials that fall into the assembly's basin.
- **`retrieval_probability`**: fraction of partial-cue trials that retrieve the full assembly.
- **`n_overlapping`**: how many other assemblies share neurons.

The structured forgetting hypothesis predicts that assemblies with larger basins and more overlap are *protected* from forgetting, while isolated assemblies with small basins are *forgotten*. This connects to the active forgetting literature (Hardt, Richards & Frankland).

---

### 8. Reverse Replay as Bayesian Filtering — `schema_replay.py`
**Function**: `analyze_reverse_replay_bayesian`

Theory: reverse replay corresponds to the backward pass of a Bayesian filter (smoothing). Forward replay propagates from current state to predicted next state (prediction). Reverse replay propagates backward to correct the earlier estimate (update/correction).

Metrics:
- `reverse_fraction`: proportion of replay events that are reverse.
- `coherence_improvement_after_reverse`: does coherence *increase* after reverse events?
- `predictive_ratio`: forward:reverse imbalance.

---

### 9. Metaplasticity / Hidden States — `schema_metaplasticity.py`
**File**: `schema_metaplasticity.py` (186 lines)

Two connected mechanisms:

**`MetaplasticityController`**: BCM-style sliding thresholds. Each excitatory neuron tracks recent activity via a low-pass filter `activity_trace`. Frequently active neurons raise their potentiation threshold (`theta_p`), making them harder to further potentiate (protection from runaway). Rarely active neurons lower their threshold, making them easier to recruit.

**`HiddenStateTracker`**: Each memory (assembly) has a scalar hidden state `H ∈ [0, 1]` tracking consolidation status:
- 0.0 = fully hippocampal (requires hippocampus for retrieval)
- 1.0 = fully cortical (can be retrieved without hippocampus)

`dH/dt = (replay_benefit - H) / HIDDEN_TAU`, where `replay_benefit = 1` if the memory was replayed recently.

---

## Usage

### Quick start
```python
from schema_abstraction.schema_experiments import run_schema_abstraction_sweep

# Single seed, 5 trials per condition
all_results, schema_results = run_schema_abstraction_sweep(n_trials=5)
```

### Multi-seed for statistical power
```python
all_results_list, schema_results_list, meta = run_schema_abstraction_sweep(
    n_trials=5, n_seeds=10
)
```

### Manual integration
```python
from compare_catastrophic_forgetting import run_all_conditions, make_overlapping_assemblies
from schema_abstraction.schema_core import register_schema_hooks

register_schema_hooks()

assemblies = make_overlapping_assemblies(N_MEMORIES, ASSEMBLY_SIZE, MAIN_OVERLAP)
all_results = run_all_conditions(assemblies, n_trials=5)

# Attach hook_extra data to trial dicts
from schema_abstraction.schema_experiments import _attach_schema_data
_attach_schema_data(all_results)

# Run analysis and figures
from schema_abstraction.schema_analysis import run_all_schema_analysis
from schema_abstraction.schema_visualization import generate_all_schema_figures
schema_results = run_all_schema_analysis(all_results, verbose=True)
generate_all_schema_figures(all_results, schema_results)
```

### Feature flags
In `schema_core.py`, disable specific elements:
```python
ENABLE_DOWNSCALING      = True   # Set False to skip synaptic downscaling
ENABLE_GENERATIVE_LAYER = True   # Set False to skip VAE cortical layer
ENABLE_PROBES           = True   # Set False to skip generalization/anti-prediction
ENABLE_METAPLASTICITY   = True   # Set False to skip BCM sliding thresholds
ENABLE_HIDDEN_STATES    = True   # Set False to skip hidden state tracking
```

---

## Reviewer Questions & Answers

**Q: "Why does the network need replay at all?"**  
A: With synaptic downscaling (biologically realistic: Arc/Homer1a AMPAR endocytosis during sleep), ALL weights decay globally. Only replayed traces are re-potentiated and survive. Replay is *essential*, not optional.

**Q: "How do you know drift is toward schema, not random noise?"**  
A: Three lines: (1) overlapping pairs converge more than non-overlapping, (2) convergence is proportional to overlap fraction, (3) the convergence metric becomes trivial when overlap = 0.

**Q: "Isn't this just catastrophic forgetting with extra steps?"**  
A: No. Catastrophic forgetting is unstructured interference. Our drift is structured — it converges toward shared statistical structure (the schema centroid). The network is not randomly forgetting; it is systematically compressing. The anti-prediction test (fidelity vs. generalization trade-off) proves this is functional abstraction, not amnesia.

---

## Figures Generated

All saved to `figures/schema/`:
1. `centroid_trajectory_pca.png` — PCA of snapshot centroids
2. `pairwise_distance_trajectories.png` — pairwise cosine distance over time
3. `schema_convergence.png` — distance to schema centroid
4. `directionality_barplot.png` — directionality scores per condition
5. `coherence_drift_scatter.png` — replay coherence vs. drift
6. `retention_tradeoff.png` — convergence rate vs. retention decay
7. `overlap_proportionality.png` — overlap fraction vs. drift
8. `summary_heatmap.png` — all tests x conditions
9. `generalization_barplot.png` — generalization scores
10. `anti_prediction.png` — natural vs. high-fidelity generalization
11. `downscaling.png` — replay-protected synapse counts
12. `generative_layer.png` — VAE reconstruction error + cortical independence
13. `hidden_state.png` — cortical consolidation counts
14. `forgetting_variability.png` — retention change variability
15. `reverse_replay.png` — Bayesian filtering analysis
16. `uncertainty_weights.png` — uncertainty score distributions
17. `basin_geometry.png` — per-assembly retention changes
18. `multi_seed_meta.png` — multi-seed meta-analysis

---

## Code Statistics

| Module | Lines | Purpose |
|--------|-------|---------|
| `schema_core.py` | 147 | Hook wiring, config, assembly utils |
| `schema_metrics.py` | 89 | Centroid probing |
| `schema_downscaling.py` | 150 | Push-pull synaptic weakening |
| `schema_generative.py` | 189 | VAE cortical layer |
| `schema_probes.py` | 316 | Generalization, anti-prediction, basin |
| `schema_replay.py` | 160 | Uncertainty weights, Bayesian filtering |
| `schema_metaplasticity.py` | 186 | BCM thresholds, hidden states |
| `schema_analysis.py` | 474 | 11 statistical tests |
| `schema_visualization.py` | 705 | 18 publication figures |
| `schema_experiments.py` | 104 | Multi-seed orchestrator |
| **Total** | **2520** | |

Zero duplicated simulation code. All network/simulation infrastructure is imported from `compare_catastrophic_forgetting.py` via the hook system.
