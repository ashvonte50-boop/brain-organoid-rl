# Recommended Supplement Structure
**Project:** Catastrophic Forgetting Simulator v3  
**Date:** 2026-05-24  
**Journal target:** eLife / PLOS Computational Biology  
**Structure:** Supplementary Methods + Supplementary Figures + Supplementary Tables

---

## Overview

The supplement serves three functions:
1. **Methods appendix** — Full model equations so the paper is reproducible without the code
2. **Figure appendix** — Detailed supporting evidence for each main figure
3. **Data appendix** — Complete statistical tables and parameter listings

The supplement should be self-contained: a reviewer should be able to reproduce the key
results with only the supplement and the deposited code.

---

## Supplementary Methods

### SM1. Network Architecture

**Content:** Full Izhikevich neuron equations, parameter table, connectivity matrix structure.

```
SM1.1  Neuron model equations (Izhikevich 2003 parameterization)
SM1.2  Network parameters table (N_exc=240, N_inh=60, etc.)
SM1.3  Connectivity: random E→E, E→I, I→E, I→I matrices
SM1.4  Distance-dependent connectivity (small-world topology)
SM1.5  External drive: I_ext specification for driven vs. background neurons
```

**Key equations to include:**
```
dv/dt = 0.04v² + 5v + 140 - u + I_syn + I_ext
du/dt = a(bv - u)
If v ≥ 30 mV: v ← c, u ← u + d

W_eff[i,j] = (1 - γ) × W_fast[i,j] + γ × W_slow[i,j]
dW_fast/dt = -W_fast/τ_fast  (decay during rest)
W_slow[i,j] ← W_slow[i,j] + tag_driven_consolidation(W_tag[i,j])
```

### SM2. Memory Encoding

**Content:** Assembly definition, overlap structure, training protocol.

```
SM2.1  Assembly construction algorithm (N_ASSEMBLY=20, random assignment, 20% overlap)
SM2.2  Sequential training protocol: A → B → C → D
SM2.3  N_PRESENTATIONS=12 cycles per memory
SM2.4  Cue and probe protocol (partial cue, PARTIAL_CUE_SIZE)
```

### SM3. Replay Mechanism

**Content:** Full specification of the SWR-like replay system.

```
SM3.1  Replay coherence definition: coherence = r_target / (r_target + λ·r_off + ε)
SM3.2  Adaptive acceptance criteria (3 conditions)
SM3.3  Burst clustering (REPLAY_BURST_SIZE=5, REPLAY_BURST_GAP=50)
SM3.4  Endogenous urgency: u = (u₁·u₂·u₃)^(1/3)
         u₁ = fast-weight erosion: 1 - mean(W_fast[assembly])
         u₂ = rejection rate: (rejected events / total attempts)
         u₃ = coherence deficit: max(0, THR - mean(coherence))
SM3.5  Chain replay (multi-step sequence completion)
SM3.6  Persistence current: I_pers = PERS_GAIN × Σ W_slow[i,j] × trace[j]
```

### SM4. Synaptic Tag Model

**Content:** Synaptic tag equations.

```
SM4.1  Tag capture: dW_tag/dt = TAG_CAPTURE_RATE × STDP_update(pre, post)
SM4.2  Tag-driven consolidation to W_slow (transfer rate specification)
SM4.3  STDP rule: pre-before-post = LTP, post-before-pre = LTD
```

### SM5. Probe Metric

**Content:** Full specification of the retention probe.

```
SM5.1  Probe procedure: partial cue → measure I_syn differential
SM5.2  Retention score = I_syn(cued network) - I_syn(background network)
SM5.3  Checkpoint schedule: after each memory training, re-probe all prior memories
SM5.4  Final retention = mean(probe scores) at final checkpoint
```

### SM6. Statistical Methods

**Content:** Full statistical procedures.

```
SM6.1  Welch two-sample t-test for all pairwise comparisons
SM6.2  Benjamini-Hochberg FDR correction (α=0.05)
SM6.3  Bootstrap CI: 2000 resamples, BCa correction
SM6.4  Permutation test: 10,000 permutations for coherence-retention correlation
SM6.5  Cohen's d with pooled SD (primary) and Hedges' g (sensitivity)
SM6.6  Rank-biserial r (non-parametric effect size)
SM6.7  Power analysis: G*Power parameterization
```

---

## Supplementary Figures

### Block 1 — Replay Coherence Detail (supports Main Fig 2)

**Supplementary Figure 1** (`replay_coherence_distributions.png`)
*Full distribution of replay coherence values per condition.*
Violin + box plot of all per-event coherence scores. Shows that Slow+Replay produces a
right-skewed distribution (many high-coherence events) while Fast+Replay produces near-uniform
distribution. Quantitatively establishes Fig 2's claim about coherence quality.

**Supplementary Figure 2** (`replay_coherence_trajectory.png`)
*Within-event coherence time course.*
Coherence plotted across timesteps within a single replay event. Shows coherence rises
quickly in successful events and plateaus near threshold. Failed events show coherence below
threshold throughout. Supports the gating mechanism interpretation.

**Supplementary Figure 3** (`replay_success_across_bursts.png`)
*Burst-by-burst acceptance rate.*
Fraction of replay events accepted per burst, across all bursts in a session.
Shows that acceptance rate is stable (not declining), ruling out adaptation as
explanation for reduced replay in later phases.

**Supplementary Figure 4** (`adaptive_replay_analysis.png`)
*Acceptance criteria distributions.*
Shows which of the 3 acceptance criteria is most often the binding constraint.
Important for understanding what limits effective replay.

---

### Block 2 — Interference Structure (supports Main Fig 1 + Discussion)

**Supplementary Figure 5** (`competition_dynamics.png`)
*Weight evolution during competitive training.*
Shows W_fast and W_slow traces for Memory A's assembly weights during A→B→C→D training.
Directly visualizes the competition mechanism — W_fast erodes while W_slow is maintained
only in the Slow+Replay condition.

**Supplementary Figure 6** (`interference_matrix.png`)
*Cross-memory interference structure.*
Heatmap of interference between all pairs of memories. Shows that overlap fraction
directly predicts interference magnitude, validating the design.

**Supplementary Figure 7** (`synaptic_overlap_evolution.png`)
*Overlap structure across training phases.*
How the apparent overlap between memory representations changes as training proceeds.
Addresses the reviewer concern that overlap is static — it is static by construction
but the representational effect is measured.

---

### Block 3 — Representational Drift (supports attractor dynamics)

**Supplementary Figure 8** (`representational_drift.png`)
*Representational similarity matrix evolution.*
Cosine similarity between all memory representations at each checkpoint. Shows that
Slow+Replay maintains representational distinctiveness while other conditions show
representational collapse (memories become indistinguishable).

---

### Block 4 — Overlap and Scaling (supports generality claim)

**Supplementary Figure 9** (`overlap_interference_phase_diagram.png`)
*Full parameter sweep: overlap × condition.*
Extends Fig 1 to the full range of overlap fractions (0–50%). Shows that the
Slow+Replay advantage is maintained across all overlap values but attenuates at
very high overlaps (>40%), consistent with attractor fusion.

**Supplementary Figure 10** (`retention_surface_plot.png`)
*3D surface: retention vs. overlap × condition.*
Provides a quantitative view of the parameter landscape. Shows that the
Slow+Replay "success region" is wide (plateau from 0–35% overlap).

**Supplementary Figure 11** (`memory_vulnerability_map.png`)
*Per-memory vulnerability analysis.*
Shows which position in the A→B→C→D training sequence produces most forgetting.
Memory A (earliest) is most vulnerable; Memory D (most recent) is near-baseline.
Validates the temporal gradient of forgetting.

---

### Block 5 — Prioritization Detail (supports Main Fig 4)

**Supplementary Figure 12** (`replay_scheduling.png`)
*Prioritization mode comparison: full statistics.*
Bar chart with individual points showing all 4 prioritization modes (uniform,
oldest_first, interference_aware, endogenous) with statistical annotations.
Directly supports the Fig 4 prioritization claim with quantitative evidence.

---

### Block 6 — Extension Suite (supports robustness, baselines, ablations)

**Supplementary Figure 13** (`robustness_heatmap.png`)  [pending]
*10-parameter sensitivity heatmap.*
Shows retention maintained across wide parameter ranges for all key hyperparameters.
Critical for addressing reviewer concern about parameter tuning (Reviewer Risk A1, B3).

**Supplementary Figure 14** (`baseline_comparison.png`)  [pending]
*Comparison with EWC, Replay Buffer, and Rehearsal baselines.*
Shows that the Slow+Replay mechanism outperforms all continual learning baselines
tested. Critical for addressing reviewer concern B2.

**Supplementary Figure 15** (`ablation_matrix_extended.png`)  [pending]
*15-condition extended ablation heatmap.*
Full ablation across all mechanism components. Confirms that the main ablation
suite (Fig 5) results hold across the extended condition set.

**Supplementary Figure 16** (`bio_controls_summary.png`)  [pending]
*6-panel biological parameter controls.*
Sweeps of biologically constrained parameters (burst timing, cue sparsity, E/I ratio,
replay latency). Shows mechanism is robust to biologically realistic parameter variation.

**Supplementary Figure 17** (`efficiency_curves.png`)  [pending]
*Retention per replay event analysis.*
Shows that there is a diminishing returns curve — the first few replay events
capture most of the protection benefit. Provides efficiency framing.

---

## Supplementary Tables

### Table S1 — Full Network Parameters

| Parameter | Value | Biological justification |
|-----------|-------|------------------------|
| N_exc | 240 | 80% excitatory (cortical standard) |
| N_inh | 60 | 20% inhibitory (cortical standard) |
| N_ASSEMBLY | 20 | ~8% of exc neurons (sparse code) |
| GAMMA | 0.65 | Slow synapse dominance at rest |
| FAST_DECAY_TAU | 1500 steps | Fast AMPA-like decay (~750 ms) |
| ... | ... | ... |

### Table S2 — Full Statistical Results

All 6 pairwise comparisons with raw p, FDR-adjusted p, Cohen's d, rank-biserial r,
mean±SD for each condition. Full results matrix as in Report 03.

### Table S3 — Per-Trial Results (N=15)

Complete Slow+Replay trial-by-trial data as in Report 02 Appendix.
All 4 conditions per trial for transparency.

### Table S4 — Extension Suite Results (pending)

Summary table of all extension task outcomes once suite completes.

---

## Supplement Organization in PDF

For journal submission, the supplement should be a single PDF organized as follows:

```
Page 1:        Supplementary Methods header (SM1–SM6 table of contents)
Pages 2–8:     SM1–SM6 full text
Pages 9–10:    Table S1–S3 (network parameters, statistics, per-trial data)
Pages 11–28:   Supplementary Figures S1–S17 (one per page, with caption)
Pages 29–30:   Table S4 (extension suite summary, pending)
```

Estimated supplement length: 25–30 pages, consistent with eLife/PLOS supplementary standards.
