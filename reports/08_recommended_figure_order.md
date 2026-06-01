# Recommended Final Figure Order
**Project:** Catastrophic Forgetting Simulator v3  
**Date:** 2026-05-24  
**Total main figures:** 6  
**Total supplementary figures:** 17 (12 current + 5 from extension suite)

---

## Main Paper Figures (Final Recommended Order)

```
Fig 1  →  catastrophic_forgetting_curves.png
Fig 2  →  replay_coherence_vs_retention.png
Fig 3  →  attractor_dynamics.png
Fig 4  →  endogenous_prioritization.png
Fig 5  →  ablation_suite.png
Fig 6  →  publication_summary.png
```

### Rationale for this order

**Fig 1 — The Problem and Solution** (`catastrophic_forgetting_curves.png`)
Opens with the central result: a 4-panel learning curve showing all conditions across all
checkpoint probes. The reader immediately sees catastrophic forgetting (red line collapses
to zero) and its rescue (green line holds near 0.87). No prior reading required. This figure
must come first because it defines all subsequent vocabulary: "Slow+Replay", "Fast/NoReplay",
retention score, sequential training protocol.
*Caption hook:* "Sequential learning causes catastrophic forgetting that is rescued by the
combination of replay-driven consolidation and slow synaptic plasticity."

**Fig 2 — The Mechanism (Part 1: What replay quality does)** (`replay_coherence_vs_retention.png`)
Immediately after Fig 1 establishes "Slow+Replay works", Fig 2 asks: why? It shows that
replay coherence — not replay frequency — predicts retention. Two-panel format contrasts
Slow+Replay (tight r > 0.6 correlation) against Fast+Replay (weak scatter). This is the
paper's most novel mechanistic contribution and earns Fig 2 placement.
*Caption hook:* "Replay efficacy is determined by pattern coherence, not event count."

**Fig 3 — The Mechanism (Part 2: Why slow consolidation enables coherent replay)** (`attractor_dynamics.png`)
Fig 2 raises the question: why does slow consolidation produce more coherent replay?
Fig 3 answers: slow weights create deeper attractor basins that survive interference.
Shows phase-space attractor structure before/after replay events. Provides mechanistic
intuition for the two-pathway model.
*Caption hook:* "Slow consolidation creates deep attractor basins that stabilize pattern-selective
reactivation during replay."

**Fig 4 — Intelligent Replay** (`endogenous_prioritization.png`)
Having established that slow+replay works and why, Fig 4 extends to the efficiency question:
can the network optimize its own replay schedule? Shows urgency signals (erosion, rejection
rate, coherence deficit) rising for the most vulnerable memory and directing replay events.
*Caption hook:* "Network-intrinsic vulnerability signals autonomously concentrate replay toward
memories at greatest risk of interference."

**Fig 5 — Mechanism Isolation (Ablations)** (`ablation_suite.png`)
Fig 4 raises the question: is each component truly necessary, or is the system robust to
component removal? Fig 5 shows systematic ablations confirming that coherence gating,
persistence current, synaptic tags, and burst clustering each contribute independently.
*Caption hook:* "Each mechanistic component contributes independently to memory protection,
as shown by targeted ablations."

**Fig 6 — Complete Story** (`publication_summary.png`)
Synthesis figure. Four-panel: (A) condition comparison, (B) coherence-retention scatter,
(C) overlap sensitivity, (D) prioritization mode comparison. Provides the overview a reviewer
reads last to confirm that all claims are simultaneously supported.
*Caption hook:* "Integrated view of memory retention, replay coherence, overlap sensitivity,
and adaptive prioritization across all experimental conditions."

---

## Supplementary Figure Order

### Block 1 — Replay Mechanism Detail (S1–S4)

| Figure | File | Purpose |
|--------|------|---------|
| S1 | `replay_coherence_distributions.png` | Full per-condition coherence distributions |
| S2 | `replay_coherence_trajectory.png` | Within-event coherence time course |
| S3 | `replay_success_across_bursts.png` | Burst-by-burst acceptance rates |
| S4 | `adaptive_replay_analysis.png` | Acceptance criteria distributions |

These directly support the Fig 2 claim. Reviewers will ask for the distributions underlying
the scatter plot. S1–S4 provide that.

### Block 2 — Interference Structure (S5–S7)

| Figure | File | Purpose |
|--------|------|---------|
| S5 | `competition_dynamics.png` | Weight evolution during competitive training |
| S6 | `interference_matrix.png` | Cross-memory interference quantification |
| S7 | `synaptic_overlap_evolution.png` | Overlap structure across training phases |

Support the overlap scaling and interference claims. Reviewers will ask: is the interference
quantified, or just asserted? These figures answer that.

### Block 3 — Representational Analysis (S8)

| Figure | File | Purpose |
|--------|------|---------|
| S8 | `representational_drift.png` | RSM (cosine similarity matrix) evolution |

Shows at the representational level what happens to memories across training.

### Block 4 — Parameter Sensitivity (S9–S11)

| Figure | File | Purpose |
|--------|------|---------|
| S9 | `overlap_interference_phase_diagram.png` | Full overlap sweep characterization |
| S10 | `retention_surface_plot.png` | 3D retention vs. overlap × condition |
| S11 | `memory_vulnerability_map.png` | Which memories are most vulnerable |

Critical for addressing reviewer B3 (parameter tuning concern).
S9 and S10 demonstrate the mechanism survives across overlap regimes.

### Block 5 — Prioritization Detail (S12)

| Figure | File | Purpose |
|--------|------|---------|
| S12 | `replay_scheduling.png` | Prioritization mode comparison detail |

Provides the complete quantitative support for Fig 4.

### Block 6 — Extension Suite Figures (S13–S17)

| Figure | File | Purpose |
|--------|------|---------|
| S13 | `robustness_heatmap.png` (ext) | 10-parameter sensitivity heatmap |
| S14 | `baseline_comparison.png` (ext) | EWC / Buffer / Rehearsal comparison |
| S15 | `ablation_matrix_extended.png` (ext) | 15-condition extended ablations |
| S16 | `bio_controls_summary.png` (ext) | Biological parameter controls |
| S17 | `efficiency_curves.png` (ext) | Retention per replay event |

These figures are pending extension suite completion but are architecturally planned.

---

## Figures to Archive (Do Not Submit)

| Figure | Reason |
|--------|--------|
| `replay_activity_raster.png` | Individual raster — too noisy, not quantitative enough |
| `replay_chain_trajectories.png` | Superseded by coherence figures |
| `replay_preserves_old_memories.png` | Partial redundancy with Fig 1 |
| `forgetting_curves.png` | Subset of catastrophic_forgetting_curves — archived |

---

## Caption Writing Guide

For each main figure, the caption should have exactly 3 parts:

1. **One-sentence headline** (bold, first sentence): The main scientific claim this figure demonstrates.
2. **Panel descriptions** (A–D): Factual description of what is shown in each panel.
3. **Statistical annotation sentence**: "Statistical comparisons: t-test or Welch t-test;
   *p < 0.05, **p < 0.01, ***p < 0.001. N=15 trials per condition; error bars = SEM."

Avoid: restating what the axes show, describing methods in the caption, using "clearly shows"
or "demonstrates that" — let the numbers speak.

---

## Cross-Reference Map

This table maps each figure to the claims it supports and the text section that references it.

| Figure | Primary claim | Text section | Reviewer risk |
|--------|--------------|--------------|--------------|
| Fig 1 | Slow+Replay vs baseline | Results §1 | Low |
| Fig 2 | Coherence-retention link | Results §2 | Low |
| Fig 3 | Attractor mechanism | Results §3 | Moderate (A2) |
| Fig 4 | Endogenous prioritization | Results §4 | Moderate (A3) |
| Fig 5 | Ablation necessity | Results §5 | Low |
| Fig 6 | Integrated overview | Discussion | Low |
| S1–S4 | Replay coherence detail | Supplement §1 | Low |
| S5–S7 | Interference structure | Supplement §2 | Low |
| S13–S14 | Baselines + robustness | Supplement §4 | Critical (B2, B3) |
