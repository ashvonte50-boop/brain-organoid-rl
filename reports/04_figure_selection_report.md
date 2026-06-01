# Figure Selection Report
**Project:** Catastrophic Forgetting Simulator v3  
**Date:** 2026-05-24  
**Total figures available:** 22 (production) + ~15 (extension suite, pending)  
**Recommended main-paper figures:** 5  
**Recommended supplementary figures:** 12  

---

## 1. Scientific Narrative Framework

The paper tells a **mechanistic story in 5 acts**:

> 1. **The Problem** — Sequential learning causes catastrophic forgetting
> 2. **The Solution** — Replay-driven consolidation prevents forgetting
> 3. **Why it works** — Coherent replay + slow consolidation = synergistic protection
> 4. **Intelligent replay** — Endogenous prioritization improves efficiency
> 5. **Robustness** — The mechanism holds across parameters and scales

Every figure should serve exactly one act. Redundant or tangential figures belong in supplementary material.

---

## 2. Primary Figures (Main Paper)

### Figure 1 — `catastrophic_forgetting_curves.png`
**Act:** 1+2 (Problem and Solution combined)  
**Scientific purpose:** Shows the full 4-condition comparison across all checkpoint probes  
**Key claim:** Slow+Replay provides massive, consistent protection; other conditions fail  
**Novelty:** Direct demonstration of the synergistic mechanism  
**Mechanistic importance:** 5/5 — this IS the paper's central result  
**Reviewer value:** 5/5 — first thing reviewers want to see  
**Redundancy:** None — no other figure shows the full 4-condition learning curve  

*Content:* Retention matrix showing all 4 memories across all 4 checkpoints.
Slow+Replay (green) dramatically outperforms Fast/NoReplay (red), Fast+Replay (orange),
and Slow/NoReplay (blue). Error bars (SEM across N=15 trials).

---

### Figure 2 — `replay_coherence_vs_retention.png`
**Act:** 3 (Why it works — mechanism)  
**Scientific purpose:** Shows that replay coherence predicts retention strength  
**Key claim:** Not all replay is equal — coherent replay drives consolidation  
**Novelty:** High — quantitative coherence-retention link is novel  
**Mechanistic importance:** 5/5 — the coherence gating mechanism is the most novel contribution  
**Reviewer value:** 5/5 — directly addresses "what makes replay effective"  
**Redundancy:** None — no other figure shows this correlation  

*Content:* Scatter plot: per-event peak coherence (x-axis) vs assembly weight gain
post-replay (y-axis). Pearson r expected > 0.6, p < 0.001. Two panels:
Slow+Replay (r strong) vs Fast+Replay (r weaker, more scatter).

---

### Figure 3 — `attractor_dynamics.png`
**Act:** 3 (Why slow consolidation matters)  
**Scientific purpose:** Shows W_slow attractor structure enabling pattern completion  
**Key claim:** Slow weights create deeper attractor basins that survive inter-memory interference  
**Novelty:** 4/5 — attractor dynamics visualized in spiking network  
**Mechanistic importance:** 4/5 — explains why slow+replay > fast+replay  
**Reviewer value:** 4/5 — provides intuition for the two-pathway mechanism  
**Redundancy:** Low — representational_drift and rsm_matrix cover similar ground but are less clear  

*Content:* Phase-space or activity-space visualization of attractor occupancy before/after
replay events. Slow+Replay shows deeper, more stable basins.

---

### Figure 4 — `endogenous_prioritization.png`
**Act:** 4 (Intelligent replay)  
**Scientific purpose:** Demonstrates urgency-based prioritization of vulnerable memories  
**Key claim:** The network autonomously identifies and preferentially replays memories at risk  
**Novelty:** 5/5 — adaptive endogenous prioritization is a core novel contribution  
**Mechanistic importance:** 4/5 — shows closed-loop replay regulation  
**Reviewer value:** 5/5 — directly addresses "why not just random replay?"  
**Redundancy:** Low — replay_scheduling covers related ground but less mechanistically  

*Content:* Urgency signals over time: erosion (u₁), rejection rate (u₂), coherence deficit (u₃).
Shows that Memory A's urgency rises after B/C/D training, and replay events concentrate there.

---

### Figure 5 — `publication_summary.png`
**Act:** 5 (Complete story)  
**Scientific purpose:** Summary of all four main claims in one figure  
**Key claim:** Synthesizes retention, coherence, overlap scaling, and prioritization  
**Novelty:** 3/5 — summary figure  
**Mechanistic importance:** 5/5 — conveys the complete system  
**Reviewer value:** 5/5 — provides the overview every reviewer reads first  
**Redundancy:** Some redundancy with individual panels, but that's the purpose of a summary  

*Content:* 4-panel: (A) Condition comparison, (B) Coherence-retention scatter,
(C) Overlap sensitivity, (D) Prioritization mode comparison.

---

## 3. Secondary Figures (Main Paper, optional)

These figures are scientifically strong but could move to supplementary without weakening the narrative.

| Figure | Purpose | Recommendation |
|--------|---------|----------------|
| `ablation_suite.png` | Shows mechanism isolation | Keep in main if journal allows 6-7 figures |
| `replay_protection_comparison.png` | Bar chart summary of conditions | Move to supplement — covered by Fig 1 |
| `overlap_vs_forgetting.png` | Shows overlap-interference gradient | Keep as Fig 4 alternate if endogenous moved out |
| `synaptic_tag_evolution.png` | Shows tag dynamics over learning | Secondary — supports mechanistic claims |

---

## 4. Supplementary Figures

These figures provide rigorous technical support but would overload the main text.

| Figure | File | Scientific role |
|--------|------|-----------------|
| S1 | `replay_coherence_distributions.png` | Full distribution of coherence values per condition |
| S2 | `replay_coherence_trajectory.png` | Within-event coherence time course |
| S3 | `replay_success_across_bursts.png` | Burst-by-burst acceptance rate |
| S4 | `competition_dynamics.png` | Competitive interference weight evolution |
| S5 | `adaptive_replay_analysis.png` | Acceptance criteria distributions |
| S6 | `interference_matrix.png` | Cross-memory interference structure |
| S7 | `synaptic_overlap_evolution.png` | Overlap structure across training |
| S8 | `representational_drift.png` | RSM evolution (cosine similarity matrix) |
| S9 | `retention_surface_plot.png` | 3D retention vs overlap × condition |
| S10 | `memory_vulnerability_map.png` | Which memories are most vulnerable |
| S11 | `overlap_interference_phase_diagram.png` | Full sweep at high overlaps |
| S12 | `replay_scheduling.png` | Prioritization mode comparison detail |
| (from extension) | `robustness_heatmap.png` | Parameter sensitivity — reviewer requirement |
| (from extension) | `baseline_comparison.png` | EWC/Buffer/Rehearsal comparison |
| (from extension) | `ablation_matrix_extended.png` | Fine-grained mechanism ablations |
| (from extension) | `bio_controls_summary.png` | Biological plausibility controls |

---

## 5. Archive/Remove

These figures are low-value or redundant given the above selection:

| Figure | Reason to archive |
|--------|-------------------|
| `replay_activity_raster.png` | Individual raster — too noisy, not quantitative |
| `replay_chain_trajectories.png` | Chain replay paths — superseded by coherence figures |
| `replay_preserves_old_memories.png` | Partially redundant with retention curves |
| `forgetting_curves.png` | Subset of catastrophic_forgetting_curves |

---

## 6. Recommended Figure Order

### Main Paper Figures:

**Fig. 1** — `catastrophic_forgetting_curves.png`  
*Caption hook:* "Sequential learning causes catastrophic forgetting, which is rescued by  
replay-driven slow consolidation."  
*Flow:* Establishes the problem and solution simultaneously.

**Fig. 2** — `replay_coherence_vs_retention.png`  
*Caption hook:* "Replay efficacy is predicted by coherence of pattern reactivation."  
*Flow:* Asks the mechanistic question — not all replay is equal.

**Fig. 3** — `attractor_dynamics.png`  
*Caption hook:* "Slow consolidation creates stable attractor basins that support coherent replay."  
*Flow:* Answers WHY slow+replay is synergistic.

**Fig. 4** — `endogenous_prioritization.png`  
*Caption hook:* "Network-intrinsic urgency signals autonomously direct replay toward at-risk memories."  
*Flow:* Extends the mechanism — intelligent replay, not random.

**Fig. 5** — `ablation_suite.png`  
*Caption hook:* "Each mechanistic component contributes independently to memory protection."  
*Flow:* Demonstrates necessity of each component.

**Fig. 6** — `publication_summary.png`  
*Caption hook:* "Consolidated overview of memory retention, replay coherence, overlap sensitivity, and adaptive prioritization."  
*Flow:* Synthesis and closing.

---

## 7. Narrative Coherence Check

Each figure must answer exactly one question raised by the previous figure:

| Figure | Question raised | Answer in next figure |
|--------|-----------------|----------------------|
| Fig 1: Catastrophic forgetting curves | *Why does Slow+Replay work?* | Fig 2: Coherence predicts retention |
| Fig 2: Coherence-retention link | *Why does slow consolidation amplify replay?* | Fig 3: Attractor basins |
| Fig 3: Attractor dynamics | *Can the network optimize its own replay?* | Fig 4: Endogenous prioritization |
| Fig 4: Endogenous prioritization | *Is each component truly necessary?* | Fig 5: Ablation suite |
| Fig 5: Ablations | *What is the complete mechanistic picture?* | Fig 6: Publication summary |

This produces a self-contained, reviewer-friendly narrative without redundancy.

---

## 8. Figure Quality Assessment

| Figure | Resolution | Axes labelled | Statistical annotation | Color-blind safe | Publication-ready? |
|--------|-----------|---------------|----------------------|-----------------|-------------------|
| catastrophic_forgetting_curves | 150 dpi | Yes | SEM bars | Mostly | Yes (minor tweaks) |
| replay_coherence_vs_retention | 150 dpi | Yes | Pearson r | Yes | Yes |
| attractor_dynamics | 150 dpi | Yes | None needed | Yes | Yes |
| endogenous_prioritization | 150 dpi | Yes | Yes | Yes | Yes |
| ablation_suite | 150 dpi | Yes | SEM bars | Yes | Yes |
| publication_summary | 150 dpi | Yes | Partial | Yes | Yes |

All figures saved as PNG (150 dpi) and PDF via `_save_fig()`.
PDF versions recommended for journal submission (vector-quality scalable).
