# Replay-Gated Cascade Consolidation (RGCC)
### Harmonic Replay Scheduling Produces the Serial Position Effect from Synaptic First Principles

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: BICA 2026 Submission](https://img.shields.io/badge/status-BICA%202026-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository contains the full experimental pipeline, analysis code, and publication materials for:

> **"Replay-Gated Cascade Consolidation Produces the Serial Position Effect from Synaptic First Principles"**  
> Ashwajit Warwatkar — BICA 2026 Submission

**Core finding:** A biologically plausible spiking neural network with two-timescale synaptic consolidation (W_fast → recency; W_slow → primacy) and harmonic-series replay scheduling spontaneously reproduces the Glanzer & Cunitz (1966) serial position effect — **without any special architectural assumptions**. Earlier-encoded memories receive more replay events (harmonic scheduling), accumulating disproportionately more W_slow consolidation, yielding primacy. W_fast decays rapidly, yielding recency. The serial position curve emerges from the interaction of these two mechanisms.

---

## The RGCC Model

### Two-Timescale Synaptic Architecture

Each synapse maintains two independent weight components:

```
W_eff(t) = (1 − γ) · W_fast(t) + γ · W_slow(t)
```

| Component | Role | Time constant | Mechanism |
|-----------|------|--------------|-----------|
| **W_fast** | Recency buffer | τ_fast = 1,500 ms | Potentiated by every STDP event; decays rapidly |
| **W_slow** | Primacy / long-term engram | τ_slow = 3,000–4,000 ms | Updated only via synaptic-tag capture during replay |

At γ = 0.65, W_slow dominates the network dynamics and determines which memories survive delayed recall.

### Harmonic Replay Scheduling

When N memories have been encoded, the expected number of replay events received by memory at serial position k (0-indexed) is:

```
E[replay(k)] = R · (H(N) − H(k))
```

where H(n) = Σ 1/i is the n-th harmonic number and R is total replay events. Earlier-encoded memories (lower k) receive more replay, accumulating more W_slow consolidation.

### Position-to-Consolidation Law

Empirically derived from M1 (20 seeds):

```
E[retention(k)] = 0.247 + 0.074 · (H(N) − H(k))     R² = 0.828
```

This linear relationship between expected retention and harmonic replay count is the quantitative backbone of the serial position model.

### Network Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_NEURONS` | 1,000 | Total neurons (750 exc / 250 inh) |
| `CORE_SIZE` | 20 | Schema core neurons shared across all memories |
| `N_MEMORIES` | 4 | Memories per experiment |
| `GAMMA` | 0.65 | W_slow mixing coefficient |
| `TAU_SLOW` | 3,000–4,000 ms | Slow weight consolidation time constant |
| `W_MAX` | 1.5 | Synaptic weight ceiling |
| `A_PLUS / A_MINUS` | 0.006 / 0.003 | STDP LTP/LTD amplitudes |
| `REPLAY_COHERENCE_THR` | 0.50 | Coherence gate threshold for replay |
| `TAG_CAPTURE_RATE` | 0.15 | Synaptic tag → W_slow transfer rate |

### Schema Assembly Architecture

```
Schema Core:   neurons [0–19]       — shared across ALL N memories
Memory M0:     [0–19] ∪ [20–39]    (core + 20 unique neurons)
Memory M1:     [0–19] ∪ [40–59]
Memory M2:     [0–19] ∪ [60–79]
Memory M3:     [0–19] ∪ [80–99]
```

---

## Key Results

### Primary Finding: Serial Position Effect (30 seeds, E2)

| Serial Position | Immediate Recall (W_fast) | Delayed Recall (W_slow) |
|-----------------|--------------------------|------------------------|
| M0 (first) | Low | **Highest** (primacy) |
| M1 | Medium | High |
| M2 | Medium | Low |
| M3 (last) | **Highest** (recency) | Lowest |

Causal intervention (blocking replay) eliminates the primacy gradient: t = −19.3, p < 0.001 (30 seeds).

### W_slow Block Structure (M3, 10 seeds)

W_slow organises into a structured block matrix reflecting the assembly architecture:

| Block | Mean ± SEM | Interpretation |
|-------|-----------|----------------|
| W_slow[cc] — core→core | **0.610 ± 0.005** | Schema attractor; primary engram |
| W_slow[uc] — unique→core | 0.126 ± 0.003 | Memory-specific consolidation |
| W_slow[uu] — unique→unique | 0.041 ± 0.001 | Near-zero; unique neurons not self-consolidated |

The 15:1 ratio (W_slow[cc] / W_slow[uu]) confirms that the schema core acts as the true long-term memory engram.

### Catastrophic Forgetting Prevention (10 seeds, T2)

Four-condition design (Fast/Slow × NoReplay/Replay):

| Condition | Retention | vs. Baseline |
|-----------|-----------|-------------|
| Fast + No Replay | 0.029 ± 0.020 | — |
| Fast + Replay | 0.104 ± 0.031 | 3.6× |
| Slow + No Replay | 0.165 ± 0.045 | 5.7× |
| **Slow + Replay** | **0.1802 ± 0.0046** | **6.2×; t = 13.5, p = 0.005** |

Cohen's d = 25.78 (10-seed replication). Interaction is **13.9× superadditive**.

### Null Model Control (M4)

Shuffled-replay control with identical replay count but destroyed temporal ordering:  
t = −175.3 (real vs. shuffled), confirming replay structure — not mere activity — drives consolidation.

### E3 Learned Schema

Schema emerges without pre-specified core assembly when networks are trained with RGCC: schema crystallisation index (SCI) increases monotonically with replay count (r = 0.97).

---

## Experiment Series

### Core Mechanistic Chain

| Series | Description | Seeds | Key Result |
|--------|-------------|-------|-----------|
| **E1** | Adequate-seed boost confirmation | 15 | Replay necessary for primacy gradient |
| **E2** | 30-seed causal replay intervention | 30 | t = −19.3, p < 0.001; eliminates primacy without replay |
| **E3** | Learned schema (no pre-spec core) | 10 | Schema self-organises from RGCC dynamics |

### Mechanistic Decomposition (M1–M5)

| Task | Description | Key Result |
|------|-------------|-----------|
| **M1** | 20-seed baseline, per-trial retention | Position-to-consolidation law, R² = 0.828 |
| **M2** | Attractor diagnostics | W_slow[cc] confirmed as primary basin |
| **M3** | Dual metrics (immediate + delayed probe) | Primacy/recency dissociation replicated |
| **M4** | Null model (shuffled replay) | t = −175.3; structure, not activity, matters |
| **M5** | Encoding-order manipulation | Harmonic schedule confirmed causal |

### MAJOR Experiments

| Task | Description | Key Result |
|------|-------------|-----------|
| **MAJOR-1** | Harmonic scheduling decoupling | Replay count vs. order separated; count drives primacy |
| **MAJOR-2** | Immediate vs. delayed behavioral readout | Glanzer & Cunitz double-dissociation confirmed |
| **MAJOR-3** | W_slow as memory substrate | W_slow[cc]=0.610; destroying W_slow → 7.5% retention (p<0.001) |
| **MAJOR-4** | Position-to-consolidation narrative | E[retention]=0.247+0.074·(H(N)−H(k)), R²=0.828 |
| **MAJOR-5** | Benna–Fusi cascade comparison | RGCC outperforms B–F on schema metrics |

### MOD Experiments (Robustness & Extensions)

| Task | Description | Key Result |
|------|-------------|-----------|
| **MOD-1** | Network scaling (N=500–4000) | Effect holds at all scales |
| **MOD-2** | Consolidation law regression | Power-law fit: R²=0.94 |
| **MOD-3** | Learned schema, 15 seeds | SCI r=0.97 with replay count |
| **MOD-4** | Inhibitory STDP (iSTDP) | RGCC robust to inhibitory plasticity |
| **MOD-5** | Bio-parameter sweep | Primacy gradient stable across τ_slow=2000–6000 ms |

### Q Series (Parameter Sensitivity)

| Task | Description |
|------|-------------|
| **Q1–Q4** | Core-size, N_memories, γ, τ_slow sensitivity analysis |
| **Q5** | Dose-response: replay count vs. primacy strength |
| **Q6** | Full parameter sweep (36-point grid) |

### Ablation Suite (10 seeds, 10 figures)

Full 4-file ablation pipeline with PDF report:
- W_slow block ablation, replay-count ablation, coherence-gate ablation, γ sensitivity
- STDP amplitude, tag-capture rate, slow-weight decay, overlap-exclusion cue
- Output: `ablation_results/ablation_report.pdf`

---

## Repository Structure

```
brain-organoid-rl/
│
├── compare_catastrophic_forgetting.py   # Master simulation engine (v3)
│                                        # Izhikevich SNN, STDP, W_slow, replay scheduler
│
├── schema_abstraction/                  # Schema analysis package
│   ├── schema_experiments.py            # Assembly construction (CORE_SIZE=20)
│   ├── schema_metrics.py                # Core metric implementations
│   ├── schema_novel_metrics.py          # Schema Crystallization Index (SCI)
│   ├── schema_generative.py             # Generative model probes
│   ├── schema_metaplasticity.py         # Metaplasticity analysis
│   ├── schema_downscaling.py            # Homeostatic downscaling
│   └── schema_probes.py                 # Probe utilities
│
├── task2_worker.py / task2_analyze.py   # T2: catastrophic forgetting (4-condition)
├── task10_worker.py / task10_analyze.py # T10: W_slow substrate confirmation
├── task11_figures.py                    # Publication figure generation (Fig1 + Fig2)
├── task{3..9}_worker.py                 # Worker scripts for T3–T9
│
├── e1_boost_adequate_seed.py            # E1: boost confirmation
├── e2_task105_30seeds.py                # E2: 30-seed causal intervention
├── e3_learned_schema.py                 # E3: self-organised schema
│
├── m1_task105_20seeds.py                # M1: 20-seed mechanistic baseline
├── m2_attractor_diagnostics.py          # M2: attractor diagnostics
├── m3_dual_metrics.py                   # M3: dual retention metrics
├── m4_null_model.py                     # M4: shuffled-replay null model
├── m5_encoding_order.py                 # M5: encoding-order manipulation
│
├── major1_scheduling_test.py            # MAJOR-1: harmonic scheduling decoupling
├── major2_behavioral_readout.py         # MAJOR-2: immediate vs. delayed probe
├── major3_wslow_panel.py                # MAJOR-3: W_slow substrate
├── major4_narrative_reframe.py          # MAJOR-4: position-to-consolidation law
├── major5_benna_fusi.py                 # MAJOR-5: Benna–Fusi comparison
│
├── q5_dose_response.py                  # Q5: replay count dose-response
├── q6_param_sweep.py                    # Q6: full parameter sweep
├── ablation_pipeline.py                 # Ablation runner (10 seeds)
├── ablation_figures.py                  # Ablation figure generation
├── ablation_report.py                   # PDF report generation
│
├── e1_results/ e2_results/ e3_results/  # E-series outputs (CSVs, PNGs)
├── m1_results/ … m5_results/            # M-series outputs
├── major1_results/ … major5_results/    # MAJOR-series outputs
├── mod_results/                         # MOD-series outputs
├── ablation_results/                    # Ablation pipeline outputs (PDFs, PNGs, CSVs)
├── serial_position_experiment/          # Flagship serial position experiment
│
├── important_paper_materials_and_results/
│   ├── FINAL_PAPER_corrected_1.docx     # BICA2026 submission (latest)
│   ├── FINAL_PAPER_corrected.docx       # Prior revision
│   ├── RGCC_Revision_and_Corrections.docx
│   ├── MAJOR3_wslow_substrate_FINAL.png # Publication figure (600 DPI)
│   ├── MAJOR3_wslow_substrate_FINAL.pdf
│   ├── MAJOR3_wslow_substrate_FINAL.svg
│   ├── figures_selected/                # Top 15 ranked figures + captions
│   │   ├── 00_ALL_FIGURE_CAPTIONS.txt   # Publication-ready captions
│   │   └── Figure{1,2}_*_FIXED.{png,pdf}
│   ├── figures_all_sorted_by_relevance/ # All 223 figures ranked by score
│   ├── results_extracted/
│   │   └── 02_key_numbers_for_paper.json  # 82 extracted statistics
│   ├── paper_sections/
│   │   ├── 04_complete_paper_outline.txt  # Full 5-section outline
│   │   └── 05_RESULTS_CHEATSHEET.txt      # All key numbers
│   └── methods_extracted/               # Extracted parameter summaries
│
└── figures/
    └── paper/
        ├── serial_position_flagship.png  # Flagship serial position figure
        └── serial_position_flagship.pdf
```

---

## Retention Metric

```python
retention = probe_memory(net, assembly)["isyn_score"]
```

Normalised post-synaptic drive when the memory assembly is cued via partial input (4 neurons). Measured at encoding (baseline) and after all N memories are learned + rest period (final). Used throughout all experiment series as the primary readout.

---

## Serial Position Dissociation

| Probe timing | Dominant weight | Serial position curve |
|--------------|----------------|----------------------|
| Immediate (0 ms rest) | W_fast | **Recency** (M3 > M2 > M1 > M0) |
| Delayed (2,000 ms rest) | W_slow | **Primacy** (M0 > M1 > M2 > M3) |

This double dissociation mirrors the Glanzer & Cunitz (1966) finding: selective interference with the recency component (filled delay, in our case W_fast decay) leaves the primacy gradient intact, and vice versa.

---

## Running Experiments

### Requirements

```bash
pip install -r requirements.txt
# python >= 3.10, numpy, scipy, matplotlib, torch
```

### Quick Start

```bash
# Catastrophic forgetting 4-condition experiment (10 seeds)
python run_task2.py

# E2: 30-seed causal replay intervention
python run_e_experiments.py

# M1-M5: mechanistic decomposition
python run_m_tasks.py

# Full MAJOR series
python run_major_fixes.py

# Ablation suite (10 seeds, ~2 hours)
python run_ablation_study.py

# Serial position flagship figure
python serial_position_experiment/code/run_experiment.py
```

### Regenerate Publication Figures

```bash
# Figure 1 (mechanistic architecture) + Figure 2 (experimental validation)
python task11_figures.py

# Fixed versions (resolved overlaps, tight layout)
python important_paper_materials_and_results/task11_figures_FIXED.py

# MAJOR-3 W_slow substrate figure (Nature-style, 600 DPI)
python important_paper_materials_and_results/major3_wslow_figure.py
```

### Ablation Report

```bash
python ablation_pipeline.py   # runs 10 seeds × 10 ablation conditions
python ablation_figures.py    # generates 10 figures
python ablation_report.py     # compiles PDF report
```

---

## Scientific Claims (Validated)

1. **Serial position effect emerges from RGCC without special assumptions** — harmonic replay scheduling + two-timescale synapses are sufficient (E2: t = −19.3, p < 0.001, 30 seeds)
2. **W_slow is the primary memory engram** — destroying W_slow[cc] reduces retention to 7.5% of control (p < 0.001; MAJOR-3)
3. **Position-to-consolidation law** — E[retention(k)] = 0.247 + 0.074·(H(N)−H(k)), R² = 0.828 (M1, 20 seeds)
4. **Replay structure, not activity, drives consolidation** — null model t = −175.3 (M4)
5. **Slow consolidation + replay is superadditive** — 13.9× interaction, d = 25.78 (T2, 10 seeds)
6. **Schema self-organises** — SCI increases monotonically with replay count, r = 0.97 (E3, MOD-3)
7. **Effect is scale-invariant** — holds across N = 500–4,000 neurons (MOD-1)

---

## Publication

**BICA 2026 Submission**  
Title: *Replay-Gated Cascade Consolidation Produces the Serial Position Effect from Synaptic First Principles*  
Author: Ashwajit Warwatkar  
Manuscript: `important_paper_materials_and_results/FINAL_PAPER_corrected_1.docx`

```bibtex
@inproceedings{warwatkar2026rgcc,
  title     = {Replay-Gated Cascade Consolidation Produces the Serial Position Effect
               from Synaptic First Principles},
  author    = {Warwatkar, Ashwajit},
  booktitle = {Proceedings of the Biologically Inspired Cognitive Architectures (BICA) 2026},
  year      = {2026},
  note      = {Code: https://github.com/ashvonte50-boop/brain-organoid-rl},
}
```

---

## Author

**Ashwajit Warwatkar**  
Email: ashvonte50@gmail.com  
GitHub: [@ashvonte50-boop](https://github.com/ashvonte50-boop)

---

## References

- Glanzer, M., & Cunitz, A. R. (1966). Two storage mechanisms in free recall. *Journal of Verbal Learning and Verbal Behavior*, 5(4), 351–360.
- Benna, M. K., & Fusi, S. (2016). Computational principles of synaptic memory consolidation. *Nature Neuroscience*, 19(12), 1697–1706.
- Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on Neural Networks*, 14(6), 1569–1572.
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102(3), 419–457.
- Tse, D., Langston, R. F., Kakeyama, M., et al. (2007). Schemas and memory consolidation. *Science*, 316(5821), 76–82.
- Wilson, M. A., & McNaughton, B. L. (1994). Reactivation of hippocampal ensemble memories during sleep. *Science*, 265(5172), 676–679.
