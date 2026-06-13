# TASK 5.5 -- FORMATION-TIME CAUSAL TEST OF Wcc

## Background

Task 5 showed that **post-hoc** Wcc destruction causes only a small, non-significant retention drop, ruling out Wcc as the primary recall substrate.  
Task 5.5 tests whether Wcc must **grow during learning** to build replay-protected memories.  Each condition is a fully independent training run from the same seed.

## Conditions

| Condition | Intervention during training |
|---|---|
| FULL | Standard training, no intervention |
| WCC_FROZEN | Core-core block restored to init values after every STDP step |
| WCC_CLAMPED_ZERO | Core-core block zeroed after every STDP step |
| WCC_NO_STDP | plastic_mask zeros out core-core pairs |

## Table 1: Per-seed retention

| Seed | Condition | Wcc | Wuc | Wuu | S1 | Retention | Retrieval |
|---|---|---|---|---|---|---|---|
| 42 | FULL | 0.0903 | 0.0361 | 0.0178 | 0.0542 | 0.3058 | 0.0438 |
| 42 | WCC_FROZEN | 0.0131 | 0.0332 | 0.0175 | -0.0201 | 0.2725 | 0.0563 |
| 42 | WCC_CLAMPED_ZERO | 0.0000 | 0.0331 | 0.0174 | -0.0331 | 0.2639 | 0.0625 |
| 42 | WCC_NO_STDP | 0.0378 | 0.0341 | 0.0175 | 0.0037 | 0.2815 | 0.0688 |
| 1042 | FULL | 0.0636 | 0.0224 | 0.0085 | 0.0412 | 0.2811 | 0.0375 |
| 1042 | WCC_FROZEN | 0.0023 | 0.0202 | 0.0080 | -0.0179 | 0.2610 | 0.0375 |
| 1042 | WCC_CLAMPED_ZERO | 0.0000 | 0.0188 | 0.0075 | -0.0188 | 0.2467 | 0.0312 |
| 1042 | WCC_NO_STDP | 0.0076 | 0.0203 | 0.0078 | -0.0126 | 0.2619 | 0.0312 |

## Table 2: Condition means

| Condition | Wcc | Wuc | Wuu | Retention | Retrieval |
|---|---|---|---|---|---|
| FULL | 0.0769 | 0.0293 | 0.0131 | 0.2934 | 0.0406 |
| WCC_FROZEN | 0.0077 | 0.0267 | 0.0127 | 0.2668 | 0.0469 |
| WCC_CLAMPED_ZERO | 0.0000 | 0.0260 | 0.0124 | 0.2553 | 0.0469 |
| WCC_NO_STDP | 0.0227 | 0.0272 | 0.0127 | 0.2717 | 0.0500 |

## Table 3: Statistics (FULL vs each intervention)

| Comparison | delta retention | Cohen d | t | p | sig |
|---|---|---|---|---|---|
| FULL vs WCC_FROZEN | +0.0267 | +1.96 | +1.96 | 0.239 | n.s. |
| FULL vs WCC_CLAMPED_ZERO | +0.0382 | +2.54 | +2.54 | 0.141 | n.s. |
| FULL vs WCC_NO_STDP | +0.0217 | +1.38 | +1.38 | 0.308 | n.s. |

## Answers

- **Q1** Can schema form when Wcc growth is prevented? Partially — retention drops
- **Q2** Can replay still protect retention without Wcc formation? Yes — substantial retention preserved
- **Q3** Do Wuc/Wuu compensate? Wuc up=False, Wuu up=False
- **Q4** Retention lost when Wcc never forms: +0.0267
- **Q5** Task 5 negative result (post-hoc) still holds at formation time? YES — small drop only

## Final Verdict: D) Wcc only reflects other processes — manipulation inconclusive

## Figures

- `figures/fig1_retention.png` — Retention by condition
- `figures/fig2_wcc.png` — Final Wcc by condition
- `figures/fig3_weight_decomposition.png` — Weight decomposition (Wcc, Wuc, Wuu)
- `figures/fig4_effect_sizes.png` — Effect sizes (Cohen d)
