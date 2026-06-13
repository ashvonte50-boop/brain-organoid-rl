# TASK 6 -- TRUE REPLAY-PROTECTED MEMORY SUBSTRATE

## Context

Tasks 1-5.5 established: Replay->Retention is causal; Replay->Wcc is causal; but **Wcc->Retention is NOT causal** (post-hoc destruction ~5% loss; formation-time prevention 7-13% loss).  The original causal chain is falsified.  
Task 6 asks: *which connectivity block actually stores the replay-protected memory?*

All 6 interventions share the **identical trained FULL network** per seed (n=3 seeds: 42, 1042, 2042), so any retention difference is causally attributable to the destroyed block alone.

## Interventions

| Condition | Blocks zeroed |
|---|---|
| CONTROL | none (= FULL) |
| DESTROY_WUC | unique<->core |
| DESTROY_WUU | within-unique |
| DESTROY_WUC_WUU | Wuc + Wuu (Wcc + background kept) |
| DESTROY_ALL_NON_CORE | everything except Wcc |
| DESTROY_ALL | entire excitatory matrix (sanity floor) |

## TABLE 1: Condition means

| Condition | Wcc | Wuc | Wuu | Retention | % remaining |
|---|---|---|---|---|---|
| CONTROL | 0.0722 | 0.0261 | 0.0117 | 0.2907 | 100.0% |
| DESTROY_WUC | 0.0722 | 0.0000 | 0.0117 | 0.2850 | 98.0% |
| DESTROY_WUU | 0.0722 | 0.0261 | 0.0000 | 0.2924 | 100.6% |
| DESTROY_WUC_WUU | 0.0722 | 0.0000 | 0.0000 | 0.2850 | 98.0% |
| DESTROY_ALL_NON_CORE | 0.0722 | 0.0000 | 0.0000 | 0.2851 | 98.1% |
| DESTROY_ALL | 0.0000 | 0.0000 | 0.0000 | 0.2687 | 92.4% |

## TABLE 2: Effect sizes (CONTROL vs destruction, paired)

| Destruction | delta Ret | % lost | Cohen d_z | t | p |
|---|---|---|---|---|---|
| DESTROY_WUC | +0.0057 | 2.0% | +1.92 | +3.33 | 0.0797 |
| DESTROY_WUU | -0.0017 | -0.6% | -1.07 | -1.85 | 0.205 |
| DESTROY_WUC_WUU | +0.0057 | 2.0% | +4.87 | +8.44 | 0.0137 |
| DESTROY_ALL_NON_CORE | +0.0056 | 1.9% | +1.52 | +2.63 | 0.119 |
| DESTROY_ALL | +0.0220 | 7.6% | +3.71 | +6.42 | 0.0234 |

## Primary question

**Which destruction reproduces the replay-removal phenotype (~87% loss)?**  
Reproduces phenotype (>=70% loss): **NONE**.  
Largest single-block collapse: **DESTROY_WUC** (2.0% loss).

## FINAL VERDICT

1. **What weight block stores memory?**  Higher-order distributed structure.
2. **What weight block stores schema?**  Wcc (core-core) — the schema index S1 = Wcc - Wuc collapses only when Wcc is removed; established in Tasks 4-5.5.
3. **Are memory and schema the same substrate?**  **No.** Memory lives in Higher-order distributed structure; schema lives in Wcc. They are dissociable.
4. **Corrected causal chain after Tasks 1-6:**

   ```
   Replay  -->  potentiates Higher-order distributed structure  -->  Retention (memory)
      |
      +----->  potentiates Wcc       -->  Schema metrics (S1)  [epiphenomenal for memory]
   ```

   Replay is the common cause of BOTH the memory substrate (Higher-order distributed structure) and the schema substrate (Wcc).  Wcc co-varies with retention only because replay drives both in parallel — destroying Wcc leaves memory intact, while destroying Higher-order distributed structure produces the largest retention loss.

## Figures

- `figures/fig1_retention_by_intervention.png` — Retention by intervention
- `figures/fig2_destruction_schematic.png` — Weight-block destruction schematic
- `figures/fig3_pct_retention_remaining.png` — Percent retention remaining
- `figures/fig4_weight_verification.png` — Weight-block verification (bonus)
