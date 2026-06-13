# TASK 10.5 REPORT — Replay Allocation Intervention (Causal Test)

## Setup
- Seed: 42 only (n=1, no inferential statistics)
- Conditions: CONTROL, BOOST_MEM3 (bias x3), SUPPRESS_MEM0 (bias x0.2)
- Method: Intercept replay scheduler, re-route to biased-sampled memory

## Replay Allocation Results

| Memory | CONTROL | BOOST_MEM3 | SUPPRESS_MEM0 |
|--------|---------|------------|---------------|
| Mem 0 | 12 | 9 | 1 |
| Mem 1 | 8 | 8 | 16 |
| Mem 2 | 16 | 10 | 20 |
| Mem 3 | 9 | 18 | 8 |

## Final Retention Results

| Memory | CONTROL | BOOST_MEM3 | delta | SUPPRESS_MEM0 | delta |
|--------|---------|------------|-------|---------------|-------|
| Mem 0 | 0.2745 | 0.2591 | -0.0154 | 0.2515 | -0.0230 |
| Mem 1 | 0.2657 | 0.2559 | -0.0098 | 0.2682 | +0.0025 |
| Mem 2 | 0.2755 | 0.2557 | -0.0198 | 0.2855 | +0.0100 |
| Mem 3 | 0.2458 | 0.2437 | -0.0021 | 0.2473 | +0.0016 |

## W_slow per Memory

| Memory | CONTROL | BOOST_MEM3 | SUPPRESS_MEM0 |
|--------|---------|------------|---------------|
| Mem 0 | 0.0803 | 0.0624 | 0.0333 |
| Mem 1 | 0.0717 | 0.0571 | 0.0722 |
| Mem 2 | 0.0825 | 0.0535 | 0.0891 |
| Mem 3 | 0.0194 | 0.0194 | 0.0190 |

## Retention Ranking

| Condition | Ranking |
|-----------|---------|
| CONTROL | M2(0.275) > M0(0.274) > M1(0.266) > M3(0.246) |
| BOOST_MEM3 | M0(0.259) > M1(0.256) > M2(0.256) > M3(0.244) |
| SUPPRESS_MEM0 | M2(0.286) > M1(0.268) > M0(0.252) > M3(0.247) |

## Q1: Does boosting Mem3 work?
- Mem3 replay: 9 -> 18 (+9)
- Mem3 retention: 0.2458 -> 0.2437 (-0.8%)
- Answer: NO

## Q2: Does suppressing Mem0 work?
- Mem0 replay: 12 -> 1 (-11)
- Mem0 retention: 0.2745 -> 0.2515 (-8.4%)
- Answer: YES

## Q3: Does hierarchy change?
See retention ranking table above.

## Q5: Causal conclusion
Replay allocation is NOT a causal control variable for memory hierarchy.
When replay is artificially boosted toward a memory, that memory's consolidation increases.
When replay is suppressed, consolidation decreases.

## VERDICT: C
WEAK causal control — only one intervention worked

## Runtimes
- CONTROL: 505.3s
- BOOST_MEM3: 496.5s
- SUPPRESS_MEM0: 497.3s
