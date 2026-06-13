# TASK 4.5 -- VERIFY REPLAY-STDP DORMANCY CLAIM

Seeds: [42, 1042, 2042]
Total replay events: 135
COH_THR: 0.5
STDP_GATE_ENABLED: True
STDP_GATE_BIAS: 0.5

## Table 1: Per-seed summary

| Seed | Events | STDP Calls | Pot Updates | Dep Updates | Coh Steps>THR |
|------|--------|------------|-------------|-------------|---------------|
| 42 | 45 | 1 | 0 | 0 | 0 |
| 1042 | 45 | 3 | 3 | 3 | 0 |
| 2042 | 45 | 2 | 0 | 0 | 0 |
| **TOTAL** | **135** | **6** | **3** | **3** | **0** |

## Table 2: Coherence threshold exceedance

| Threshold | Count (mean_coh) | Fraction | Count (peak_coh) | Fraction |
|-----------|------------------|----------|------------------|----------|
| 0.50 | 0 | 0.0000 | 0 | 0.0000 |
| 0.45 | 0 | 0.0000 | 0 | 0.0000 |
| 0.40 | 0 | 0.0000 | 1 | 0.0074 |
| 0.35 | 0 | 0.0000 | 2 | 0.0148 |
| 0.30 | 0 | 0.0000 | 2 | 0.0148 |
| 0.25 | 1 | 0.0074 | 11 | 0.0815 |
| 0.20 | 1 | 0.0074 | 23 | 0.1704 |
| 0.15 | 3 | 0.0222 | 36 | 0.2667 |
| 0.10 | 13 | 0.0963 | 66 | 0.4889 |
| 0.05 | 48 | 0.3556 | 66 | 0.4889 |

## Table 3: Coherence statistics

- mean_coherence: 0.0369 +/- 0.0481 [0.0000, 0.2596]
- peak_coherence: 0.0922 +/- 0.1043 [0.0000, 0.4360]
- smooth_coh_last: 0.0126 +/- 0.0285
- n_coh_steps/event: 0.00 +/- 0.00

## Verdict

**B) REPLAY STDP RARE BUT NON-ZERO**

6 STDP calls, 3 pot, 3 dep.
Task 4 conclusion PARTIALLY INCORRECT.
