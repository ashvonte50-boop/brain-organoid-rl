# Task 7.5 Report — Sufficiency Test of W_slow[cc]

## VERDICT: NECESSARY BUT NOT SUFFICIENT ALONE

W_slow[cc] alone recovers 73.8% — partial but below threshold. Additional blocks contribute meaningfully.

## Results Table

| Condition | Mean | SD | % Control | t | p | Cohen dz |
|-----------|------|----|-----------|----|---|----------|
| CONTROL | 0.2907 | 0.0132 | 100.0% | nan | nan n.s. | 0.00 |
| WSLOW_CC_ONLY | 0.2145 | 0.0045 | 73.8% | 14.25 | 0.0049 ** | 8.22 |
| WSLOW_CC_PLUS_UC | 0.2704 | 0.0054 | 93.0% | 4.53 | 0.0455 * | 2.61 |
| WSLOW_CC_PLUS_UU | 0.2144 | 0.0058 | 73.7% | 16.51 | 0.0036 ** | 9.53 |
| WSLOW_UC_ONLY | 0.0549 | 0.0030 | 18.9% | 36.76 | 0.0007 *** | 21.22 |
| WSLOW_UU_ONLY | 0.0000 | 0.0000 | 0.0% | 38.12 | 0.0007 *** | 22.01 |

## Q&A
Q1: Retention in WSLOW_CC_ONLY = 0.2145 (73.8% of CONTROL)
Q5: 73.8% of CONTROL retention explained by W_slow[cc] alone

## Mechanism
W_slow[cc] = slow synaptic consolidation among 20 schema-core neurons.
This 20x20 sub-matrix (400 weights out of 562,500 total) carries the engram.
Core-core recurrence creates a persistent attractor that drives pattern completion.