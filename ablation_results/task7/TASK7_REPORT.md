# Task 7 Report — True Memory Substrate

## Hypothesis
Memory is stored in **W_slow** (the slow synaptic consolidation matrix),
not in the fast weight matrix W.
Task 6 interventions only zeroed W, leaving W_slow intact — explaining
why no single-block destruction reproduced the replay-removal effect.

## Key Experimental Results

| Condition | Mean Ret | % Control | t | p | Cohen d_z |
|-----------|----------|-----------|---|---|-----------|
| CONTROL | 0.2907 | 100.0% | nan | nan | 0.00 |
| DESTROY_W_ALL | 0.2694 | 92.7% | 5.39 | 0.0327* | 3.11 |
| WSLOW_ONLY | 0.2688 | 92.5% | 5.58 | 0.0306* | 3.22 |
| W_ONLY | 0.0217 | 7.5% | 63.98 | 0.0002*** | 36.94 |
| DESTROY_WSLOW_ALL | 0.0218 | 7.5% | 63.43 | 0.0002*** | 36.62 |
| DESTROY_BOTH | 0.0000 | 0.0% | 38.12 | 0.0007*** | 22.01 |
| DESTROY_WSLOW_CC | 0.0765 | 26.3% | 64.88 | 0.0002*** | 37.46 |
| DESTROY_WSLOW_UC | 0.2367 | 81.4% | 32.59 | 0.0009*** | 18.82 |
| DESTROY_WSLOW_UU | 0.2913 | 100.2% | -0.47 | 0.6833 | -0.27 |
| DESTROY_WSLOW_NON_CC | 0.2365 | 81.3% | 64.08 | 0.0002*** | 36.99 |

## Interpretation

- **DESTROY_W_ALL ≈ CONTROL** → The fast weight matrix W is NOT the memory substrate.
  Destroying all fast excitatory weights barely reduces retention because W has
  already partially decayed toward baseline during post-training rest.

- **DESTROY_WSLOW_ALL ≈ No-replay baseline** → W_slow IS the memory substrate.
  Zeroing the slow weight matrix collapses retention to the same level as
  training without replay — confirming that replay builds consolidation in W_slow.

- **WSLOW_ONLY ≈ CONTROL** → Even with W = 0, W_slow alone sustains retention.
  The γ=0.65 mixing coefficient means W_slow contributes 65% of W_eff;
  after rest, W ≈ W_baseline so W_slow is the dominant effective weight.

## Mechanistic Summary

```
Training:      STDP builds W patterns
Replay (rest): Re-fires assemblies → STDP on W → W_slow follows W upward
               (tau_slow=3000; asymmetric ratchet)
Long rest:     W decays → W_baseline (tau_fast=1500)
               W_slow persists (tau_very_slow=200,000)
Probe:         W_eff = 0.35·W + 0.65·W_slow ≈ 0.65·W_slow
               → Memory signal comes entirely from W_slow

Without replay: W_slow never consolidates the full assembly
               → Poor retention at probe
```

## W_slow Block Analysis

Sub-task B identifies WHICH W_slow block is critical.
See Fig 2 for block-specific retention values.

## Figures
- Fig 1: All conditions retention bar chart
- Fig 2: W_slow block decomposition
- Fig 3: W vs W_slow dissociation
- Fig 4: Centroid geometry (PCA)
- Fig 5: Assembly overlap matrix
- Fig 6: W_slow norm vs retention scatter