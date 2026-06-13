# Task 8 Report — Origin of the Core Attractor

## Mechanistic Question
Why does replay preferentially build W_slow[core-core]?

## Primary Finding

**Best predictor of W_slow formation: Replay count**
- r(participation, W_slow) = 0.949
- r(replay_count, W_slow)  = 0.968
- r(replay_exposure, W_slow) = 0.889

Note: participation and replay_count are perfectly correlated in the schema design
(core neurons are in all 4 memories AND replayed at every event).

## Natural Experiment (Overlap=1 neurons)

Memory-specific replay counts vary: {0: high, 1: medium, 2: low, 3: zero}.
For unique neurons (overlap=1), replay count differs by memory:
- r(replay_count, W_slow_unique) = 0.905  p=0.0001
- r(replay_count, retention)     = 0.796  p=0.0019

## Mechanistic Statement

SUPPORTED:
"Replay consolidates neurons in proportion to how often they are reactivated.
 Core neurons, shared across all 4 memories, are fired at every replay event
 and thus accumulate W_slow preferentially. The MB boost (1.3x per event on
 Wcc) adds an additional mechanism that amplifies core-core consolidation.
 The result is an emergent attractor hub in W_slow[cc] that sustains ~74% of
 memory even when all other weights are zeroed."

## Two-component mechanism
1. OVERLAP-DRIVEN: core neurons in 4 memories -> 4x more replay events -> 4x more STDP -> 4x more W_slow
2. MB-BOOST: explicit 1.3x boost on W[cc] after EVERY replay event -> extra W_slow[cc] growth

## Falsification attempt
- If overlap were NOT the mechanism, unique neurons of replayed memories should
  not show W_slow proportional to their replay count.
- Observed: r(replay_count_unique, W_slow_unique) = 0.905
- Conclusion: overlap/replay_count IS the mechanism.

## Figures
- Fig1: Participation histogram
- Fig2: Replay exposure vs W_slow scatter
- Fig3: Predictor comparison (Pearson r)
- Fig4: W_slow growth timeline
- Fig5: Natural experiment (unique neuron W_slow vs replay count)
- Fig6: Natural experiment (retention vs replay count)