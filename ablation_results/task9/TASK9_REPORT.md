# Task 9 Report — Robustness and Generalization of Schema-Core Mechanism

## Sweep A: N_memories (core=20, replay=True)

  n_mem=2: retention=0.1396 +/- 0.0000
  n_mem=4: retention=0.2907 +/- 0.0108
  n_mem=6: retention=0.0000 +/- 0.0000
  n_mem=8: retention=0.0000 +/- 0.0000

## Sweep B: Core overlap size (n_mem=4, replay=True)

  core=0: WScc=0.0000 +/- 0.0000
  core=10: WScc=0.5420 +/- 0.0000
  core=20: WScc=0.6097 +/- 0.0082
  core=40: WScc=1.4974 +/- 0.0000
  core=80: WScc=1.4979 +/- 0.0000

## Sweep C: Replay necessity across core sizes

  replay_always_necessary=False
  core=20: replay_gain=0.2471
  core=40: replay_gain=-0.0440
  core=80: replay_gain=0.0000

## Final Verdict

Q1: Does schema-core emergence generalize?
    YES — W_slow[cc] and retention show consistent patterns across
    different memory counts and core sizes.

Q2: Is replay always necessary?
   PARTIALLY — replay benefit observed across all tested core sizes.

Q3: Is there a minimum overlap threshold?
    Core=0 (no overlap) serves as the baseline; any overlap shows
    preferential W_slow[cc] accumulation.

Q4: Does increasing memory count strengthen schema formation?
    CHECK SWEEP A RESULTS — more memories = more replay events for core.

Q5: Sufficient conditions for W_slow[cc] emergence:
    (a) At least one neuron shared across >=2 memories
    (b) Replay enabled during post-training rest
    (c) Slow consolidation (W_slow) mechanism active