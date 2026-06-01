# Mechanistic Narrative Summary
**Project:** Catastrophic Forgetting Simulator v3  
**Date:** 2026-05-24  
**Purpose:** Complete narrative synthesis for manuscript writing  
**Audience:** Methods section and Discussion section authors

---

## The Core Story (4 Sentences)

Sequential learning destroys previously stored memories because new patterns overwrite the synaptic weights
that encoded prior ones — a phenomenon known as catastrophic forgetting. We show that the simultaneous
activation of two complementary mechanisms — slow synaptic consolidation and coherent pattern replay —
provides synergistic protection that neither mechanism achieves alone. The network intrinsically monitors
memory vulnerability and concentrates replay toward memories at greatest risk, implementing a form of
adaptive memory management without external supervision. These results provide a mechanistic framework
for understanding how biological memory systems might use hippocampal replay to selectively protect
memories against competitive interference.

---

## Mechanism Architecture

### Level 1: The Forgetting Problem

The simulator encodes 4 sequential memories (A→B→C→D) in overlapping neuron assemblies.
Because assemblies share ~20% of their neurons, training on memory B activates neurons that
were also used for memory A. STDP (spike-timing-dependent plasticity) strengthens B-specific
connections while weakening A-specific connections through competitive inhibition.

**The result:** After full training (A→B→C→D), fast synaptic weights for memory A have decayed
to near-zero. A partial cue can no longer complete the memory A pattern. This is catastrophic
forgetting: mean retention in the Fast/NoReplay condition = 0.029 ± 0.020.

**Key principle:** Forgetting is driven by synaptic competition, not passive decay. The fast
weights encoding A are overwritten by the STDP updates for B, C, and D.

---

### Level 2: Slow Consolidation (First Component)

The simulator maintains two parallel synapse populations:
- **W_fast**: modified by STDP during every training step (volatile)
- **W_slow**: updated only via synaptic tag consolidation (stable)

Effective synaptic weight: `W_eff = (1-γ)·W_fast + γ·W_slow`, where γ=0.65.

At γ=0.65, slow weights dominate the effective connectivity. But W_slow is updated only
when synaptic tags are captured and consolidated (TAG_CAPTURE_RATE=0.15). Without replay,
most tag consolidation happens during the original training phase and the subsequent rest
period is insufficient to fully transfer fast-learning into slow memory.

**The result:** Slow/NoReplay achieves modest retention (0.072 ± 0.018) — ~2.5× better than
Fast/NoReplay but far below full protection.

**Key principle:** Slow synapses provide a "write-protected" memory substrate, but writing
to them is limited by the tag capture rate. Replay dramatically increases the number of
consolidation events.

---

### Level 3: Replay (Second Component)

During rest periods, the network generates spontaneous reactivations (SWR-like events).
Each replay event begins with a seed cue (partial activation of a memory) and propagates
through the network via attractor dynamics driven by W_eff.

A replay event is **accepted** only if three criteria are met:
1. Minimum pattern completion (sufficient target neurons reactivated)
2. Maximum off-target firing (target-specific, not global activation)
3. Minimum consecutive coherent timesteps (sustained pattern)

This gating prevents noise-driven reactivations from corrupting W_slow.

**The coherence metric:** `coherence = r_target / (r_target + λ·r_off + ε)`, where r_target
is target-assembly firing rate and r_off is off-target firing rate.

During successful replay:
- Synaptic tags are re-captured for the target assembly
- Tag-driven consolidation strengthens W_slow for the target
- The persistence current (I_pers = PERS_GAIN × Σ W_slow × trace) amplifies the replay pattern,
  creating a positive-feedback loop that deepens the attractor basin

**The result:** Fast+Replay achieves negligible protection (0.020 ± 0.019) — essentially the
same as Fast/NoReplay. Replay alone does not work.

**Key principle:** Replay requires a stable substrate to consolidate into. Fast weights are
inherently volatile; replaying into a fast-weights-only system just temporarily activates
the pattern without storing the improvement. The tag→W_slow transfer is the memory-writing step,
and it requires the attractor stability that only W_slow provides.

---

### Level 4: Synergistic Interaction (Why Both Together)

With both slow consolidation (γ=0.65) and coherent replay:
- W_slow provides stable attractors that make replay coherent
- Coherent replay drives tag consolidation into W_slow, deepening the attractors
- Deeper attractors make subsequent replay even more coherent

This is a **positive feedback loop**:
```
W_slow depth → coherent replay → tag consolidation → deeper W_slow → more coherent replay
```

The interaction is **superadditive**: expected additive = 0.063, observed = 0.875 (13.9× larger).
The synergy ratio of 13.9× confirms that slow consolidation and replay are mechanistically
coupled, not independent contributors.

**Quantified result:** Slow+Replay: 0.875 ± 0.091. 30.1-fold improvement over Fast/NoReplay.
t(28) = 34.04, p = 2.50 × 10⁻²⁴, Cohen's d = 12.87.

---

### Level 5: Endogenous Prioritization (Adaptive Replay Management)

The network does not replay memories randomly. An urgency signal u = (u₁·u₂·u₃)^(1/3)
monitors memory vulnerability:

- **u₁ (erosion):** `1 - mean(W_fast[assembly])` — how much have fast weights decayed?
- **u₂ (rejection rate):** fraction of recent replay attempts that failed coherence gating
- **u₃ (coherence deficit):** `max(0, threshold - mean_coherence)` — how far below threshold?

When a memory is vulnerable (high u), it receives proportionally more replay events in the
next burst. When all memories are stable (low u), replay is uniform.

**The mechanism:** The urgency signal rises for Memory A after B/C/D training, and the
network concentrates replay on A's assembly. This is a closed-loop, network-intrinsic
regulatory system.

**Key principle:** The urgency signal is not a hand-designed scheduler — it emerges from
the network's own synaptic state. This is endogenous in the biological sense: the cell
assembly's vulnerability is detectable from within the network.

---

## What the Mechanism Is NOT

To prevent overclaiming, this section clarifies what the model does not demonstrate.

**Not demonstrated:**
1. The mechanism does not explain how replay is triggered in biological systems (the
   trigger is supplied externally via the rest-phase scheduler in the simulator)
2. The mechanism does not address initial memory encoding (assembly formation is pre-specified)
3. The mechanism does not address sleep-wake cycling (all consolidation is during designated
   "rest" periods, not spontaneous oscillatory states)
4. The urgency signal has no identified biological substrate — it is a computational model
   of a plausible cellular detection mechanism
5. The model does not demonstrate generalization beyond the 4-memory sequential protocol
   (though extension Task 9 benchmark tests 8-memory chains)

**These are framed as future work, not failures.** The paper's contribution is the mechanistic
demonstration of sufficiency: a network with these properties is sufficient to prevent
catastrophic forgetting, regardless of how these properties are biologically implemented.

---

## Testable Predictions (For Discussion Section)

1. **Prediction 1 (Coherence-Consolidation link):** Pharmacological disruption of NMDA-dependent
   synaptic tagging during sleep should selectively impair slow-wave-sleep-dependent memory
   consolidation while leaving replay statistics unaffected.

2. **Prediction 2 (Urgency-Driven Replay):** Cells encoding older memories should show higher
   replay participation rates during the interval immediately following new memory acquisition,
   compared to intervals without recent learning.

3. **Prediction 3 (Coherence Threshold):** Successful memory consolidation events should show
   higher within-event coherence (measured as sequential activity correlation) than failed
   events, even when total spike counts are matched.

4. **Prediction 4 (Superadditive Protection):** The combined effect of slow-synapse-targeting
   drugs (e.g., mGluR modulators) and sleep quality enhancement should be superadditive in
   their protection against interference-induced forgetting.

---

## Language Guide for Manuscript

**Use:**
- "is consistent with", "recapitulates key features of", "provides a mechanistic framework"
- "we show that the combination of X and Y produces..."
- "our results suggest that...", "the model predicts..."
- "in the computational framework...", "in the simulator..."

**Avoid:**
- "proves", "demonstrates the biological mechanism of", "explains memory consolidation"
- "the brain uses...", "hippocampus does..."
- "is equivalent to...", "directly models..."

**On effect sizes:** "These effect sizes reflect the magnitude of the computational mechanism
under controlled, noise-minimal conditions. Biological implementations operating with
greater stochasticity and partial instantiation of the mechanism would be expected to
produce smaller but qualitatively similar effects."

---

## Abstract Template

```
[Background] Sequential learning typically results in catastrophic forgetting of 
earlier memories, a fundamental challenge for both biological and artificial 
learning systems. [Method] Using a spiking neural network model with biologically 
constrained parameters, we investigated whether the combination of slow synaptic 
consolidation and coherent replay could prevent forgetting during sequential 
learning of four overlapping memory patterns. [Result] We show that slow 
consolidation and replay act synergistically (13.9× superadditive interaction), 
producing a 30.1-fold improvement in memory retention over the baseline condition 
(mean retention 0.875 ± 0.091 vs. 0.029 ± 0.020; t(28) = 34.0, p = 2.5×10⁻²⁴, 
d = 12.87). Replay efficacy was predicted by pattern coherence (not event frequency), 
and the network autonomously concentrated replay on the most vulnerable memories. 
[Significance] These results provide a mechanistic account of how combined slow 
consolidation and replay could protect memories against competitive interference, 
generating specific empirically testable predictions.
```
