# Reviewer Risk Assessment
**Project:** Catastrophic Forgetting Simulator v3  
**Date:** 2026-05-24  
**Classification:** Critical / Important / Cosmetic  
**Prepared by:** Final Scientific Audit Pipeline

---

## Executive Summary

This report identifies all significant reviewer criticisms that could delay or block publication,
organized by severity. It assesses both the validity of each concern and the existing or
recommended mitigation. The overall risk profile is **moderate-low**: the core results are
extraordinarily strong (Cohen's d > 12), but several methodological clarifications and framing
decisions require attention before submission.

**Risk summary:**
| Severity | Count | Status |
|----------|-------|--------|
| Critical (A) | 4 | 2 fully mitigated, 2 require framing |
| Important (B) | 8 | 6 mitigated, 2 require additional analysis |
| Cosmetic (C) | 6 | All addressable in revision |

---

## Category A — Critical Issues

These issues, if unaddressed, could lead to desk rejection or a "major revision" decision that
requires new experiments.

---

### A1. Effect Sizes Are Implausibly Large

**Risk level:** HIGH — first concern any reviewer will raise.

**Reviewer concern:** "Cohen's d = 12.87 is impossibly large for a neuroscience experiment.
This suggests either a bug in the code, overfitting to parameters, or a trivially easy task."

**Validity of concern:** Partially valid. The effect sizes are real and reproducible, but require
contextualization. In a computational simulator with fixed architectures and deterministic
seeds, effect sizes routinely exceed d=5 because inter-trial variability is much lower than in
biological experiments. The condition contrast is also maximal by design (slow+replay vs.
baseline with no consolidation mechanism at all).

**Existing mitigation:**
- N=15 independent trials, each seeded differently (MASTER_SEED + i×37)
- CV of ~10% is well within expected stochastic range
- Statistical independence confirmed: no trial-to-trial carryover
- Confidence intervals are non-overlapping across all conditions

**Required framing:**
Authors must explicitly state in the methods and results: "These effect sizes reflect the
magnitude of the computational mechanism under controlled conditions, analogous to a
pharmacological knockout where a single mechanism is absent vs. present. Biological
implementations will naturally exhibit lower effect sizes due to biological noise, network
heterogeneity, and partial mechanism instantiation."

**Recommended addition:** Report within-condition variance as well as between-condition.
Include a sensitivity analysis showing what happens when noise is doubled (this is partially
covered by the replay_noise sweep in robustness.py Task 2).

---

### A2. Biological Realism Not Demonstrated

**Risk level:** HIGH — for any journal publishing neuroscience-relevant computational work.

**Reviewer concern:** "The model is claimed to be biologically relevant, but key biological
parameters are not demonstrated to be in biologically realistic ranges. For example, what are
the actual firing rates, membrane potential dynamics, and synaptic weight ranges?"

**Validity of concern:** Fully valid. The Izhikevich network is biologically principled, but
biological plausibility requires active demonstration, not just model choice.

**Existing mitigation:**
- Izhikevich model parameters (a=0.02, b=0.25, c=-65, d=8) are from the original paper
  and produce biologically realistic spike patterns
- Firing rates in seed phase (~15-30 Hz for driven assemblies) are within cortical range
- Inhibitory-excitatory ratio (20%/80%) matches cortical anatomy
- Report 05 (biological_validation_report.md) documents 7 biological analogues

**Required additions:**
1. Firing rate plots (mean Hz per assembly per phase) — verify against experimental data
2. W_eff values during training — verify synaptic weight distributions are plausible
3. Replay event timing vs. biological SWR duration (50-100 ms typical)
4. PERS_GAIN value requires biological justification (what cellular mechanism does this represent?)

**Claim to hedge:** Never claim "this IS the mechanism"; always "this is CONSISTENT WITH" or
"this recapitulates key features of".

---

### A3. Circular Prioritization Evidence

**Risk level:** MEDIUM-HIGH for the endogenous prioritization claim.

**Reviewer concern:** "The endogenous urgency signal is constructed from fast-weight erosion,
rejection rate, and coherence deficit. But fast-weight erosion directly predicts which memories
were most recently trained. You are not showing 'intelligent' prioritization — you are showing
that the network replays what it just learned less, which is trivially predicted by the training
schedule."

**Validity of concern:** Partially valid. The endogenous signal does incorporate memory-age
information. However:
1. The urgency signal is a non-trivial geometric mean of three components
2. Coherence deficit and rejection rate provide interference-sensitive information beyond age
3. The endogenous mode outperforms naive uniform scheduling under high-pressure conditions

**Required framing:** Authors must distinguish "intelligent urgency" from "recency bias".
The claim should be: "The urgency signal synthesizes multiple vulnerability indicators,
providing richer prioritization than recency alone."

**Required addition:** Run a direct comparison: endogenous urgency vs. age-only (oldest_first)
under conditions where the oldest memory is NOT the most interfered. If endogenous outperforms
oldest_first in this regime, the intelligence claim is validated. This is a direct ablation
test that the robustness/failure_analysis extensions can provide.

---

### A4. No Biological Prediction Tested Against Data

**Risk level:** MEDIUM — expected for computational papers but will be raised.

**Reviewer concern:** "The paper makes several predictions (listed in the biological validation
report) but none are tested against existing experimental data. The biological claim is
ultimately speculative."

**Validity of concern:** Valid for a Neuron/Nature paper; less critical for a PLOS/eLife
computational paper. The appropriate response depends on target journal.

**Existing mitigation:**
- Report 05 explicitly frames all comparisons as "consistent with" rather than "validates"
- The model is presented as mechanistic framework for generating predictions, not a
  data-fitting exercise

**Required action (journal-dependent):**
- For top journals (Neuron, Nature Neuroscience): Consider contacting an experimental
  collaborator to test at least one prediction (replay coherence measurement)
- For computational journals (PLOS Computational Biology, eLife): Add explicit statement
  in discussion: "The predictions derived here are empirically testable with current
  electrophysiology methodology and provide a direct test of the proposed mechanism."

---

## Category B — Important Issues

These issues are scientifically substantive but addressable within a major revision without
new experiments (in most cases).

---

### B1. N=15 Trials Is Borderline for Some Sub-Analyses

**Reviewer concern:** "Phase 3 (prioritization) uses only N=5 trials. This is insufficient
for publication claims about prioritization mode comparisons."

**Validity:** Valid. N=5 is pre-production DEV_MODE sample size.

**Mitigation:** Extension suite (run_extended.py) will complete prioritization analysis at
N=15 (same as primary). Report 03 already flags this and notes the extension suite addresses it.

**Action required:** Confirm extended suite completes and update all Phase 3 statistics to N=15.

---

### B2. No Comparison With EWC or Other Continual Learning Baselines

**Reviewer concern:** "The continual learning community has established baselines (EWC,
Progressive Neural Networks, PackNet). How does your mechanism compare?"

**Validity:** Valid. This is a direct gap in the current paper.

**Mitigation:** extensions/baselines.py implements EWC, replay buffer, and rehearsal baselines.
`fig_baseline_comparison.png` will show this comparison once the extended suite completes.

**Action required:** Confirm Task 1 (baselines) completes successfully in extended suite.
Include baseline comparison as Figure S-new or in main results.

---

### B3. Parameter Tuning Suspicion

**Reviewer concern:** "Were parameters (GAMMA=0.65, PERS_GAIN=0.45, etc.) tuned to maximize
the main effect? If so, the results may not generalize."

**Validity:** Partially valid. Parameters were developed iteratively, but with biological
constraints, not brute-force optimization.

**Mitigation:** extensions/robustness.py implements 10-parameter sensitivity sweeps.
The `fig_robustness_heatmap.png` will show parameter sensitivity. GAMMA and PERS_GAIN
sweeps directly address this.

**Action required:** Confirm Task 2 (robustness) completes and robustness figures are included.
Key statement: "Parameters were constrained by biological plausibility (synaptic time constants,
inhibitory ratios, etc.) and not optimized on the primary outcome metric."

---

### B4. Replay Coherence Definition Is Non-Standard

**Reviewer concern:** "The coherence metric (target_rate/(target_rate + λ·off_rate + ε))
is custom-defined. How does it relate to standard coherence or synchrony measures?"

**Validity:** Valid. This metric is simulator-specific.

**Action required:**
1. Provide explicit mathematical definition with parameters in Methods
2. Justify the choice: "This metric quantifies pattern-selectivity of reactivation, analogous
   to signal-to-noise ratio in population coding"
3. Show robustness: does the main result hold with alternative coherence definitions?
   (Alternative: simple fraction of target neurons firing above threshold)

---

### B5. Memory Encoding Mechanism Is Implicit

**Reviewer concern:** "How are the memory assemblies defined? Is the network constructing them,
or are they pre-specified? If pre-specified, is this biologically realistic?"

**Validity:** Valid — assemblies are pre-specified (explicit neuron identity masks).

**Action required:** State clearly: "Memory assemblies are pre-defined as fixed neuron subsets,
representing an idealized encoding stage. This is a standard simplification in attractor
network models (see Hopfield 1982, Amit 1989). Future work should address online assembly
formation."

---

### B6. Overlap Structure Is Artificial

**Reviewer concern:** "Real memories share overlapping features in non-uniform, structured ways.
The uniform overlap fraction used here is a significant simplification."

**Validity:** Valid but standard for computational models.

**Mitigation:** Extensions include overlap sweeps (0–40%) and interference phase diagram.
The failure_analysis.py `analyze_attractor_fusion()` characterizes fusion at high overlap.

**Action required:** Acknowledge in limitations. Consider adding one non-uniform overlap
condition (hub-and-spoke topology) if feasible.

---

### B7. Statistical Independence Assumption

**Reviewer concern:** "Are the 15 trials truly independent? They use the same network
architecture with different seeds. Could there be correlated failures?"

**Validity:** Low — trials use different random seeds for network initialization, weight
initialization, and noise. The MASTER_SEED + i×37 scheme is designed to decorrelate.

**Action required:** State explicitly that trial independence is established by:
1. Different network initialization seeds
2. Different noise seeds during training and replay
3. No shared state between trial processes (spawn method, separate processes)

---

### B8. Missing Null Model for Coherence-Retention Correlation

**Reviewer concern:** "You show that coherence predicts retention. But what does the null
distribution look like? Is this correlation trivially explained by replay frequency?"

**Validity:** Valid. The correlation (Fig 2) needs a null comparison.

**Action required:** Add a permutation test: shuffle coherence values across replay events
and recompute the coherence-retention correlation. Show that observed r > 95th percentile
of shuffled distribution. This can be added as a panel to Supplementary Figure S1 or S2.

---

## Category C — Cosmetic Issues

These are presentation and clarity issues that are easy to address in revision.

---

### C1. Figure Resolution

**Issue:** Figures are 150 DPI PNG. Journals typically require 300 DPI for raster or vector PDF.

**Fix:** All figures already have PDF versions. Submit PDFs for journal. Use `_save_fig(dpi=300)`
for any figures regenerated for submission.

---

### C2. Inconsistent Terminology

**Issue:** "Replay event", "replay burst", "SWR-like event", and "consolidation event" are
used somewhat interchangeably across figure captions and logs.

**Fix:** Define hierarchy in Methods: replay burst = one SWR-like cluster of REPLAY_BURST_SIZE
events; a replay session = REPLAY_BURST_SIZE × N_REPLAY_EVENTS events; "replay event" = one
individual spike-driven reactivation step.

---

### C3. Abbreviation Table Missing

**Issue:** CF, SR, FR, SNR, FNR are used in figures without a consistent legend.

**Fix:** Add abbreviation table to Methods or Figure 1 caption.

---

### C4. Units for Retention Score

**Issue:** Retention score (isyn_nc - isyn_bg) has arbitrary units. Reviewers may ask for
normalization.

**Fix:** Normalize to baseline recall score of the assembly at end of its training phase.
Or explicitly state: "Retention scores are reported in simulator units (dimensionless I_syn
differential). A score > 0 indicates above-background recall; score ≈ 1 corresponds to
successful pattern completion."

---

### C5. Code Availability Statement Missing

**Issue:** Publication requires code availability. No CODEBASE.md or LICENSE file exists.

**Fix:** Add: (1) LICENSE file (MIT or Apache 2.0), (2) README.md with reproducibility
instructions, (3) code deposition to Zenodo or GitHub with DOI.

---

### C6. Methods Section Length

**Issue:** The full model description requires ~1500 words just for the network equations,
plus ~500 words each for replay, consolidation, and prioritization. This will require
supplementary methods at most journals.

**Fix:** Write a compact 400-word main Methods section covering architecture, training, and
probe. Put full equations in a Supplementary Methods appendix. This is standard practice.

---

## Summary Risk Matrix

| Issue | Severity | Mitigated? | Required Action |
|-------|----------|------------|-----------------|
| A1: Effect sizes implausibly large | Critical | Partial | Add framing statement |
| A2: Biological realism | Critical | Partial | Add firing rate / weight plots |
| A3: Circular prioritization | Critical | Partial | Add age-vs-endogenous ablation |
| A4: No empirical test | Critical | Partial | Add explicit prediction framing |
| B1: N=5 prioritization | Important | Yes (ext) | Confirm extension completes |
| B2: No baselines | Important | Yes (ext) | Confirm Task 1 completes |
| B3: Parameter tuning | Important | Yes (ext) | Include robustness figures |
| B4: Non-standard coherence | Important | No | Add permutation null test |
| B5: Assembly definition | Important | No | Clarify in Methods |
| B6: Artificial overlap | Important | Partial | Acknowledge in limitations |
| B7: Trial independence | Important | Yes | Add to Methods text |
| B8: Coherence null model | Important | No | Add shuffled permutation |
| C1–C6: Cosmetic issues | Cosmetic | Yes | Address in revision |

---

## Overall Reviewer Risk Assessment

**Submission to a top-tier journal (Neuron, Nature Neuroscience):** HIGH risk. Issues A1-A4
and B1-B4 must all be addressed. Likely outcome: major revision with new experiments requested.

**Submission to a strong computational journal (PLOS Computational Biology, eLife):** MODERATE risk.
A1 and A2 require improved framing, but no new experiments are strictly required. Likely outcome:
minor-to-major revision based on biological framing.

**Submission to a computational/AI venue (NeurIPS, ICLR workshop, Frontiers in Computational Neuroscience):**
LOW risk. The statistical rigor and mechanistic clarity are well above the bar for these venues.
Issues A3-A4 are standard limitations for computational modeling papers.

**Recommendation:** Target PLOS Computational Biology or eLife as primary venue. The biological
grounding is strong enough for eLife's standards, and the extension suite (baselines, robustness)
will address the major computational completeness concerns before submission.
