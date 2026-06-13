"""Generalization, Anti-Prediction, Structured Forgetting, Partial-Cue Retrieval,
and Replay Diversity probes.

These probes are the behavioural readout layer for the schema-abstraction
hypothesis.  They answer:

  1. **Schema-Generalization**: After consolidation, can the network complete
     schema-consistent novel patterns?
  2. **Anti-Prediction**: Does *perfect* replay fidelity impair generalisation?
  3. **Structured Forgetting**: Is forgetting selective by attractor basin
     geometry?
  4. **Partial-Cue Retrieval**: Cue with 20-80% of the assembly and measure
     completion probability.
  5. **Schema-Core Generalization**: Cue only the Schema Core — does the
     network activate a blended pattern?
  6. **Replay Diversity**: Entropy, fragmentation, and variability of replay.

All probes use adaptive thresholds that scale with mean weight magnitude.
"""

import numpy as np
import torch

from compare_catastrophic_forgetting import DEVICE, CUE_STRENGTH, TEST_NOISE, _safe_mean
import compare_catastrophic_forgetting as _ccf

# DEV_MODE speed-up
try:
    from compare_catastrophic_forgetting import DEV_MODE as _DEV
except ImportError:
    _DEV = False

# ── Config ──────────────────────────────────────────────────────────────
PROBE_STEPS = 50 if _DEV else 200       # ms for each probe window
N_GENERALIZATION_TRIALS = 3 if _DEV else 10  # trials for each generalization probe
N_CUE_FRACTIONS = [0.20, 0.30, 0.50, 0.75, 1.0]  # partial cue fractions

# Adaptive threshold: mean_weight_multiplier * mean(W_ee)
ADAPTIVE_THRESHOLD_MULT = 3.0


# ── Adaptive threshold helper ───────────────────────────────────────────

def _adaptive_threshold(net):
    """Return a firing-rate threshold scaled by mean W_ee magnitude."""
    ei = slice(0, _ccf.N_EXC)
    mean_w = float(net.W.data[ei, ei].abs().mean().item())
    return mean_w * ADAPTIVE_THRESHOLD_MULT + 0.01


# ── State save/restore helper ───────────────────────────────────────────

def _save_net_state(net):
    s = dict(
        v=net.v.detach().clone(),
        u=net.u.detach().clone(),
        I_syn=net.I_syn.detach().clone(),
        spikes=net.spikes.detach().clone(),
        noise_std=net.noise_std,
    )
    if hasattr(net, "pre_trace") and net.pre_trace is not None:
        s["pre_trace"] = net.pre_trace.detach().clone()
        s["post_trace"] = net.post_trace.detach().clone()
    if hasattr(net, "stdp_enabled"):
        s["stdp_enabled"] = net.stdp_enabled
    return s


def _restore_net_state(net, s):
    with torch.no_grad():
        net.v.copy_(s["v"])
        net.u.copy_(s["u"])
        net.I_syn.copy_(s["I_syn"])
        net.spikes.copy_(s["spikes"])
        net.noise_std = s["noise_std"]
        if "pre_trace" in s:
            net.pre_trace.copy_(s["pre_trace"])
            net.post_trace.copy_(s["post_trace"])
        if "stdp_enabled" in s:
            net.stdp_enabled = s["stdp_enabled"]


# ═════════════════════════════════════════════════════════════════════════
# 1.  PARTIAL-CUE RETRIEVAL EXPERIMENTS
# ═════════════════════════════════════════════════════════════════════════

def test_partial_cue_retrieval(net, assemblies, core_mask=None,
                                cue_fractions=N_CUE_FRACTIONS, n_trials=5):
    """Test completion accuracy across cue fractions.

    For each assembly and each cue fraction, measure whether the network
    can complete the full assembly pattern from a partial cue.

    Returns:
        dict mapping assembly index -> {
            cue_fraction -> completion_probability
        }
    """
    saved = _save_net_state(net)
    n_mem = len(assemblies)
    threshold = _adaptive_threshold(net)
    net.noise_std = TEST_NOISE * 0.3

    results = {}
    for aidx in range(min(n_mem, 8)):
        asm = assemblies[aidx]
        asm_exc = asm[asm < _ccf.N_EXC]
        asm_size = len(asm_exc)
        cue_results = {}
        for frac in cue_fractions:
            correct = 0
            for _ in range(n_trials):
                cue_size = max(1, int(asm_size * frac))
                if core_mask is not None:
                    core_exc = core_mask[core_mask < _ccf.N_EXC]
                    core_in_asm = np.intersect1d(asm_exc, core_exc)
                    unique_in_asm = np.setdiff1d(asm_exc, core_exc)
                    if len(core_in_asm) >= cue_size:
                        cue_neurons = np.random.choice(core_in_asm, cue_size, replace=False)
                    elif len(unique_in_asm) > 0:
                        n_core = min(len(core_in_asm), cue_size // 2)
                        n_unique = cue_size - n_core
                        cue_neurons = np.concatenate([
                            np.random.choice(core_in_asm, max(1, n_core), replace=False),
                            np.random.choice(unique_in_asm, max(1, n_unique), replace=False),
                        ])
                    else:
                        cue_neurons = np.random.choice(asm_exc, cue_size, replace=False)
                else:
                    cue_neurons = np.random.choice(asm_exc, cue_size, replace=False)

                stim = torch.zeros(_ccf.N_NEURONS, device=DEVICE)
                stim[cue_neurons] = CUE_STRENGTH * 0.5

                net.reset_state()
                with torch.no_grad():
                    for _ in range(PROBE_STEPS):
                        net.forward(stim)
                    evoked = torch.where(net.spikes > threshold)[0].cpu().numpy()
                    overlap = len(np.intersect1d(evoked, asm_exc))
                    if overlap >= len(asm_exc) * 0.5:
                        correct += 1
            cue_results[frac] = correct / max(1, n_trials)
        results[aidx] = cue_results

    _restore_net_state(net, saved)
    return results


# ═════════════════════════════════════════════════════════════════════════
# 2.  COMPLETION ACCURACY CURVES
# ═════════════════════════════════════════════════════════════════════════

def compute_completion_curves(net, assemblies, core_mask=None,
                               cue_fractions=N_CUE_FRACTIONS, n_trials=5):
    """Compute completion accuracy curves across cue fractions.

    Returns a flat dict mapping (aidx, frac) -> completion_probability,
    plus mean over assemblies.
    """
    results = test_partial_cue_retrieval(net, assemblies, core_mask,
                                          cue_fractions, n_trials)
    # Flatten and aggregate
    flat = {}
    mean_by_frac = {f: [] for f in cue_fractions}
    for aidx, cue_data in results.items():
        for frac, prob in cue_data.items():
            flat[(int(aidx), float(frac))] = prob
            mean_by_frac[frac].append(prob)
    return {
        "per_assembly": results,
        "mean_by_fraction": {f: _safe_mean(mean_by_frac[f]) for f in cue_fractions},
    }


# ═════════════════════════════════════════════════════════════════════════
# 3.  SCHEMA-CORE GENERALIZATION PROBE
# ═════════════════════════════════════════════════════════════════════════

def test_schema_core_generalization(net, assemblies, core_mask, n_trials=5):
    """Cue only the Schema Core (no unique neurons).

    Measures whether the network activates a 'blended' pattern across all
    unique sets (schema core has become an independent attractor) or
    remains silent.

    Returns:
        dict with:
          - "blended_activation": mean firing rate of all unique sets
          - "specificity": max(unique_i) / mean(unique_j for j != i)
          - "core_completion": fraction of core neurons that fire
          - "activation_by_memory": list of mean firing per unique set
    """
    saved = _save_net_state(net)
    threshold = _adaptive_threshold(net)
    n_mem = len(assemblies)
    net.noise_std = TEST_NOISE * 0.3

    if core_mask is None or len(core_mask) == 0:
        _restore_net_state(net, saved)
        return {"blended_activation": 0.0, "specificity": 0.0,
                "core_completion": 0.0, "activation_by_memory": []}

    core_exc = core_mask[core_mask < _ccf.N_EXC]
    core_cue = np.random.choice(core_exc, size=min(15, len(core_exc)), replace=False)

    activation_by_memory = []
    for aidx in range(min(n_mem, 8)):
        asm = assemblies[aidx]
        asm_exc = asm[asm < _ccf.N_EXC]
        if core_mask is not None:
            unique_exc = np.setdiff1d(asm_exc, core_exc)
        else:
            unique_exc = asm_exc
        if len(unique_exc) == 0:
            continue
        stim = torch.zeros(_ccf.N_NEURONS, device=DEVICE)
        stim[core_cue] = CUE_STRENGTH * 0.5

        net.reset_state()
        core_fired = 0
        unique_fired = 0
        with torch.no_grad():
            for _ in range(PROBE_STEPS):
                net.forward(stim)
                core_fired += int((net.spikes[core_exc] > threshold).sum().item())
                unique_fired += int((net.spikes[unique_exc] > threshold).sum().item())

        activation_by_memory.append(unique_fired / max(1, PROBE_STEPS * len(unique_exc)))

    if not activation_by_memory:
        _restore_net_state(net, saved)
        return {"blended_activation": 0.0, "specificity": 0.0,
                "core_completion": 0.0, "activation_by_memory": []}

    blended = _safe_mean(activation_by_memory)
    if len(activation_by_memory) > 1:
        arr = np.array(activation_by_memory)
        specificity = float(max(arr) / max(np.mean(np.delete(arr, np.argmax(arr))), 1e-10))
    else:
        specificity = 1.0

    # Core completion rate
    stim = torch.zeros(_ccf.N_NEURONS, device=DEVICE)
    stim[core_cue] = CUE_STRENGTH * 0.5
    net.reset_state()
    with torch.no_grad():
        for _ in range(PROBE_STEPS):
            net.forward(stim)
    core_final = torch.where(net.spikes[core_exc] > threshold)[0].cpu().numpy()
    core_completion = len(core_final) / max(1, len(core_exc))

    _restore_net_state(net, saved)
    return {
        "blended_activation": float(blended),
        "specificity": float(specificity),
        "core_completion": float(core_completion),
        "activation_by_memory": activation_by_memory,
    }


# ═════════════════════════════════════════════════════════════════════════
# 4.  GENERALIZATION PROBE (Schema-Consistent Novel Inputs)
# ═════════════════════════════════════════════════════════════════════════

def create_schema_consistent_novel_input(net, assemblies, base_idx=0,
                                          core_mask=None):
    """Create a novel input that blends the base assembly with its neighbour.

    If core_mask is provided, cues only the Schema Core neurons.
    """
    n_mem = len(assemblies)
    stim = torch.zeros(_ccf.N_NEURONS, device=DEVICE)
    threshold = _adaptive_threshold(net)

    if core_mask is not None and len(core_mask) > 0:
        core_exc = core_mask[core_mask < _ccf.N_EXC]
        n_cue = min(10, len(core_exc))
        cue = np.random.choice(core_exc, n_cue, replace=False)
        stim[cue] = CUE_STRENGTH * 0.5
    else:
        if base_idx < n_mem:
            asm_a = assemblies[base_idx]
            asm_a_exc = asm_a[asm_a < _ccf.N_EXC]
            n_a = max(1, int(0.7 * len(asm_a_exc)))
            idx_a = np.random.choice(asm_a_exc, n_a, replace=False)
            stim[idx_a] = CUE_STRENGTH * 0.6
        if base_idx + 1 < n_mem:
            asm_b = assemblies[base_idx + 1]
            asm_b_exc = asm_b[asm_b < _ccf.N_EXC]
            n_b = max(1, int(0.3 * len(asm_b_exc)))
            idx_b = np.random.choice(asm_b_exc, n_b, replace=False)
            stim[idx_b] = CUE_STRENGTH * 0.4

    stim += torch.randn(_ccf.N_NEURONS, device=DEVICE) * 0.3
    return stim


def create_high_fidelity_input(net, assemblies, base_idx=0):
    """Full, veridical input — the complete assembly with no mixing."""
    stim = torch.zeros(_ccf.N_NEURONS, device=DEVICE)
    if base_idx < len(assemblies):
        stim[assemblies[base_idx]] = CUE_STRENGTH
    return stim


def test_generalization(net, assemblies, core_mask=None,
                         n_trials=N_GENERALIZATION_TRIALS):
    """Test how well the network generalises to schema-consistent novel inputs.

    Uses adaptive thresholds.  Measures completion of the expected next
    assembly pattern.

    Returns:
        dict mapping assembly index -> {
            "generalization_score": float (0-1),
            "completion_rate": float (0-1),
            "assembly_activations": list of floats,
        }
    """
    saved = _save_net_state(net)
    n_mem = len(assemblies)
    threshold = _adaptive_threshold(net)
    net.noise_std = TEST_NOISE * 0.5
    results = {}
    for aidx in range(n_mem):
        correct_completions = 0
        all_activations = []
        completion_trials = 0
        for _ in range(n_trials):
            stim = create_schema_consistent_novel_input(net, assemblies, aidx, core_mask)
            net.reset_state()
            if net.stdp_enabled:
                net.pre_trace.zero_()
                net.post_trace.zero_()
            act = np.zeros(n_mem, dtype=np.float32)
            with torch.no_grad():
                for _ in range(PROBE_STEPS):
                    net.forward(stim)
                    for ai in range(n_mem):
                        asm = assemblies[ai]
                        n_fired = int((net.spikes[asm] > threshold).sum().item())
                        act[ai] += n_fired
            act /= max(1, PROBE_STEPS)
            all_activations.append(act.copy())
            expected_next = aidx + 1 if aidx + 1 < n_mem else aidx
            if expected_next < n_mem and expected_next != aidx:
                winner = int(np.argmax(act))
                if winner == expected_next:
                    correct_completions += 1
                # Completion check: does target assembly fire?
                target = assemblies[aidx]
                target_exc = target[target < _ccf.N_EXC]
                if len(target_exc) > 0:
                    target_rate = act[aidx] / max(1, len(target_exc))
                    if target_rate > threshold * 0.3:
                        completion_trials += 1
        results[aidx] = {
            "generalization_score": correct_completions / max(1, n_trials),
            "expected_next": aidx + 1 if aidx + 1 < n_mem else aidx,
            "assembly_activations": np.mean(all_activations, axis=0).tolist(),
            "completion_rate": completion_trials / max(1, n_trials),
        }
    _restore_net_state(net, saved)
    return results


# ═════════════════════════════════════════════════════════════════════════
# 5.  ANTI-PREDICTION PROBE
# ═════════════════════════════════════════════════════════════════════════

def test_anti_prediction(net, assemblies, core_mask=None, n_trials=5):
    """Compare generalisation after veridical (high-fidelity) cue vs
    natural (schema-blend) cue.

    Uses adaptive thresholds.  Measures suppression of complementary
    unique sets (anti-prediction: the unique parts of non-target memories
    should be actively suppressed below baseline).

    Returns:
        dict with "natural_gen", "high_fidelity_gen", "suppression_depth"
    """
    saved = _save_net_state(net)
    threshold = _adaptive_threshold(net)
    net.noise_std = TEST_NOISE * 0.5
    n_mem = len(assemblies)
    natural_scores = []
    hf_scores = []
    suppression_depths = []
    for aidx in range(n_mem):
        if aidx + 1 >= n_mem:
            continue
        nat_correct = 0
        hf_correct = 0
        for _ in range(n_trials):
            net.reset_state()
            if net.stdp_enabled:
                net.pre_trace.zero_()
                net.post_trace.zero_()
            stim_nat = create_schema_consistent_novel_input(net, assemblies, aidx, core_mask)
            with torch.no_grad():
                act_nat = np.zeros(n_mem, dtype=np.float32)
                for _ in range(PROBE_STEPS):
                    net.forward(stim_nat)
                    for ai in range(n_mem):
                        act_nat[ai] += int((net.spikes[assemblies[ai]] > threshold).sum().item())
                act_nat /= max(1, PROBE_STEPS)
            expected = aidx + 1
            if np.argmax(act_nat) == expected:
                nat_correct += 1

            # Suppression depth: measure firing of non-target unique sets
            if core_mask is not None and len(core_mask) > 0:
                core_exc = core_mask[core_mask < _ccf.N_EXC]
                target_unique = np.setdiff1d(assemblies[aidx][assemblies[aidx] < _ccf.N_EXC], core_exc)
                other_unique_rates = []
                for oi in range(n_mem):
                    if oi == aidx:
                        continue
                    other_asm = assemblies[oi]
                    other_u = np.setdiff1d(other_asm[other_asm < _ccf.N_EXC], core_exc)
                    if len(other_u) > 0:
                        rate = act_nat[oi] / max(1, len(other_u))
                        other_unique_rates.append(rate)
                if other_unique_rates:
                    suppression_depths.append(-_safe_mean(other_unique_rates))

            net.reset_state()
            if net.stdp_enabled:
                net.pre_trace.zero_()
                net.post_trace.zero_()
            stim_hf = create_high_fidelity_input(net, assemblies, aidx)
            with torch.no_grad():
                act_hf = np.zeros(n_mem, dtype=np.float32)
                for _ in range(PROBE_STEPS):
                    net.forward(stim_hf)
                    for ai in range(n_mem):
                        act_hf[ai] += int((net.spikes[assemblies[ai]] > threshold).sum().item())
                act_hf /= max(1, PROBE_STEPS)
            if np.argmax(act_hf) == expected:
                hf_correct += 1
        natural_scores.append(nat_correct / max(1, n_trials))
        hf_scores.append(hf_correct / max(1, n_trials))
    _restore_net_state(net, saved)
    return {
        "natural_generalization": _safe_mean(natural_scores),
        "high_fidelity_generalization": _safe_mean(hf_scores),
        "per_assembly_natural": natural_scores,
        "per_assembly_hf": hf_scores,
        "suppression_depth": _safe_mean(suppression_depths) if suppression_depths else 0.0,
    }


# ═════════════════════════════════════════════════════════════════════════
# 6.  BASIN GEOMETRY (STRUCTURED FORGETTING)
# ═════════════════════════════════════════════════════════════════════════

def compute_basin_geometry(net, assemblies, core_mask=None, n_trials=20):
    """Compute attractor basin geometry for each assembly.

    Measures:
      - basin_volume: fraction of noise-initialised trials
      - retrieval_probability: fraction of partial-cue trials
      - basin_persistence: how long the attractor sustains after cue
      - overlap_score: number of overlapping assemblies

    Uses adaptive thresholds.
    """
    saved = _save_net_state(net)
    n_mem = len(assemblies)
    threshold = _adaptive_threshold(net)
    net.noise_std = TEST_NOISE
    results = {}
    for aidx in range(n_mem):
        asm = assemblies[aidx]
        asm_exc = asm[asm < _ccf.N_EXC]
        # Basin volume
        basin_hits = 0
        for _ in range(n_trials):
            net.reset_state()
            if net.stdp_enabled:
                net.pre_trace.zero_()
                net.post_trace.zero_()
            noise_stim = torch.randn(_ccf.N_NEURONS, device=DEVICE) * 0.5
            with torch.no_grad():
                for _ in range(PROBE_STEPS):
                    net.forward(noise_stim)
                evoked = torch.where(net.spikes > threshold)[0].cpu().numpy()
                max_overlap = 0
                best_match = -1
                for ai in range(n_mem):
                    overlap = len(np.intersect1d(evoked, assemblies[ai]))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_match = ai
                if best_match == aidx and max_overlap > 0:
                    basin_hits += 1
        # Retrieval probability with partial cue (core-first if available)
        retrieval_hits = 0
        n_retrieval = 10
        for _ in range(n_retrieval):
            net.reset_state()
            if net.stdp_enabled:
                net.pre_trace.zero_()
                net.post_trace.zero_()

            if core_mask is not None and len(core_mask) > 0:
                core_exc = core_mask[core_mask < _ccf.N_EXC]
                core_in_asm = np.intersect1d(asm_exc, core_exc)
                if len(core_in_asm) >= 3:
                    cue = np.random.choice(core_in_asm, min(5, len(core_in_asm)), replace=False)
                else:
                    cue = np.random.choice(asm_exc, max(1, len(asm_exc) // 3), replace=False)
            else:
                cue = np.random.choice(asm_exc, max(1, len(asm_exc) // 3), replace=False)

            stim = torch.zeros(_ccf.N_NEURONS, device=DEVICE)
            stim[cue] = CUE_STRENGTH
            with torch.no_grad():
                for _ in range(PROBE_STEPS):
                    net.forward(stim)
                evoked = torch.where(net.spikes > threshold)[0].cpu().numpy()
                overlap = len(np.intersect1d(evoked, asm_exc))
                if overlap >= len(asm_exc) * 0.5:
                    retrieval_hits += 1

        # Persistence: measure sustained firing after cue offset
        persistence = 0.0
        net.reset_state()
        if net.stdp_enabled:
            net.pre_trace.zero_()
            net.post_trace.zero_()
        stim = torch.zeros(_ccf.N_NEURONS, device=DEVICE)
        cue = np.random.choice(asm_exc, min(10, len(asm_exc)), replace=False)
        stim[cue] = CUE_STRENGTH
        with torch.no_grad():
            for _ in range(5):
                net.forward(stim)
            stim.zero_()
            sustained = 0
            for s in range(PROBE_STEPS):
                net.forward(stim)
                sustained += int((net.spikes[asm_exc] > threshold).sum().item())
            persistence = sustained / max(1, PROBE_STEPS * len(asm_exc))

        overlap_count = 0
        for ai in range(n_mem):
            if ai != aidx:
                shared = len(np.intersect1d(assemblies[aidx], assemblies[ai]))
                if shared > 0:
                    overlap_count += 1
        results[aidx] = {
            "basin_volume": basin_hits / max(1, n_trials),
            "retrieval_probability": retrieval_hits / max(1, n_retrieval),
            "basin_persistence": float(persistence),
            "n_overlapping": overlap_count,
            "assembly_size": len(asm_exc),
        }
    _restore_net_state(net, saved)
    return results


def compute_structured_forgetting(net, assemblies, baseline_scores, final_scores):
    """Compute which memories were 'forgotten' and correlate with basin geometry."""
    basin = compute_basin_geometry(net, assemblies)
    retention_changes = {}
    for aidx in range(len(assemblies)):
        bl = baseline_scores[aidx] if aidx < len(baseline_scores) else 0.0
        fn = final_scores[aidx] if aidx < len(final_scores) else 0.0
        retention_changes[aidx] = fn - bl
    volumes = np.array([basin[aidx]["basin_volume"] for aidx in range(len(assemblies))])
    changes = np.array([retention_changes[aidx] for aidx in range(len(assemblies))])
    valid = np.isfinite(volumes) & np.isfinite(changes)
    if valid.sum() >= 3:
        from scipy.stats import pearsonr as _pr
        r_val, p_val = _pr(volumes[valid], changes[valid])
        basin_protection = {"r": r_val, "p": p_val}
    else:
        basin_protection = {"r": 0.0, "p": 1.0}
    return {
        "basin_geometry": basin,
        "retention_changes": retention_changes,
        "basin_protection": basin_protection,
    }


# ═════════════════════════════════════════════════════════════════════════
# 7.  REPLAY DIVERSITY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

def analyze_replay_diversity(all_replay_metrics):
    """Compute replay entropy, fragmentation, variability.

    Args:
        all_replay_metrics: list of per-event replay metric dicts.

    Returns:
        dict with entropy, fragmentation, variability scores.
    """
    if not all_replay_metrics:
        return {"entropy": 0.0, "fragmentation": 0.0, "variability": 0.0,
                "n_events": 0}

    # Entropy of assembly selection
    assembly_counts = {}
    for ev in all_replay_metrics:
        idx = ev.get("assembly_idx", -1)
        assembly_counts[idx] = assembly_counts.get(idx, 0) + 1
    total = sum(assembly_counts.values())
    if total > 0:
        probs = np.array([c / total for c in assembly_counts.values()])
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log(probs)))
    else:
        entropy = 0.0

    # Fragmentation: mean number of coherent runs per event
    run_lengths = []
    for ev in all_replay_metrics:
        runs = ev.get("coherent_run_lengths", [])
        if runs:
            run_lengths.extend(runs)
    fragmentation = float(np.std(run_lengths)) / max(float(np.mean(run_lengths)), 1e-10) if run_lengths else 0.0

    # Variability: std of mean_coherence across events
    coherences = [ev.get("mean_coherence", 0.0) for ev in all_replay_metrics]
    variability = float(np.std(coherences)) if len(coherences) >= 2 else 0.0

    return {
        "entropy": entropy,
        "fragmentation": fragmentation,
        "variability": variability,
        "n_events": len(all_replay_metrics),
    }


# ═════════════════════════════════════════════════════════════════════════
# 8.  HOMEOSTATIC ATTRACTOR BASIN STABILIZATION
# ═════════════════════════════════════════════════════════════════════════

def stabilize_attractor_basins(net, assemblies, core_mask=None):
    """Apply homeostatic attractor basin stabilization.

    Normalises recurrent excitation per assembly, maintains minimum
    recurrent support, and scales inhibition adaptively.

    This is called during the final probe phase to ensure basin
    measurements reflect stable attractor states.
    """
    n_mem = len(assemblies)
    ei = slice(0, _ccf.N_EXC)
    w = net.W.data[ei, ei]

    for aidx in range(n_mem):
        asm = assemblies[aidx]
        asm_exc = asm[asm < _ccf.N_EXC]
        if len(asm_exc) < 2:
            continue

        # Normalize recurrent excitation per assembly
        sub = w[np.ix_(asm_exc, asm_exc)]
        mean_w = float(sub.mean().item())
        if mean_w > 1e-6:
            target_mean = 0.1  # target mean weight within assembly
            scale = target_mean / mean_w
            with torch.no_grad():
                sub.mul_(min(scale, 2.0))

    # Adaptive inhibition scaling: maintain E/I balance
    n_total = net.W.shape[0]
    if n_total > _ccf.N_EXC:
        mean_e = float(w[ei, ei].mean().item())
        mean_ei = float(net.W.data[:_ccf.N_EXC, _ccf.N_EXC:].abs().mean().item())
        if mean_e > 1e-6 and mean_ei > 1e-6:
            target_ei_ratio = 3.0
            current_ratio = mean_e / max(mean_ei, 1e-10)
            if current_ratio < target_ei_ratio * 0.5:
                with torch.no_grad():
                    net.W.data[:_ccf.N_EXC, _ccf.N_EXC:].mul_(1.05)
            elif current_ratio > target_ei_ratio * 2.0:
                with torch.no_grad():
                    net.W.data[:_ccf.N_EXC, _ccf.N_EXC:].mul_(0.95)


# ── Hook callbacks ─────────────────────────────────────────────────────

def _probes_final_hook(net, assemblies, n_mem, **_):
    """Run all probes at the final hook and store in hook_extra."""
    extra = getattr(net, "_hook_extra", None)
    if extra is None:
        return

    core_mask = getattr(net, "_schema_core_mask", None)

    # Stabilize attractors before probing
    try:
        stabilize_attractor_basins(net, assemblies, core_mask)
    except Exception as e:
        extra["stabilization_error"] = str(e)

    try:
        extra["generalization"] = test_generalization(net, assemblies, core_mask)
    except Exception as e:
        extra["generalization"] = {"error": str(e)}
    try:
        extra["anti_prediction"] = test_anti_prediction(net, assemblies, core_mask)
    except Exception as e:
        extra["anti_prediction"] = {"error": str(e)}

    # Partial cue retrieval experiments
    try:
        extra["completion_curves"] = compute_completion_curves(net, assemblies, core_mask)
    except Exception as e:
        extra["completion_curves"] = {"error": str(e)}

    # Schema-core generalization
    try:
        extra["schema_core_gen"] = test_schema_core_generalization(net, assemblies, core_mask)
    except Exception as e:
        extra["schema_core_gen"] = {"error": str(e)}
