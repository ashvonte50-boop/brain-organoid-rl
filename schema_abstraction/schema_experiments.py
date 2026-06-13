"""Orchestrator: hierarchical schema architecture, multi-seed sweeps,
overlap sweeps, downscale-rate sweeps, and publication experiments.

Assembly Layout (Hierarchical Schema):
  Schema Core: indices [0, SCHEMA_CORE_SIZE) — shared across all memories.
  Memory A:  Schema Core + unique_A  [SCHEMA_CORE_SIZE, SCHEMA_CORE_SIZE + UNIQUE_SIZE)
  Memory B:  Schema Core + unique_B  [SCHEMA_CORE_SIZE + UNIQUE_SIZE, SCHEMA_CORE_SIZE + 2*UNIQUE_SIZE)
  ...etc.

Test memories (post-training):
  Memory E: Schema Core + unique_E (novel, same schema — tests forward transfer)
  Memory F: fully random (no schema overlap — baseline control)

Supports:
  - Multi-seed execution for statistical power.
  - Overlap sweeps (0%, 10%, 20%, 40%, 60%).
  - Downscale-rate sweeps (0.0005, 0.001, 0.002, 0.003).
  - Ablation experiments (natural vs perfect replay).
"""

import numpy as np
import torch

from .schema_core import register_schema_hooks
from .schema_downscaling import DOWNSCALE_RATE

import compare_catastrophic_forgetting as _ccf
from compare_catastrophic_forgetting import (
    MASTER_SEED, N_MEMORIES, ASSEMBLY_SIZE,
    run_all_conditions as _run_all_conditions,
    make_overlapping_assemblies as _make_overlapping_assemblies,
)
# Live references — use _ccf.N_NEURONS / _ccf.N_EXC (not bare imports)

# ── Hierarchical Schema Config ──────────────────────────────────────────
SCHEMA_CORE_SIZE = 20       # shared core neurons (reduced from 40 to prevent runaway)
UNIQUE_SIZE = 20            # unique neurons per memory (reduced from 40)
N_HIER_MEMORIES = 4         # A, B, C, D

# Test memories
TEST_MEMORY_E_CORE = True   # Memory E shares the core (forward transfer test)
TEST_MEMORY_F_RANDOM = True # Memory F is fully random (no core — control)

# Baselines for comparison
OVERLAP_FRACS_SWEEP = [0.0, 0.1, 0.2, 0.4, 0.6]
DOWNSCALE_RATES_SWEEP = [0.0005, 0.001, 0.002, 0.003]

# Seeds for multi-seed robustness
SWEEP_SEEDS = [1000, 2000, 3000, 4000, 5000]


def make_schema_assemblies(n_memories=N_HIER_MEMORIES,
                           core_size=SCHEMA_CORE_SIZE,
                           unique_size=UNIQUE_SIZE,
                           total_neurons=400):
    """Create hierarchical schema assemblies.

    All assemblies share a Schema Core.  Each also has its own unique
    set of neurons.  Total used neurons = core_size + n_memories * unique_size.

    Returns:
        assemblies: list of index arrays (length n_memories)
        core_mask: array of schema-core neuron indices
    """
    assert core_size + n_memories * unique_size <= total_neurons, \
        f"Need {core_size + n_memories * unique_size} neurons but only {total_neurons} available"
    assert total_neurons <= _ccf.N_EXC, "All assemblies must be within excitatory pool"

    core_mask = np.arange(core_size, dtype=int)
    assemblies = []
    for m in range(n_memories):
        start = core_size + m * unique_size
        unique = np.arange(start, start + unique_size, dtype=int)
        asm = np.concatenate([core_mask, unique])
        assemblies.append(asm)
    return assemblies, core_mask


def make_test_memory_e(assemblies, core_mask,
                       unique_size=UNIQUE_SIZE):
    """Create Memory E that shares the Schema Core but has a new unique set.

    Returns a single assembly array, or None if no space.
    """
    n_existing = len(assemblies)
    start_unique = SCHEMA_CORE_SIZE + n_existing * unique_size
    end_unique = start_unique + unique_size
    if end_unique > _ccf.N_EXC:
        return None
    unique_e = np.arange(start_unique, end_unique, dtype=int)
    mem_e = np.concatenate([core_mask, unique_e])
    return mem_e


def make_test_memory_f(core_size=SCHEMA_CORE_SIZE,
                       unique_size=UNIQUE_SIZE,
                       n_hier_memories=N_HIER_MEMORIES):
    """Create Memory F that is fully random (no core overlap — baseline control).

    Returns a single assembly array, or None if no space.
    """
    start_f = SCHEMA_CORE_SIZE + (n_hier_memories + 1) * unique_size
    end_f = start_f + ASSEMBLY_SIZE
    if end_f > _ccf.N_EXC:
        return None
    mem_f = np.arange(start_f, end_f, dtype=int)
    return mem_f


# ── Metrics: Schema Coherence, Specificity, Schema Extraction Index ─────

def measure_schema_coherence(net, assemblies, core_mask):
    """When cueing a memory, measure core vs background firing."""
    if core_mask is None or len(core_mask) == 0:
        return {"core_rate": 0.0, "bg_rate": 0.0, "ratio": 0.0}
    n_mem = len(assemblies)
    from compare_catastrophic_forgetting import CUE_STRENGTH, PROBE_STEPS, DEVICE
    probe_steps = 50
    core_exc = core_mask[core_mask < _ccf.N_EXC]
    bg_exc = np.setdiff1d(np.arange(_ccf.N_EXC), core_exc)
    if len(bg_exc) > 1000:
        bg_exc = np.random.choice(bg_exc, 1000, replace=False)

    core_rates = []
    bg_rates = []
    for aidx in range(min(n_mem, 4)):
        asm = assemblies[aidx]
        asm_exc = asm[asm < _ccf.N_EXC]
        cue_strength = CUE_STRENGTH
        stim = torch.zeros(_ccf.N_NEURONS, device=DEVICE)
        cue = np.random.choice(asm_exc, size=min(15, len(asm_exc)), replace=False)
        stim[cue] = cue_strength * 0.5

        if hasattr(net, 'noise_std'):
            orig_noise = net.noise_std
        saved_v = net.v.detach().clone()
        saved_u = net.u.detach().clone()
        saved_I = net.I_syn.detach().clone()
        saved_spikes = net.spikes.detach().clone()

        net.reset_state()
        core_fired_total = 0
        bg_fired_total = 0
        with torch.no_grad():
            for _ in range(probe_steps):
                net.forward(stim)
                core_fired_total += int(net.spikes[core_exc].sum().item())
                bg_fired_total += int(net.spikes[bg_exc].sum().item())

        core_rate = core_fired_total / max(1, len(core_exc))
        bg_rate = bg_fired_total / max(1, len(bg_exc))
        core_rates.append(core_rate)
        bg_rates.append(bg_rate)

        net.v.copy_(saved_v)
        net.u.copy_(saved_u)
        net.I_syn.copy_(saved_I)
        net.spikes.copy_(saved_spikes)

    return {
        "core_rate": float(np.mean(core_rates)),
        "bg_rate": float(np.mean(bg_rates)),
        "ratio": float(np.mean(core_rates) / max(np.mean(bg_rates), 1e-10)),
    }


def measure_memory_specificity(net, assemblies, core_mask):
    """When cueing Memory A, measure unique-A vs unique-B/C/D firing."""
    n_mem = len(assemblies)
    from compare_catastrophic_forgetting import CUE_STRENGTH, DEVICE
    probe_steps = 50
    results = {}
    for aidx in range(min(n_mem, 4)):
        asm = assemblies[aidx]
        asm_exc = asm[asm < _ccf.N_EXC]
        if core_mask is not None:
            core_exc = core_mask[core_mask < _ccf.N_EXC]
            unique_exc = np.setdiff1d(asm_exc, core_exc)
        else:
            unique_exc = asm_exc
        if len(unique_exc) == 0:
            continue

        cue = np.random.choice(asm_exc, size=min(15, len(asm_exc)), replace=False)
        stim = torch.zeros(_ccf.N_NEURONS, device=DEVICE)
        stim[cue] = CUE_STRENGTH * 0.5

        saved_v = net.v.detach().clone()
        saved_u = net.u.detach().clone()
        saved_I = net.I_syn.detach().clone()
        saved_spikes = net.spikes.detach().clone()

        net.reset_state()
        self_unique_fired = 0
        other_unique_fired = 0
        total_other = 0
        with torch.no_grad():
            for _ in range(probe_steps):
                net.forward(stim)
                self_unique_fired += int(net.spikes[unique_exc].sum().item())
                for oidx in range(n_mem):
                    if oidx == aidx:
                        continue
                    oasm = assemblies[oidx]
                    if core_mask is not None:
                        o_exc = np.setdiff1d(oasm[oasm < _ccf.N_EXC], core_exc)
                    else:
                        o_exc = oasm[oasm < _ccf.N_EXC]
                    if len(o_exc) > 0:
                        other_unique_fired += int(net.spikes[o_exc].sum().item())
                        total_other += 1

        self_rate = self_unique_fired / max(1, len(unique_exc) * probe_steps)
        other_rate = other_unique_fired / max(1, total_other * probe_steps) if total_other > 0 else 0.0

        results[aidx] = {
            "self_unique_rate": float(self_rate),
            "other_unique_rate": float(other_rate),
            "specificity": float(self_rate / max(other_rate, 1e-10)),
        }

        net.v.copy_(saved_v)
        net.u.copy_(saved_u)
        net.I_syn.copy_(saved_I)
        net.spikes.copy_(saved_spikes)

    return results


def compute_schema_extraction_index(net, core_mask):
    """(mean core weight - mean random weight) / (mean core + mean random)."""
    ei = slice(0, _ccf.N_EXC)
    w = net.W.data[ei, ei]
    if core_mask is None or len(core_mask) == 0:
        return 0.0
    core_exc = core_mask[core_mask < _ccf.N_EXC]
    core_idx = torch.tensor(core_exc, device=w.device, dtype=torch.long)
    core_weights = w[core_idx[:, None], core_idx]
    core_mean = float(core_weights.mean().item())
    n_total = w.shape[0]
    all_idx = torch.randperm(n_total, device=w.device)[:len(core_exc)]
    rand_weights = w[all_idx[:, None], all_idx]
    rand_mean = float(rand_weights.mean().item())
    denom = core_mean + rand_mean
    if denom < 1e-10:
        return 0.0
    return (core_mean - rand_mean) / denom


def measure_schema_forward_transfer(net, mem_e, mem_f, n_steps=50):
    """Measure encoding speed for Memory E (core) vs Memory F (random).

    Returns dict with steps_to_encode_e, steps_to_encode_f, ratio.
    """
    from compare_catastrophic_forgetting import CUE_STRENGTH, DEVICE
    probe_steps = 5
    stim_e = torch.zeros(_ccf.N_NEURONS, device=DEVICE)
    cue_e = np.random.choice(mem_e[mem_e < _ccf.N_EXC],
                             size=min(10, len(mem_e)), replace=False)
    stim_e[cue_e] = CUE_STRENGTH

    stim_f = torch.zeros(_ccf.N_NEURONS, device=DEVICE)
    cue_f = np.random.choice(mem_f[mem_f < _ccf.N_EXC],
                             size=min(10, len(mem_f)), replace=False)
    stim_f[cue_f] = CUE_STRENGTH

    # Measure activity overlap after brief forward pass
    def _compute_overlap(stim, asm):
        net.reset_state()
        asm_exc = asm[asm < _ccf.N_EXC]
        with torch.no_grad():
            for _ in range(probe_steps):
                net.forward(stim)
        evoked = torch.where(net.spikes > 0)[0].cpu().numpy()
        return len(np.intersect1d(evoked, asm_exc)) / max(1, len(asm_exc))

    saved_v = net.v.detach().clone()
    saved_u = net.u.detach().clone()
    saved_I = net.I_syn.detach().clone()
    saved_spikes = net.spikes.detach().clone()

    overlap_e = 0.0
    overlap_f = 0.0
    steps_e = n_steps
    steps_f = n_steps

    for step in range(1, n_steps + 1):
        oe = _compute_overlap(stim_e, mem_e)
        of = _compute_overlap(stim_f, mem_f)
        if oe > 0.3 and steps_e == n_steps:
            steps_e = step
        if of > 0.3 and steps_f == n_steps:
            steps_f = step
        overlap_e = oe
        overlap_f = of

    net.v.copy_(saved_v)
    net.u.copy_(saved_u)
    net.I_syn.copy_(saved_I)
    net.spikes.copy_(saved_spikes)

    return {
        "steps_to_encode_e": steps_e,
        "steps_to_encode_f": steps_f,
        "ratio": steps_f / max(steps_e, 1),
        "final_overlap_e": float(overlap_e),
        "final_overlap_f": float(overlap_f),
        "forward_transfer_benefit": float(max(0, steps_f - steps_e) / max(steps_f, 1)),
    }


def _attach_schema_data(all_results):
    """Move hook_extra data from each trial into top-level keys for analysis."""
    for res in all_results:
        for t in res.get("trials", []):
            extra = t.pop("hook_extra", None)
            if extra is None:
                t["centroid_snapshots"] = []
                t["distance_trajectories"] = {}
                t["schema_convergence"] = {}
                t["downscale_summary"] = None
                t["generative_layer"] = None
                t["generalization"] = None
                t["anti_prediction"] = None
                t["metaplasticity"] = None
                t["hidden_state"] = None
                t["novel_metrics"] = None
                t["schema_coherence"] = None
                t["forward_transfer"] = None
                t["replay_diversity"] = None
            else:
                t["centroid_snapshots"] = extra.get("centroid_snapshots", [])
                t["distance_trajectories"] = extra.get("distance_trajectories", {})
                t["schema_convergence"] = extra.get("schema_convergence", {})
                t["downscale_summary"] = extra.get("downscale_summary")
                t["generative_layer"] = extra.get("generative_layer")
                t["generalization"] = extra.get("generalization")
                t["anti_prediction"] = extra.get("anti_prediction")
                t["metaplasticity"] = extra.get("metaplasticity")
                t["hidden_state"] = extra.get("hidden_state")
                t["novel_metrics"] = extra.get("novel_metrics")
                t["schema_coherence"] = extra.get("schema_coherence")
                t["forward_transfer"] = extra.get("forward_transfer")
                t["replay_diversity"] = extra.get("replay_diversity")


def run_schema_abstraction_sweep(n_trials=10, n_seeds=1, **kwargs):
    """Run the full experiment sweep with schema-abstraction hooks active.

    Args:
        n_trials: trials per condition.
        n_seeds: number of random seeds for statistical power.
        **kwargs: forwarded to ``run_all_conditions``.

    Returns:
        (all_results_list, schema_results_list) — one element per seed.
    """
    from compare_catastrophic_forgetting import (
        run_all_conditions, MASTER_SEED,
        N_MEMORIES, ASSEMBLY_SIZE,
    )

    register_schema_hooks()

    all_seed_results = []
    all_seed_schema = []

    import compare_catastrophic_forgetting as ccf

    for seed_ix in range(n_seeds):
        actual_seed = MASTER_SEED + seed_ix * 10000

        print(f"\n{'=' * 70}", flush=True)
        print(f"SCHEMA SWEEP  seed={seed_ix + 1}/{n_seeds}  (seed_offset={actual_seed})", flush=True)
        print(f"{'=' * 70}", flush=True)

        ccf.MASTER_SEED = actual_seed
        ccf.torch.manual_seed(actual_seed)
        ccf.np.random.seed(actual_seed)

        # Use hierarchical schema assemblies
        assemblies, core_mask = make_schema_assemblies(
            n_memories=N_MEMORIES,
            core_size=SCHEMA_CORE_SIZE,
            unique_size=UNIQUE_SIZE,
        )

        # Store core mask on module for hooks to access
        import schema_abstraction.schema_core as sc
        sc._SCHEMA_CORE_MASK = core_mask

        all_results = _run_all_conditions(assemblies, n_trials=n_trials, **kwargs)

        _attach_schema_data(all_results)

        from .schema_analysis import run_all_schema_analysis
        from .schema_visualization import generate_all_schema_figures

        schema_results = run_all_schema_analysis(all_results, verbose=True)
        try:
            generate_all_schema_figures(all_results, schema_results)
        except Exception as e:
            print(f"  [SKIP] Schema figures: {e}", flush=True)

        all_seed_results.append(all_results)
        all_seed_schema.append(schema_results)

    if n_seeds > 1:
        from .schema_analysis import run_multi_seed_meta_analysis
        meta = run_multi_seed_meta_analysis(all_seed_schema)
        print("\n" + "=" * 70, flush=True)
        print("MULTI-SEED META-ANALYSIS", flush=True)
        print("=" * 70, flush=True)
        for test_name, test_data in meta.items():
            if isinstance(test_data, dict):
                for cond, vals in test_data.items():
                    if isinstance(vals, dict) and "mean_over_seeds" in vals:
                        print(f"  {test_name:25s} {cond:20s}: "
                              f"m={vals['mean_over_seeds']:.4f} "
                              f"se={vals['sem_over_seeds']:.4f} "
                              f"n_seeds={vals['n_seeds']}", flush=True)
        return all_seed_results, all_seed_schema, meta

    return all_seed_results, all_seed_schema


def run_overlap_sweep(n_trials=3, seeds=SWEEP_SEEDS[:3], verbose=True):
    """Run overlap sweep across [0%, 10%, 20%, 40%, 60%].

    Tests whether schema convergence scales monotonically with overlap.
    """
    from compare_catastrophic_forgetting import (
        run_all_conditions as _run_conditions,
        N_MEMORIES, ASSEMBLY_SIZE, MASTER_SEED,
    )

    register_schema_hooks()

    all_sweep_results = {}

    for overlap in OVERLAP_FRACS_SWEEP:
        print(f"\n{'=' * 60}", flush=True)
        print(f"OVERLAP SWEEP  overlap={overlap:.1f} ({int(overlap*100)}%)", flush=True)
        print(f"{'=' * 60}", flush=True)

        cond_results = []
        for seed_ix, seed in enumerate(seeds[:min(3, len(seeds))]):
            ccf.MASTER_SEED = seed
            ccf.torch.manual_seed(seed)
            ccf.np.random.seed(seed)

            assemblies = _make_overlapping_assemblies(N_MEMORIES, ASSEMBLY_SIZE, overlap)
            results = _run_conditions(assemblies, n_trials=n_trials)

            # Attach schema data
            _attach_schema_data(results)

            cond_results.append(results)

        from .schema_analysis import run_all_schema_analysis
        from .schema_visualization import generate_all_schema_figures

        combined_results = cond_results[0]  # Use first seed results as base
        schema_results = run_all_schema_analysis(combined_results, verbose=verbose)

        all_sweep_results[overlap] = {
            "results": combined_results,
            "schema": schema_results,
        }

    return all_sweep_results


def run_downscale_rate_sweep(n_trials=2, verbose=True):
    """Run downscale rate sweep across [0.0005, 0.001, 0.002, 0.003].

    Identifies the critical transition point where replay still matters
    but weights are not annihilated.
    """
    from compare_catastrophic_forgetting import (
        run_all_conditions as _run_conditions,
        N_MEMORIES, ASSEMBLY_SIZE, MASTER_SEED,
    )

    register_schema_hooks()

    all_sweep_results = {}

    for rate in DOWNSCALE_RATES_SWEEP:
        print(f"\n{'=' * 60}", flush=True)
        print(f"DOWNSCALE RATE SWEEP  rate={rate}", flush=True)
        print(f"{'=' * 60}", flush=True)

        # Patch the downscale rate temporarily
        import schema_abstraction.schema_downscaling as sd
        original_rate = sd.DOWNSCALE_RATE
        sd.DOWNSCALE_RATE = rate

        ccf.MASTER_SEED = MASTER_SEED
        ccf.torch.manual_seed(MASTER_SEED)
        ccf.np.random.seed(MASTER_SEED)

        assemblies, core_mask = make_schema_assemblies(
            n_memories=N_MEMORIES,
            core_size=SCHEMA_CORE_SIZE,
            unique_size=UNIQUE_SIZE,
        )

        import schema_abstraction.schema_core as sc
        sc._SCHEMA_CORE_MASK = core_mask

        results = _run_conditions(assemblies, n_trials=n_trials)
        _attach_schema_data(results)

        from .schema_analysis import run_all_schema_analysis
        from .schema_visualization import generate_all_schema_figures

        schema_results = run_all_schema_analysis(results, verbose=verbose)

        all_sweep_results[rate] = {
            "results": results,
            "schema": schema_results,
        }

        # Restore original rate
        sd.DOWNSCALE_RATE = original_rate

    return all_sweep_results


def run_replay_ablation_experiment(n_trials=3, seeds=SWEEP_SEEDS[:2], verbose=True):
    """Compare natural (fragmented) replay vs perfect replay fidelity.

    Hypothesis:
      Perfect replay preserves episodic detail but reduces abstraction.
      Natural replay generalizes better and compresses structure more.
    """
    from compare_catastrophic_forgetting import (
        run_all_conditions as _run_conditions,
        N_MEMORIES, ASSEMBLY_SIZE, MASTER_SEED,
    )

    register_schema_hooks()

    results = {
        "natural": [],
        "perfect": [],
    }

    for ablation_name, ablation_dict in [("natural", None), ("perfect", {"perfect_fidelity": True})]:
        print(f"\n{'=' * 60}", flush=True)
        print(f"REPLAY ABLATION: {ablation_name}", flush=True)
        print(f"{'=' * 60}", flush=True)

        for seed in seeds[:2]:
            ccf.MASTER_SEED = seed
            ccf.torch.manual_seed(seed)
            ccf.np.random.seed(seed)

            assemblies, core_mask = make_schema_assemblies(
                n_memories=N_MEMORIES,
                core_size=SCHEMA_CORE_SIZE,
                unique_size=UNIQUE_SIZE,
            )

            import schema_abstraction.schema_core as sc
            sc._SCHEMA_CORE_MASK = core_mask

            # Pass ablation dict to conditions
            cond_results = _run_conditions(assemblies, n_trials=n_trials,
                                            ablation=ablation_dict)
            _attach_schema_data(cond_results)

            from .schema_analysis import run_all_schema_analysis
            schema_results = run_all_schema_analysis(cond_results, verbose=verbose)
            results[ablation_name].append((cond_results, schema_results))

    return results
