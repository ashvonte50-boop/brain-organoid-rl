"""
extensions/bio_controls.py — Biological plausibility controls (Task 5).

Experiments validating that the model's behaviour is consistent with
known neuroscience constraints:

  1. Replay burst timing sensitivity
     How sensitive is retention to REPLAY_BURST_SIZE × REPLAY_BURST_GAP?
  2. Sparse vs dense replay
     Retention vs PARTIAL_CUE_SIZE (fraction of assembly cued at replay onset).
  3. Sequential replay timing
     Retention vs REPLAY_SEED_DURATION (ripple duration) and
     REPLAY_SPONTANEOUS_STEPS (post-ripple consolidation window).
  4. E/I balance perturbation
     Retention vs G_INH (inhibitory gain) — shows stable operating zone.
  5. Replay noise perturbation
     Retention vs replay noise STD (bistable → saturation transition).
  6. Replay latency (INTER_MEM_REST_STEPS)
     How long can consolidation wait before memory is unrecoverable?

All experiments run Slow+Replay only (the condition being validated).
"""
import os
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import compare_catastrophic_forgetting as cf

__all__ = [
    "run_burst_timing_sweep",
    "run_cue_sparsity_sweep",
    "run_replay_window_sweep",
    "run_ei_balance_sweep",
    "run_latency_sweep",
    "fig_bio_controls_summary",
]


# ---------------------------------------------------------------------------
# Generic worker: monkey-patch module constants, run Slow+Replay trial
# ---------------------------------------------------------------------------

def _bio_trial_worker(args):
    """
    args = (patch_dict, trial_seed)
    patch_dict: {attr_name: value} for cf module
    Returns mean retention of first N_MEMORIES-1 memories.
    """
    patch_dict, trial_seed = args
    saved = {}
    for attr, val in patch_dict.items():
        if hasattr(cf, attr):
            saved[attr] = getattr(cf, attr)
            setattr(cf, attr, val)
    try:
        assemblies = cf.make_overlapping_assemblies(cf.N_MEMORIES, cf.ASSEMBLY_SIZE, 0.20)
        result = cf.run_sequential_experiment(
            use_slow=True, use_replay=True,
            assemblies=assemblies, trial_seed=trial_seed,
            prioritize="interference_aware", verbose=False,
        )
        n = len(assemblies)
        ret = float(np.nanmean(result["final_scores"][:n - 1]))
    except Exception as e:
        warnings.warn(f"_bio_trial_worker failed ({patch_dict}): {e}")
        ret = float("nan")
    finally:
        for attr, val in saved.items():
            setattr(cf, attr, val)
    return ret


def _sweep_1d(
    patch_key: str,
    values: List,
    n_trials: int,
    verbose: bool,
    label: str,
    extra_patch: Optional[dict] = None,
) -> Tuple[List, List[float], List[float]]:
    """
    1D sweep helper. Returns (values, means, sems).
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    seeds     = [cf.MASTER_SEED + i * 37 for i in range(n_trials)]
    n_workers = min(cf.N_WORKERS, n_trials)
    means, sems = [], []

    for val in values:
        patch = {patch_key: val}
        if extra_patch:
            patch.update(extra_patch)
        task_args = [(patch, s) for s in seeds]
        try:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                trial_rets = list(pool.map(_bio_trial_worker, task_args))
        except Exception as e:
            warnings.warn(f"Parallel bio sweep failed: {e}. Serial.")
            trial_rets = [_bio_trial_worker(a) for a in task_args]
        m = float(np.nanmean(trial_rets))
        s = float(np.nanstd(trial_rets) / max(1, np.sqrt(len(trial_rets))))
        means.append(m); sems.append(s)
        if verbose:
            print(f"  [{label}={val:.3g}] mean={m:.4f} ± {s:.4f}", flush=True)

    return values, means, sems


# ---------------------------------------------------------------------------
# 1. Burst timing sensitivity
# ---------------------------------------------------------------------------

def run_burst_timing_sweep(
    burst_sizes:  Optional[List[int]]   = None,
    burst_gaps:   Optional[List[int]]   = None,
    n_trials:     int = None,
    verbose:      bool = False,
) -> dict:
    """
    Sweep REPLAY_BURST_SIZE and REPLAY_BURST_GAP independently.
    Returns dict with burst_size results and burst_gap results.
    """
    if burst_sizes is None:
        burst_sizes = [1, 2, 3, 5, 7, 10, 15]
    if burst_gaps is None:
        burst_gaps = [0, 10, 25, 50, 75, 100, 200]
    if n_trials is None:
        n_trials = cf.N_TRIALS_SWEEP

    if verbose:
        print("[BioControls] Burst size sweep ...", flush=True)
    v_sz, m_sz, s_sz = _sweep_1d("REPLAY_BURST_SIZE", burst_sizes, n_trials, verbose, "burst_size")

    if verbose:
        print("[BioControls] Burst gap sweep ...", flush=True)
    v_gp, m_gp, s_gp = _sweep_1d("REPLAY_BURST_GAP", burst_gaps, n_trials, verbose, "burst_gap")

    return {
        "burst_size": {"values": v_sz, "means": m_sz, "sems": s_sz, "default": cf.REPLAY_BURST_SIZE},
        "burst_gap":  {"values": v_gp, "means": m_gp, "sems": s_gp, "default": cf.REPLAY_BURST_GAP},
    }


# ---------------------------------------------------------------------------
# 2. Cue sparsity (PARTIAL_CUE_SIZE)
# ---------------------------------------------------------------------------

def run_cue_sparsity_sweep(
    cue_sizes:  Optional[List[int]] = None,
    n_trials:   int = None,
    verbose:    bool = False,
) -> dict:
    """
    Sweep PARTIAL_CUE_SIZE from 1 to full assembly.
    Note: PARTIAL_CUE_SIZE is a function default — monkey-patching the module
    constant affects the default argument in future calls within the same
    worker process (safe: workers are spawned fresh via multiprocessing).
    """
    if cue_sizes is None:
        cue_sizes = [1, 3, 5, 8, 10, 12, 15, 18, 20]
    if n_trials is None:
        n_trials = cf.N_TRIALS_SWEEP

    if verbose:
        print("[BioControls] Cue sparsity sweep ...", flush=True)
    v, m, s = _sweep_1d("PARTIAL_CUE_SIZE", cue_sizes, n_trials, verbose, "cue_size")

    return {"values": v, "means": m, "sems": s, "default": cf.PARTIAL_CUE_SIZE,
            "label": "Partial Cue Size (neurons)"}


# ---------------------------------------------------------------------------
# 3. Replay window sweep
# ---------------------------------------------------------------------------

def run_replay_window_sweep(
    seed_durations:    Optional[List[int]] = None,
    spont_steps_vals:  Optional[List[int]] = None,
    n_trials:          int = None,
    verbose:           bool = False,
) -> dict:
    """
    Sweep REPLAY_SEED_DURATION (ripple trigger duration)
    and REPLAY_SPONTANEOUS_STEPS (post-seed window).
    """
    if seed_durations is None:
        seed_durations = [5, 10, 15, 20, 30, 50]
    if spont_steps_vals is None:
        spont_steps_vals = [20, 50, 75, 100, 150, 200, 300]
    if n_trials is None:
        n_trials = cf.N_TRIALS_SWEEP

    if verbose:
        print("[BioControls] Seed duration sweep ...", flush=True)
    v_sd, m_sd, s_sd = _sweep_1d(
        "REPLAY_SEED_DURATION", seed_durations, n_trials, verbose, "seed_dur"
    )

    if verbose:
        print("[BioControls] Spontaneous steps sweep ...", flush=True)
    v_sp, m_sp, s_sp = _sweep_1d(
        "REPLAY_SPONTANEOUS_STEPS", spont_steps_vals, n_trials, verbose, "spont_steps"
    )

    return {
        "seed_duration": {
            "values": v_sd, "means": m_sd, "sems": s_sd,
            "default": cf.REPLAY_SEED_DURATION,
            "label": "Replay Seed Duration (steps)",
        },
        "spont_steps": {
            "values": v_sp, "means": m_sp, "sems": s_sp,
            "default": cf.REPLAY_SPONTANEOUS_STEPS,
            "label": "Spontaneous Post-Seed Steps",
        },
    }


# ---------------------------------------------------------------------------
# 4. E/I balance perturbation
# ---------------------------------------------------------------------------

def _ei_trial_worker(args):
    """Worker using modified G_INH via network rebuild."""
    g_inh_val, trial_seed = args
    try:
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        assemblies = cf.make_overlapping_assemblies(cf.N_MEMORIES, cf.ASSEMBLY_SIZE, 0.20)
        # Build network with modified G_INH
        from neuron_models.izhikevich_network import IzhikevichNetwork
        net = IzhikevichNetwork(
            n_neurons=cf.N_NEURONS, n_inh=cf.N_INH,
            g_exc=cf.G_EXC, g_inh=g_inh_val,
            noise_std=cf.NOISE_STD, dt=cf.DT, device=cf.DEVICE
        ).to(cf.DEVICE)
        net.init_stdp(
            A_plus=cf.A_PLUS, A_minus=cf.A_MINUS,
            tau_plus=cf.TAU_PLUS, tau_minus=cf.TAU_MINUS, w_max=cf.W_MAX
        )
        net.init_slow_weights(
            gamma=cf.GAMMA, tau_slow=cf.TAU_SLOW,
            tau_fast=cf.FAST_DECAY_TAU, tau_very_slow=cf.TAU_VERY_SLOW
        )
        tags = cf.SynapticTags()

        # Run through the sequence manually (same protocol as run_sequential_experiment)
        n_mem = len(assemblies)
        all_replay = []
        for j, asm in enumerate(assemblies):
            cf.train_one_memory(net, asm, tags=tags, n_presentations=cf._N_PRESENTATIONS)
            if j > 0:
                cf.apply_competitive_interference(net, asm, assemblies[:j])
            if j < n_mem - 1:
                cs = [cf.probe_memory(net, assemblies[i])["isyn_score"] for i in range(j+1)]
                cf.inter_memory_rest_with_replay(
                    net, assemblies[:j+1], current_scores=cs,
                    prioritize="interference_aware", tags=tags, rest_id=j,
                    accumulated_metrics=all_replay, ablation=None,
                )
        final = np.array([cf.probe_memory(net, assemblies[i])["isyn_score"] for i in range(n_mem)])
        ret   = float(np.nanmean(final[:n_mem-1]))
    except Exception as e:
        warnings.warn(f"_ei_trial_worker failed (g_inh={g_inh_val}): {e}")
        ret = float("nan")
    return ret


def run_ei_balance_sweep(
    g_inh_values: Optional[List[float]] = None,
    n_trials: int = None,
    verbose: bool = False,
) -> dict:
    """
    Sweep G_INH (inhibitory conductance) from near-zero to very strong.
    G_INH is set directly in IzhikevichNetwork so we use a custom worker.
    Production value: G_INH = -40.0 (negative = inhibitory).
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    if g_inh_values is None:
        # Absolute values — sign is applied inside the network
        g_inh_values = [-10.0, -20.0, -30.0, -40.0, -50.0, -60.0, -80.0]
    if n_trials is None:
        n_trials = cf.N_TRIALS_SWEEP

    seeds = [cf.MASTER_SEED + i * 37 for i in range(n_trials)]
    means, sems = [], []

    for g_inh in g_inh_values:
        task_args = [(g_inh, s) for s in seeds]
        n_workers = min(cf.N_WORKERS, n_trials)
        try:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                trial_rets = list(pool.map(_ei_trial_worker, task_args))
        except Exception as e:
            warnings.warn(f"E/I sweep parallel failed: {e}. Serial.")
            trial_rets = [_ei_trial_worker(a) for a in task_args]
        m = float(np.nanmean(trial_rets))
        s = float(np.nanstd(trial_rets) / max(1, np.sqrt(len(trial_rets))))
        means.append(m); sems.append(s)
        if verbose:
            print(f"  [EI] G_INH={g_inh:.1f}: mean={m:.4f} ± {s:.4f}", flush=True)

    return {
        "values": g_inh_values, "means": means, "sems": sems,
        "default": cf.G_INH,
        "label": "Inhibitory Gain G_INH",
    }


# ---------------------------------------------------------------------------
# 5. Replay latency (INTER_MEM_REST_STEPS)
# ---------------------------------------------------------------------------

def run_latency_sweep(
    rest_steps_values: Optional[List[int]] = None,
    n_trials: int = None,
    verbose: bool = False,
) -> dict:
    """
    Sweep INTER_MEM_REST_STEPS — how long the consolidation window can wait.
    """
    if rest_steps_values is None:
        rest_steps_values = [250, 500, 1000, 1500, 2500, 4000, 6000, 10000]
    if n_trials is None:
        n_trials = cf.N_TRIALS_SWEEP

    if verbose:
        print("[BioControls] Latency (rest steps) sweep ...", flush=True)
    v, m, s = _sweep_1d(
        "INTER_MEM_REST_STEPS", rest_steps_values, n_trials, verbose, "rest_steps"
    )
    return {
        "values": v, "means": m, "sems": s,
        "default": cf.INTER_MEM_REST_STEPS,
        "label": "Inter-Memory Rest Steps",
    }


# ---------------------------------------------------------------------------
# Figure: 6-panel bio controls summary
# ---------------------------------------------------------------------------

def fig_bio_controls_summary(
    burst_data:    Optional[dict] = None,
    cue_data:      Optional[dict] = None,
    window_data:   Optional[dict] = None,
    ei_data:       Optional[dict] = None,
    latency_data:  Optional[dict] = None,
    out_dir:       str = ".",
) -> None:
    """
    6-panel summary figure for biological plausibility controls.
    Missing panels (None) are skipped (gray placeholder).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    def _plot_1d(ax, data, title, xlabel, default_val=None, color="#3498db"):
        if data is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="gray")
            ax.set_title(title, fontsize=9)
            return
        v, m, s = data["values"], data["means"], data["sems"]
        ax.plot(v, m, "o-", color=color, linewidth=2, markersize=6)
        ax.fill_between(v,
                        [mi - si for mi, si in zip(m, s)],
                        [mi + si for mi, si in zip(m, s)],
                        alpha=0.25, color=color)
        ax.set_xlabel(xlabel or data.get("label", ""), fontsize=8)
        ax.set_ylabel("Mean Retention (A/B/C)", fontsize=8)
        ax.set_title(title, fontsize=9)
        if default_val is not None:
            ax.axvline(default_val, color="black", linestyle="--", linewidth=1.2,
                       label="Production")
            ax.legend(fontsize=7)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    panel_args = [
        (burst_data["burst_size"] if burst_data else None,
         "Burst Size Sensitivity", "Burst Size (events/burst)",
         cf.REPLAY_BURST_SIZE, "#3498db"),
        (burst_data["burst_gap"] if burst_data else None,
         "Burst Gap Sensitivity", "Burst Gap (steps)",
         cf.REPLAY_BURST_GAP, "#9b59b6"),
        (cue_data, "Cue Sparsity", "Partial Cue Size (neurons)",
         cf.PARTIAL_CUE_SIZE, "#e67e22"),
        (window_data["seed_duration"] if window_data else None,
         "Replay Seed Duration", "Seed Duration (steps)",
         cf.REPLAY_SEED_DURATION, "#2ecc71"),
        (ei_data, "E/I Balance", "Inhibitory Gain G_INH",
         cf.G_INH, "#e74c3c"),
        (latency_data, "Consolidation Latency", "Rest Steps Before Replay",
         cf.INTER_MEM_REST_STEPS, "#1abc9c"),
    ]

    for ax, (data, title, xlabel, default_val, color) in zip(axes, panel_args):
        _plot_1d(ax, data, title, xlabel, default_val, color)

    fig.suptitle(
        "Biological Plausibility Controls (Slow+Replay)\n"
        "Dashed line = production operating point",
        fontsize=11, y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    cf._save_fig(fig, "bio_controls_summary")
    plt.close(fig)
    print("[FIG] Saved bio_controls_summary.png", flush=True)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os as _os, sys as _sys, pathlib as _pl
    _os.environ.setdefault("PYTHONUNBUFFERED", "1")
    # Ensure project root (parent of extensions/) is on sys.path
    _root = str(_pl.Path(__file__).resolve().parent.parent)
    if _root not in _sys.path:
        _sys.path.insert(0, _root)

    print("[bio_controls self-test] cue sparsity sweep (3 vals, 2 trials) ...", flush=True)
    cue = run_cue_sparsity_sweep(cue_sizes=[3, 8, 15], n_trials=2, verbose=True)
    print(f"  Cue sparsity: {list(zip(cue['values'], [f'{m:.3f}' for m in cue['means']]))} ")

    print("[bio_controls self-test] latency sweep (3 vals, 2 trials) ...", flush=True)
    lat = run_latency_sweep(rest_steps_values=[500, 2500, 6000], n_trials=2, verbose=True)
    print(f"  Latency: {list(zip(lat['values'], [f'{m:.3f}' for m in lat['means']]))}")
    print("[bio_controls self-test] DONE.", flush=True)
