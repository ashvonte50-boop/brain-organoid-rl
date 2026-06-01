"""
extensions/robustness.py — Robustness and sensitivity sweeps (Task 2).

Systematic parameter sweeps to prove the model is NOT narrowly tuned.
Each sweep varies one parameter while holding all others at production defaults.

Parameters swept:
  1. replay_coherence_thr   (REPLAY_COHERENCE_THR)
  2. pers_gain              (REPLAY_PERS_GAIN)
  3. n_replay_events        (_N_REPLAY_EVENTS)
  4. overlap_frac           (assembly overlap fraction)
  5. gamma                  (GAMMA — slow weight fraction)
  6. fast_decay_tau         (FAST_DECAY_TAU)
  7. replay_burst_size      (REPLAY_BURST_SIZE)
  8. assembly_size          (ASSEMBLY_SIZE)
  9. noise_std              (NOISE_STD — training noise)
  10. tag_capture_rate      (TAG_CAPTURE_RATE)

For each (parameter, value), runs N_TRIALS_SWEEP trials of Slow+Replay using a
parametric experiment function that monkey-patches module constants AFTER spawn
(safe: worker processes are isolated via spawn).

Note: Parameters embedded in function signature defaults (REPLAY_NOISE_STD,
PARTIAL_CUE_SIZE, REPLAY_SEED_STRENGTH, REPLAY_SEED_DURATION,
REPLAY_SPONTANEOUS_STEPS) require a custom parametric function because default
arguments are captured at import time, not call time.  Those are handled via
the custom _parametric_experiment() pathway.

Output:
  run_sensitivity_sweep()   → dict[param_name, list[trial_results]]
  fig_robustness_heatmap()  → saves robustness_heatmap.png
  fig_sensitivity_rankings()→ saves sensitivity_rankings.png
"""
import os
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import compare_catastrophic_forgetting as cf

__all__ = [
    "SWEEP_CONFIGS",
    "run_param_sweep",
    "run_full_sensitivity",
    "fig_robustness_heatmap",
    "fig_sensitivity_rankings",
]

# ---------------------------------------------------------------------------
# Sweep configurations
# Each entry: (param_name, display_label, values, description)
# ---------------------------------------------------------------------------

SWEEP_CONFIGS: List[dict] = [
    {
        "key":   "coherence_thr",
        "label": "Coherence\nThreshold",
        "unit":  "",
        "values": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        "default": cf.REPLAY_COHERENCE_THR,
        "patch": "REPLAY_COHERENCE_THR",
        "mode":  "module",
    },
    {
        "key":   "pers_gain",
        "label": "Persistence\nGain",
        "unit":  "",
        "values": [0.0, 0.10, 0.20, 0.30, 0.45, 0.60],
        "default": cf.REPLAY_PERS_GAIN,
        "patch": "REPLAY_PERS_GAIN",
        "mode":  "module",
    },
    {
        "key":   "n_replay_events",
        "label": "Replay\nEvents/Rest",
        "unit":  "",
        "values": [5, 10, 15, 20, 25, 35, 50],
        "default": cf._N_REPLAY_EVENTS,
        "patch": "_N_REPLAY_EVENTS",
        "mode":  "module",
    },
    {
        "key":   "overlap_frac",
        "label": "Overlap\nFraction",
        "unit":  "",
        "values": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
        "default": 0.20,
        "patch": None,
        "mode":  "overlap",   # special: rebuild assemblies
    },
    {
        "key":   "gamma",
        "label": "Gamma\n(Slow Fraction)",
        "unit":  "",
        "values": [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.75],
        "default": cf.GAMMA,
        "patch": "GAMMA",
        "mode":  "network",   # needs rebuild network with new gamma
    },
    {
        "key":   "fast_decay_tau",
        "label": "Fast Decay\nTau (steps)",
        "unit":  "",
        "values": [500, 750, 1000, 1500, 2000, 3000, 5000],
        "default": cf.FAST_DECAY_TAU,
        "patch": "FAST_DECAY_TAU",
        "mode":  "module",
    },
    {
        "key":   "burst_size",
        "label": "Burst\nSize",
        "unit":  "",
        "values": [1, 2, 3, 5, 7, 10],
        "default": cf.REPLAY_BURST_SIZE,
        "patch": "REPLAY_BURST_SIZE",
        "mode":  "module",
    },
    {
        "key":   "assembly_size",
        "label": "Assembly\nSize",
        "unit":  "neurons",
        "values": [8, 10, 15, 20, 25, 30],
        "default": cf.ASSEMBLY_SIZE,
        "patch": None,
        "mode":  "assembly",  # needs rebuild assemblies
    },
    {
        "key":   "replay_noise",
        "label": "Replay\nNoise STD",
        "unit":  "",
        "values": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
        "default": cf.REPLAY_NOISE_STD,
        "patch": "REPLAY_NOISE_STD",
        "mode":  "funcdefault",  # embedded in function default — use parametric
    },
    {
        "key":   "tag_capture_rate",
        "label": "Tag Capture\nRate",
        "unit":  "",
        "values": [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25],
        "default": cf.TAG_CAPTURE_RATE,
        "patch": "TAG_CAPTURE_RATE",
        "mode":  "module",
    },
]


# ---------------------------------------------------------------------------
# Parametric single-trial function (handles funcdefault params)
# ---------------------------------------------------------------------------

def _parametric_trial_slow_replay(
    assemblies: List[np.ndarray],
    trial_seed: int,
    replay_noise: float = None,
) -> float:
    """
    Custom trial runner for parameters embedded in function defaults.
    Returns mean retention of first N_MEMORIES-1 memories.
    Only replay_noise requires this pathway currently.
    """
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)

    rng_noise = replay_noise if replay_noise is not None else cf.REPLAY_NOISE_STD
    n_mem = len(assemblies)
    net   = cf.build_network(use_slow=True)
    tags  = cf.SynapticTags() if cf.USE_TAGGING else None

    final_scores = np.full(n_mem, np.nan)
    current_scores = []
    all_replay_metrics = []

    for j, asm in enumerate(assemblies):
        cf.train_one_memory(net, asm, tags=tags, n_presentations=cf._N_PRESENTATIONS)
        if j > 0:
            cf.apply_competitive_interference(net, asm, assemblies[:j])
        if cf._N_REPLAY_EVENTS > 0 and j < n_mem - 1:
            current_scores = [
                cf.probe_memory(net, assemblies[i])["isyn_score"]
                for i in range(j + 1)
            ]

        if j < n_mem - 1:
            # Custom rest: replay with modified noise
            _parametric_rest_with_replay(
                net, assemblies[:j+1], current_scores, tags=tags,
                rest_id=j, replay_noise=rng_noise,
                accumulated_metrics=all_replay_metrics,
            )

    final_scores = np.array([
        cf.probe_memory(net, assemblies[i])["isyn_score"]
        for i in range(n_mem)
    ])
    return float(np.nanmean(final_scores[:n_mem - 1]))


def _parametric_rest_with_replay(
    net, learned_assemblies, current_scores, tags, rest_id, replay_noise, accumulated_metrics
):
    """
    Replica of cf.inter_memory_rest_with_replay but with overridable replay_noise.
    Only the noise parameter differs from the production function.
    """
    n_steps  = cf.INTER_MEM_REST_STEPS
    n_events = cf._N_REPLAY_EVENTS
    prioritize = "interference_aware"

    half_n = n_steps // 2
    rest_n = n_steps - half_n

    def _bulk_decay(num_steps):
        f = 1.0 - float(np.exp(-num_steps / cf.FAST_DECAY_TAU))
        with torch.no_grad():
            W    = net.W.data[:cf.N_EXC, :cf.N_EXC]
            base = net.W_init[:cf.N_EXC, :cf.N_EXC]
            net.W.data[:cf.N_EXC, :cf.N_EXC] = W + (base - W) * f

    _bulk_decay(half_n)
    cf.bulk_slow_step(net, half_n)
    if tags is not None:
        tags.decay(n_steps=half_n)

    # Compute base priorities
    _base_probs = cf._replay_priorities(learned_assemblies, current_scores, prioritize)

    n_bursts  = max(1, n_events // cf.REPLAY_BURST_SIZE)
    base_sz   = n_events // n_bursts
    remainder = n_events - base_sz * n_bursts

    for b in range(n_bursts):
        burst_sz = base_sz + (remainder if b == n_bursts - 1 else 0)
        indices  = np.random.choice(len(learned_assemblies), size=burst_sz, p=_base_probs)
        for idx in indices:
            # Use custom _replay_one_event_with_noise
            _replay_one_event_parametric(
                net, learned_assemblies[idx], tags=tags,
                all_assemblies=learned_assemblies, replay_noise=replay_noise,
            )
        if b < n_bursts - 1:
            _bulk_decay(cf.REPLAY_BURST_GAP)
            cf.bulk_slow_step(net, cf.REPLAY_BURST_GAP)
            if tags is not None:
                tags.decay(n_steps=cf.REPLAY_BURST_GAP)

    _bulk_decay(rest_n)
    cf.bulk_slow_step(net, rest_n)
    if tags is not None:
        tags.decay(n_steps=rest_n)


def _replay_one_event_parametric(
    net, assembly, tags, all_assemblies, replay_noise
):
    """Stripped replay event with overridable noise (no metrics collection)."""
    use_slow = getattr(net, 'slow_enabled', False)

    # Build cue pool (overlap-exclusion)
    if all_assemblies is not None and len(all_assemblies) > 1:
        other_set = set()
        for other in all_assemblies:
            if other is not assembly:
                other_set.update(other.tolist())
        unique = np.array([n for n in assembly if n not in other_set])
        cue_pool = unique if len(unique) >= cf.PARTIAL_CUE_SIZE else assembly
    else:
        cue_pool = assembly
    cue_n = np.random.choice(cue_pool, size=min(cf.PARTIAL_CUE_SIZE, len(cue_pool)), replace=False)
    seed_stim = torch.zeros(cf.N_NEURONS, device=cf.DEVICE)
    seed_stim[cue_n] = cf.REPLAY_SEED_STRENGTH

    orig_noise    = net.noise_std
    net.noise_std = replay_noise

    target_excs = assembly[assembly < cf.N_EXC]
    target_mask = torch.zeros(cf.N_EXC, device=cf.DEVICE, dtype=torch.bool)
    if len(target_excs) > 0:
        target_mask[target_excs] = True
    off_mask  = ~target_mask
    n_target  = max(1, int(target_mask.sum().item()))
    n_off     = max(1, cf.N_EXC - n_target)
    activity  = torch.zeros(cf.N_EXC, device=cf.DEVICE)
    COH_THR   = cf.REPLAY_COHERENCE_THR
    LAMBDA    = cf.REPLAY_COHERENCE_LAMBDA

    def _coh(spk_exc):
        activity.mul_(cf.REPLAY_COHERENCE_DECAY).add_(spk_exc)
        active  = activity > cf.REPLAY_COHERENCE_ACTIVE_THR
        t_rate  = float((active & target_mask).sum().item()) / n_target
        o_rate  = float((active & off_mask).sum().item()) / n_off
        return t_rate / (t_rate + LAMBDA * o_rate + 1e-6)

    _spike_thr = int(cf.REPLAY_SPIKE_FRACTION_MAX * cf.N_NEURONS)

    with torch.no_grad():
        for _ in range(cf.REPLAY_SEED_DURATION):
            net.forward(seed_stim)
            if int(net.spikes.sum().item()) > _spike_thr:
                activity.mul_(cf.REPLAY_COHERENCE_DECAY).add_(net.spikes[:cf.N_EXC].float())
                if use_slow: net.slow_step()
                continue
            if _coh(net.spikes[:cf.N_EXC].float()) > COH_THR:
                if tags is not None: tags.update_from_spikes(net)
                net.stdp_step()
            if use_slow: net.slow_step()
        activity.zero_()
        zero_stim = torch.zeros(cf.N_NEURONS, device=cf.DEVICE)
        _consec = 0; _unlocked = False
        for _ in range(cf.REPLAY_SPONTANEOUS_STEPS):
            net.forward(zero_stim)
            coh = _coh(net.spikes[:cf.N_EXC].float())
            if coh > COH_THR:
                _consec += 1
                if not _unlocked and _consec >= cf.REPLAY_ACCEPT_MIN_CONSEC:
                    _unlocked = True
            else:
                _consec = 0
            if _unlocked and coh > COH_THR:
                if tags is not None: tags.update_from_spikes(net)
                net.stdp_step()
            if use_slow: net.slow_step()
        if tags is not None and use_slow:
            tags.tag_driven_consolidation(net, assembly)

    net.noise_std = orig_noise


# ---------------------------------------------------------------------------
# Worker functions (top-level, picklable)
# ---------------------------------------------------------------------------

def _sweep_trial_worker(args):
    """
    Generic worker. args = (trial_seed, assemblies, patch_dict)
    patch_dict = {"attr": "module_attr_name", "val": new_value}
       or       {"attr": None, "overlap": float}  for overlap rebuild
       or       {"attr": None, "asm_size": int}   for assembly rebuild
       or       {"attr": "REPLAY_NOISE_STD", "val": float, "funcdefault": True}
    """
    trial_seed, assemblies, patch = args

    # Handle special modes
    mode = patch.get("mode", "module")

    if mode == "overlap":
        assemblies = cf.make_overlapping_assemblies(cf.N_MEMORIES, cf.ASSEMBLY_SIZE, patch["val"])
    elif mode == "assembly":
        assemblies = cf.make_overlapping_assemblies(cf.N_MEMORIES, patch["val"], 0.20)
    elif mode == "funcdefault":
        # replay_noise sweep — use parametric function
        ret = _parametric_trial_slow_replay(assemblies, trial_seed, replay_noise=patch["val"])
        return {"mean_retention": ret, "trial_seed": trial_seed, "patch": patch}

    # Monkey-patch module constant
    if "attr" in patch and patch["attr"] is not None:
        old_val = getattr(cf, patch["attr"])
        setattr(cf, patch["attr"], patch["val"])

    if mode == "network":
        # gamma change requires re-initialising slow weights; handled via
        # run_sequential_experiment which calls build_network → net.init_slow_weights
        pass

    try:
        result = cf.run_sequential_experiment(
            use_slow=True, use_replay=True,
            assemblies=assemblies,
            trial_seed=trial_seed,
            prioritize="interference_aware",
            verbose=False,
        )
        n_mem = len(assemblies)
        ret = float(np.nanmean(result["final_scores"][:n_mem - 1]))
    except Exception as e:
        warnings.warn(f"_sweep_trial_worker failed: {e}")
        ret = float("nan")
    finally:
        if "attr" in patch and patch["attr"] is not None:
            setattr(cf, patch["attr"], old_val)

    return {"mean_retention": ret, "trial_seed": trial_seed, "patch": patch}


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_param_sweep(
    cfg: dict,
    assemblies: List[np.ndarray],
    n_trials: int = None,
    verbose: bool = False,
) -> Tuple[List[float], List[List[float]]]:
    """
    Run one parameter sweep (one entry from SWEEP_CONFIGS).
    Returns (values, per_value_trial_results) where each inner list is n_trials floats.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    if n_trials is None:
        n_trials = cf.N_TRIALS_SWEEP

    values  = cfg["values"]
    seeds   = [cf.MASTER_SEED + i * 37 for i in range(n_trials)]
    mode    = cfg["mode"]
    asms    = [a.copy() for a in assemblies]

    all_results = []

    for val in values:
        patch = {"mode": mode, "val": val}
        if mode == "module":
            patch["attr"] = cfg["patch"]
        elif mode == "funcdefault":
            patch["attr"] = cfg["patch"]
        elif mode == "network":
            patch["attr"] = cfg["patch"]
        elif mode == "overlap":
            pass
        elif mode == "assembly":
            pass

        task_args = [(s, asms, patch) for s in seeds]
        n_workers = min(cf.N_WORKERS, n_trials)
        try:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                trial_rets = [r["mean_retention"] for r in pool.map(_sweep_trial_worker, task_args)]
        except Exception as e:
            warnings.warn(f"Parallel sweep failed: {e}. Serial fallback.")
            trial_rets = [_sweep_trial_worker(a)["mean_retention"] for a in task_args]

        all_results.append(trial_rets)
        if verbose:
            m = np.nanmean(trial_rets)
            print(f"  [{cfg['key']}={val:.3g}] mean_ret={m:.4f}", flush=True)

    return values, all_results


def run_full_sensitivity(
    assemblies: List[np.ndarray],
    n_trials: int = None,
    verbose: bool = False,
    params: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """
    Run all (or a subset of) parameter sweeps.
    params: if provided, list of sweep keys to run (e.g. ["coherence_thr", "gamma"])
    Returns dict[key] = {"values": list, "results": list[list[float]], "config": dict}
    """
    if n_trials is None:
        n_trials = cf.N_TRIALS_SWEEP

    sweeps = [c for c in SWEEP_CONFIGS if (params is None or c["key"] in params)]
    out    = {}

    for cfg_s in sweeps:
        if verbose:
            print(f"\n[Robustness] Sweeping {cfg_s['key']} ({len(cfg_s['values'])} values, "
                  f"{n_trials} trials each) ...", flush=True)
        vals, results = run_param_sweep(cfg_s, assemblies, n_trials=n_trials, verbose=verbose)
        out[cfg_s["key"]] = {
            "values":  vals,
            "results": results,
            "config":  cfg_s,
        }

    return out


# ---------------------------------------------------------------------------
# Sensitivity metric: normalized range
# ---------------------------------------------------------------------------

def _sensitivity_score(trial_results: List[List[float]]) -> float:
    """
    Sensitivity = range of per-value medians / mean of all medians.
    Higher = more sensitive parameter.
    """
    medians = [np.nanmedian(r) for r in trial_results]
    valid   = [m for m in medians if np.isfinite(m)]
    if len(valid) < 2:
        return float("nan")
    base = float(np.mean(valid)) if np.mean(valid) != 0 else 1e-6
    return (max(valid) - min(valid)) / abs(base)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_robustness_heatmap(
    sweep_data: Dict[str, dict],
    out_dir: str = ".",
) -> None:
    """
    Heatmap: parameter value (x-axis) × parameter name (y-axis).
    Color = mean retention.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys   = list(sweep_data.keys())
    n_params = len(keys)
    if n_params == 0:
        return

    # Normalise each row to the production default
    # Max values per param (for consistent color scale within each row)
    max_vals = max(len(sweep_data[k]["values"]) for k in keys)
    hm  = np.full((n_params, max_vals), np.nan)
    xlabs = [None] * n_params

    for i, key in enumerate(keys):
        vals    = sweep_data[key]["values"]
        results = sweep_data[key]["results"]
        default = sweep_data[key]["config"]["default"]
        medians = [np.nanmedian(r) for r in results]
        for j, (v, m) in enumerate(zip(vals, medians)):
            hm[i, j] = m
        xlabs[i] = [f"{v:.3g}" + ("*" if v == default else "") for v in vals]

    # Separate figure per param for clarity (or combined if few)
    fig, ax = plt.subplots(figsize=(max(8, max_vals * 1.2), max(4, n_params * 0.8)))
    im = ax.imshow(hm, aspect="auto", cmap="RdYlGn")
    ax.set_yticks(range(n_params))
    ax.set_yticklabels([sweep_data[k]["config"]["label"].replace("\n", " ") for k in keys],
                        fontsize=8)
    ax.set_xlabel("Parameter Value (* = production default)", fontsize=9)
    ax.set_title("Parameter Robustness: Mean A/B/C Retention\n(Slow+Replay condition)",
                 fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.02, label="Mean Retention")

    # Overlay text values
    for i in range(n_params):
        for j in range(max_vals):
            if np.isfinite(hm[i, j]):
                ax.text(j, i, f"{hm[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color="black")

    fig.tight_layout()
    cf._save_fig(fig, "robustness_heatmap")
    plt.close(fig)
    print("[FIG] Saved robustness_heatmap.png", flush=True)


def fig_sensitivity_rankings(
    sweep_data: Dict[str, dict],
    out_dir: str = ".",
) -> None:
    """
    Horizontal bar chart: parameters ranked by sensitivity score.
    Includes error bars (IQR of medians across sweep values).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from extensions.stats_utils import bootstrap_ci

    keys = list(sweep_data.keys())
    if not keys:
        return

    sens_scores = {}
    for key in keys:
        results = sweep_data[key]["results"]
        sens_scores[key] = _sensitivity_score(results)

    sorted_keys   = sorted(keys, key=lambda k: sens_scores.get(k, 0), reverse=True)
    sorted_scores = [sens_scores.get(k, np.nan) for k in sorted_keys]
    sorted_labels = [sweep_data[k]["config"]["label"].replace("\n", " ") for k in sorted_keys]
    colors        = ["#e74c3c" if s > 0.5 else "#3498db" for s in sorted_scores]

    fig, ax = plt.subplots(figsize=(8, max(4, len(keys) * 0.6)))
    y = np.arange(len(sorted_keys))
    bars = ax.barh(y, sorted_scores, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_labels, fontsize=9)
    ax.set_xlabel("Sensitivity Score\n(normalized range of medians)", fontsize=9)
    ax.set_title("Parameter Sensitivity Ranking\n(higher = more sensitive)", fontsize=10)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.5, label="High sensitivity")
    ax.axvline(0.2, color="green", linestyle="--", linewidth=1, alpha=0.5, label="Robust zone")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    cf._save_fig(fig, "sensitivity_rankings")
    plt.close(fig)
    print("[FIG] Saved sensitivity_rankings.png", flush=True)


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

    asms = cf.make_overlapping_assemblies(cf.N_MEMORIES, cf.ASSEMBLY_SIZE, 0.20)

    # Quick self-test: sweep coherence threshold only, 2 trials
    cfg_test = next(c for c in SWEEP_CONFIGS if c["key"] == "coherence_thr")
    cfg_test = dict(cfg_test)
    cfg_test["values"] = [0.30, 0.50, 0.70]  # only 3 values for speed

    print("[robustness self-test] coherence_thr sweep ...", flush=True)
    vals, results = run_param_sweep(cfg_test, asms, n_trials=2, verbose=True)
    print(f"  Values: {vals}")
    print(f"  Means:  {[np.nanmean(r) for r in results]}")
    print("[robustness self-test] DONE.", flush=True)
