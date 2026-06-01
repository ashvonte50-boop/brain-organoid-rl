"""
extensions/efficiency.py — Computational efficiency analysis (Task 8).

Measures the replay overhead vs retention benefit, producing:
  1. Retention per replay event (efficiency curve vs N_REPLAY_EVENTS)
  2. Wall-time vs retention Pareto plot (time × retention tradeoff)
  3. Mechanism overhead breakdown (time fraction per mechanism)
  4. Retention per compute unit (normalized efficiency index)

Goal: prove the model is computationally frugal — the replay overhead
buys proportionally large retention improvements.
"""
import os
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import compare_catastrophic_forgetting as cf

__all__ = [
    "measure_retention_per_event",
    "measure_time_breakdown",
    "run_efficiency_sweep",
    "fig_efficiency_curves",
    "fig_efficiency_pareto",
]


# ---------------------------------------------------------------------------
# Retention per replay event
# ---------------------------------------------------------------------------

def _efficiency_worker(args):
    """(n_events, trial_seed) → (mean_ret, elapsed_s)"""
    n_events, trial_seed = args
    old_events = cf._N_REPLAY_EVENTS
    try:
        cf._N_REPLAY_EVENTS = n_events
        assemblies = cf.make_overlapping_assemblies(cf.N_MEMORIES, cf.ASSEMBLY_SIZE, 0.20)
        t0 = time.perf_counter()
        result = cf.run_sequential_experiment(
            use_slow=True, use_replay=True,
            assemblies=assemblies, trial_seed=trial_seed,
            prioritize="interference_aware", verbose=False,
        )
        elapsed = time.perf_counter() - t0
        n = len(assemblies)
        ret = float(np.nanmean(result["final_scores"][:n-1]))
    except Exception as e:
        warnings.warn(f"efficiency_worker failed (n_ev={n_events}): {e}")
        ret = float("nan"); elapsed = float("nan")
    finally:
        cf._N_REPLAY_EVENTS = old_events
    return ret, elapsed


def measure_retention_per_event(
    n_events_values: Optional[List[int]] = None,
    n_trials: int = None,
    verbose: bool = False,
) -> dict:
    """
    Sweep n_replay_events, measure mean retention and elapsed time.
    Computes retention-per-event = retention / n_events.
    Returns dict with values, rets, times, efficiency_indices.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    if n_events_values is None:
        n_events_values = [1, 3, 5, 10, 15, 20, 25, 35, 50, 70]
    if n_trials is None:
        n_trials = cf.N_TRIALS_SWEEP

    seeds     = [cf.MASTER_SEED + i * 37 for i in range(n_trials)]
    n_workers = min(cf.N_WORKERS, n_trials)

    rets, times, efficiencies = [], [], []

    for ne in n_events_values:
        task_args = [(ne, s) for s in seeds]
        try:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                trial_results = list(pool.map(_efficiency_worker, task_args))
        except Exception as e:
            warnings.warn(f"Efficiency sweep parallel failed: {e}. Serial.")
            trial_results = [_efficiency_worker(a) for a in task_args]

        r_mean = float(np.nanmean([r for r, _ in trial_results]))
        t_mean = float(np.nanmean([t for _, t in trial_results]))
        eff    = r_mean / max(ne, 1)  # retention per replay event
        rets.append(r_mean)
        times.append(t_mean)
        efficiencies.append(eff)

        if verbose:
            print(f"  [Efficiency] n_ev={ne:3d}: ret={r_mean:.4f}, "
                  f"time={t_mean:.1f}s, eff={eff:.5f}", flush=True)

    # No-replay baseline (0 events = fast only)
    fast_ret = float("nan")
    try:
        task_args_0 = [(False, False, s) for s in seeds]
        fast_results = []
        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            asms = cf.make_overlapping_assemblies(cf.N_MEMORIES, cf.ASSEMBLY_SIZE, 0.20)
            r    = cf.run_sequential_experiment(
                use_slow=True, use_replay=False,
                assemblies=asms, trial_seed=s,
                prioritize="interference_aware", verbose=False,
            )
            fast_results.append(float(np.nanmean(r["final_scores"][:cf.N_MEMORIES-1])))
        fast_ret = float(np.nanmean(fast_results))
    except Exception:
        pass

    return {
        "n_events_values": n_events_values,
        "mean_rets":       rets,
        "mean_times":      times,
        "efficiency_idx":  efficiencies,
        "no_replay_ret":   fast_ret,
        "production_n_events": cf._N_REPLAY_EVENTS,
    }


# ---------------------------------------------------------------------------
# Wall-time breakdown per mechanism
# ---------------------------------------------------------------------------

def measure_time_breakdown(trial_seed: int = cf.MASTER_SEED) -> dict:
    """
    Run a single Slow+Replay trial with timing instrumentation.
    Returns fraction of total time per mechanism.
    """
    # Reset timers
    for k in cf._TIMER:
        cf._TIMER[k] = 0.0

    assemblies = cf.make_overlapping_assemblies(cf.N_MEMORIES, cf.ASSEMBLY_SIZE, 0.20)
    t0 = time.perf_counter()
    cf.run_sequential_experiment(
        use_slow=True, use_replay=True,
        assemblies=assemblies, trial_seed=trial_seed,
        prioritize="interference_aware", verbose=False,
    )
    total = time.perf_counter() - t0

    timer_snapshot = dict(cf._TIMER)
    timer_sum      = sum(timer_snapshot.values())
    fractions      = {k: v / max(timer_sum, 1e-9) for k, v in timer_snapshot.items()}
    return {
        "absolute_s":   timer_snapshot,
        "fractions":    fractions,
        "total_s":      total,
        "timer_sum_s":  timer_sum,
    }


# ---------------------------------------------------------------------------
# Full efficiency sweep (combines the above)
# ---------------------------------------------------------------------------

def run_efficiency_sweep(
    n_trials: int = None,
    verbose: bool = False,
) -> dict:
    """
    Master efficiency analysis: retention-per-event + time breakdown.
    """
    if n_trials is None:
        n_trials = cf.N_TRIALS_SWEEP

    if verbose:
        print("[Efficiency] Retention-per-event sweep ...", flush=True)
    rpe_data = measure_retention_per_event(n_trials=n_trials, verbose=verbose)

    if verbose:
        print("[Efficiency] Time breakdown (1 trial) ...", flush=True)
    timing   = measure_time_breakdown(trial_seed=cf.MASTER_SEED)

    return {"retention_per_event": rpe_data, "timing": timing}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_efficiency_curves(data: dict, out_dir: str = ".") -> None:
    """
    3-panel efficiency figure:
      (A) Retention vs N replay events
      (B) Retention per event (efficiency index)
      (C) Mechanism time breakdown (pie)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rpe  = data.get("retention_per_event", {})
    tim  = data.get("timing", {})

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: retention curve
    ax = axes[0]
    ne  = rpe.get("n_events_values", [])
    ret = rpe.get("mean_rets", [])
    if ne and ret:
        ax.plot(ne, ret, "o-", color="#2ecc71", linewidth=2, markersize=6)
        no_rep = rpe.get("no_replay_ret", np.nan)
        if np.isfinite(no_rep):
            ax.axhline(no_rep, color="#e74c3c", linestyle="--", linewidth=1.2, label="No Replay")
        prod   = rpe.get("production_n_events", None)
        if prod:
            ax.axvline(prod, color="black", linestyle="--", linewidth=1.2, label="Production")
        ax.set_xlabel("N Replay Events / Rest", fontsize=9)
        ax.set_ylabel("Mean Retention (A/B/C)", fontsize=9)
        ax.set_title("Retention vs Replay Events", fontsize=10)
        ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: efficiency index
    ax = axes[1]
    eff = rpe.get("efficiency_idx", [])
    if ne and eff:
        ax.plot(ne, eff, "s-", color="#3498db", linewidth=2, markersize=6)
        ax.set_xlabel("N Replay Events / Rest", fontsize=9)
        ax.set_ylabel("Retention per Event", fontsize=9)
        ax.set_title("Efficiency Index\n(retention / n_events)", fontsize=10)
        if prod := rpe.get("production_n_events"):
            ax.axvline(prod, color="black", linestyle="--", linewidth=1.2, label="Production")
        ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel C: time pie
    ax = axes[2]
    fracs = tim.get("fractions", {})
    if fracs:
        labels_ = [k.replace("_", " ").title() for k in fracs]
        sizes_  = [max(v, 0) for v in fracs.values()]
        total_  = sum(sizes_)
        if total_ > 0:
            colors_ = ["#2ecc71", "#3498db", "#e74c3c", "#e67e22", "#9b59b6", "#1abc9c"]
            wedges, texts, autotexts = ax.pie(
                sizes_, labels=labels_, colors=colors_[:len(sizes_)],
                autopct="%1.0f%%", startangle=140, pctdistance=0.8,
                textprops={"fontsize": 8},
            )
            ax.set_title(f"Mechanism Time Breakdown\n(total={tim.get('total_s',0):.1f}s)", fontsize=10)

    fig.suptitle("Computational Efficiency Analysis\n(Slow+Replay, production params)", fontsize=11)
    fig.tight_layout()
    cf._save_fig(fig, "efficiency_curves")
    plt.close(fig)
    print("[FIG] Saved efficiency_curves.png", flush=True)


def fig_efficiency_pareto(data: dict, out_dir: str = ".") -> None:
    """
    Pareto plot: wall-time vs retention.
    Each point is one (n_events, mean_time, mean_ret) combination.
    Production point marked.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rpe = data.get("retention_per_event", {})
    ne  = rpe.get("n_events_values", [])
    ret = rpe.get("mean_rets", [])
    t   = rpe.get("mean_times", [])

    if not (ne and ret and t):
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(t, ret, c=ne, cmap="viridis", s=80, zorder=5)
    plt.colorbar(sc, ax=ax, label="N Replay Events", fraction=0.046)

    for i, (ti, ri, ni) in enumerate(zip(t, ret, ne)):
        if np.isfinite(ti) and np.isfinite(ri):
            ax.annotate(f"{ni}", (ti, ri), textcoords="offset points",
                        xytext=(4, 3), fontsize=7)

    prod = rpe.get("production_n_events")
    if prod and prod in ne:
        idx = ne.index(prod)
        if idx < len(t) and np.isfinite(t[idx]):
            ax.scatter([t[idx]], [ret[idx]], c="red", s=150, zorder=6,
                       marker="*", label=f"Production (n={prod})")
            ax.legend(fontsize=9)

    ax.set_xlabel("Mean Wall Time (s / trial)", fontsize=10)
    ax.set_ylabel("Mean Retention (A/B/C)", fontsize=10)
    ax.set_title("Pareto: Time-Retention Tradeoff\n(number of replay events)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    cf._save_fig(fig, "efficiency_pareto")
    plt.close(fig)
    print("[FIG] Saved efficiency_pareto.png", flush=True)


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

    print("[efficiency self-test] time breakdown ...", flush=True)
    tb = measure_time_breakdown(trial_seed=cf.MASTER_SEED)
    print(f"  Total: {tb['total_s']:.1f}s")
    print(f"  Fractions: {tb['fractions']}")

    print("[efficiency self-test] retention-per-event (3 vals) ...", flush=True)
    rpe = measure_retention_per_event(
        n_events_values=[5, 15, 35], n_trials=2, verbose=True
    )
    print(f"  efficiency_idx: {rpe['efficiency_idx']}")
    print("[efficiency self-test] DONE.", flush=True)
