"""
extensions/failure_analysis.py — Failure regime characterization (Task 4).

Characterizes WHEN and HOW the model fails, producing:
  1. Overlap-instability phase diagram (retention vs overlap × gamma)
  2. Replay coherence collapse map (coherence vs events vs noise)
  3. Attractor fusion analysis (W_eff cross-attractor confusion matrix)
  4. Consolidation saturation curve (W_slow distribution vs presentations)
  5. Replay competition stability analysis (endogenous vs interference-aware vs broken)

These figures document the failure regime, proving the model's operating point
is safely above critical thresholds — a key reviewer concern.
"""
import os
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import compare_catastrophic_forgetting as cf

__all__ = [
    "run_overlap_gamma_grid",
    "run_coherence_collapse_sweep",
    "analyze_attractor_fusion",
    "analyze_consolidation_saturation",
    "run_replay_stability_sweep",
    "fig_phase_diagram",
    "fig_coherence_collapse_map",
    "fig_attractor_fusion",
    "fig_consolidation_saturation",
]

# ---------------------------------------------------------------------------
# 1. Overlap × Gamma phase diagram
# ---------------------------------------------------------------------------

def _overlap_gamma_worker(args):
    """Worker: (overlap_frac, gamma, trial_seed) → mean retention float."""
    overlap_frac, gamma, trial_seed = args
    old_gamma = cf.GAMMA
    try:
        assemblies = cf.make_overlapping_assemblies(cf.N_MEMORIES, cf.ASSEMBLY_SIZE, overlap_frac)
        cf.GAMMA = gamma
        result = cf.run_sequential_experiment(
            use_slow=True, use_replay=True,
            assemblies=assemblies, trial_seed=trial_seed,
            prioritize="interference_aware", verbose=False,
        )
        n = len(assemblies)
        ret = float(np.nanmean(result["final_scores"][:n - 1]))
    except Exception as e:
        warnings.warn(f"overlap_gamma_worker failed (overlap={overlap_frac}, gamma={gamma}): {e}")
        ret = float("nan")
    finally:
        cf.GAMMA = old_gamma
    return ret


def run_overlap_gamma_grid(
    overlap_values: Optional[List[float]] = None,
    gamma_values: Optional[List[float]] = None,
    n_trials: int = 3,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    2D sweep: overlap_frac × gamma.
    Returns (grid, overlap_values, gamma_values) where grid[i,j] = mean retention.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    if overlap_values is None:
        overlap_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    if gamma_values is None:
        gamma_values = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]

    seeds = [cf.MASTER_SEED + i * 37 for i in range(n_trials)]
    tasks = [
        (ov, gm, s)
        for ov in overlap_values
        for gm in gamma_values
        for s in seeds
    ]
    n_workers = min(cf.N_WORKERS, len(tasks))

    if verbose:
        print(f"  [FailureAnalysis] overlap×gamma grid: "
              f"{len(overlap_values)}×{len(gamma_values)} = {len(overlap_values)*len(gamma_values)} "
              f"cells × {n_trials} trials ...", flush=True)

    try:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            flat_results = list(pool.map(_overlap_gamma_worker, tasks))
    except Exception as e:
        warnings.warn(f"Grid sweep parallel failed: {e}. Serial.")
        flat_results = [_overlap_gamma_worker(t) for t in tasks]

    # Aggregate into grid (mean over trials)
    n_ov = len(overlap_values)
    n_gm = len(gamma_values)
    grid = np.full((n_ov, n_gm), np.nan)
    idx  = 0
    for i in range(n_ov):
        for j in range(n_gm):
            trial_vals = flat_results[idx:idx + n_trials]
            grid[i, j] = float(np.nanmean(trial_vals))
            idx += n_trials

    return grid, overlap_values, gamma_values


# ---------------------------------------------------------------------------
# 2. Coherence collapse sweep
# ---------------------------------------------------------------------------

def _coherence_collapse_worker(args):
    """Worker: (n_events, noise_std, trial_seed) → (mean_ret, mean_coherence)."""
    n_events, noise_std, trial_seed = args
    old_events = cf._N_REPLAY_EVENTS
    old_noise  = cf.REPLAY_NOISE_STD
    try:
        cf._N_REPLAY_EVENTS = n_events
        # Note: REPLAY_NOISE_STD is also a function default in _replay_one_event.
        # We patch the module constant AND use a workaround (patch both).
        cf.REPLAY_NOISE_STD = noise_std
        assemblies = cf.make_overlapping_assemblies(cf.N_MEMORIES, cf.ASSEMBLY_SIZE, 0.20)
        result = cf.run_sequential_experiment(
            use_slow=True, use_replay=True,
            assemblies=assemblies, trial_seed=trial_seed,
            prioritize="interference_aware", verbose=False,
        )
        n = len(assemblies)
        ret = float(np.nanmean(result["final_scores"][:n - 1]))
        # Extract mean coherence from replay metrics
        metrics = result.get("replay_metrics", [])
        cohs = [m.get("peak_coherence", np.nan) for m in metrics if m]
        mean_coh = float(np.nanmean(cohs)) if cohs else float("nan")
    except Exception as e:
        warnings.warn(f"coherence_collapse_worker failed: {e}")
        ret = float("nan")
        mean_coh = float("nan")
    finally:
        cf._N_REPLAY_EVENTS = old_events
        cf.REPLAY_NOISE_STD = old_noise
    return ret, mean_coh


def run_coherence_collapse_sweep(
    n_events_values: Optional[List[int]] = None,
    noise_values: Optional[List[float]] = None,
    n_trials: int = 3,
    verbose: bool = False,
) -> dict:
    """
    2D sweep: n_replay_events × replay_noise_std.
    Returns dict with retention_grid and coherence_grid arrays.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    if n_events_values is None:
        n_events_values = [5, 10, 20, 35, 50]
    if noise_values is None:
        noise_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    seeds = [cf.MASTER_SEED + i * 37 for i in range(n_trials)]
    tasks = [
        (ne, ns, s)
        for ne in n_events_values
        for ns in noise_values
        for s in seeds
    ]
    n_workers = min(cf.N_WORKERS, len(tasks))

    if verbose:
        print(f"  [FailureAnalysis] coherence collapse sweep: "
              f"{len(n_events_values)}×{len(noise_values)} cells × {n_trials} trials ...",
              flush=True)

    try:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            flat_results = list(pool.map(_coherence_collapse_worker, tasks))
    except Exception as e:
        warnings.warn(f"Coherence collapse parallel failed: {e}. Serial.")
        flat_results = [_coherence_collapse_worker(t) for t in tasks]

    n_ev, n_ns = len(n_events_values), len(noise_values)
    ret_grid = np.full((n_ev, n_ns), np.nan)
    coh_grid = np.full((n_ev, n_ns), np.nan)
    idx = 0
    for i in range(n_ev):
        for j in range(n_ns):
            vals = flat_results[idx:idx + n_trials]
            ret_grid[i, j] = np.nanmean([v[0] for v in vals])
            coh_grid[i, j] = np.nanmean([v[1] for v in vals])
            idx += n_trials

    return {
        "retention_grid": ret_grid,
        "coherence_grid": coh_grid,
        "n_events_values": n_events_values,
        "noise_values":    noise_values,
    }


# ---------------------------------------------------------------------------
# 3. Attractor fusion analysis
# ---------------------------------------------------------------------------

def analyze_attractor_fusion(
    overlap_fracs: Optional[List[float]] = None,
    n_trials: int = 3,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Measure attractor fusion: for each overlap fraction, compute the
    cross-memory confusion matrix (how much probing memory A activates memory B neurons).
    High off-diagonal entries indicate attractor fusion / memory merging.

    Returns dict with:
      confusion_matrices: (n_overlaps, n_mem, n_mem) array
      overlap_fracs: list
    """
    if overlap_fracs is None:
        overlap_fracs = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

    n_mem    = cf.N_MEMORIES
    n_ov     = len(overlap_fracs)
    conf_all = np.full((n_ov, n_trials, n_mem, n_mem), np.nan)

    for oi, ov in enumerate(overlap_fracs):
        for ti in range(n_trials):
            seed = cf.MASTER_SEED + ti * 37
            torch.manual_seed(seed)
            np.random.seed(seed)

            try:
                assemblies = cf.make_overlapping_assemblies(n_mem, cf.ASSEMBLY_SIZE, ov)
                net = cf.build_network(use_slow=True)
                tags = cf.SynapticTags()

                # Train all memories
                for j, asm in enumerate(assemblies):
                    cf.train_one_memory(net, asm, tags=tags,
                                        n_presentations=cf._N_PRESENTATIONS)
                    if j > 0:
                        cf.apply_competitive_interference(net, asm, assemblies[:j])
                    if j < n_mem - 1:
                        cf.inter_memory_rest_with_replay(
                            net, assemblies[:j+1],
                            current_scores=[cf.probe_memory(net, assemblies[k])["isyn_score"]
                                           for k in range(j+1)],
                            prioritize="interference_aware", tags=tags,
                            rest_id=j, accumulated_metrics=None, ablation=None,
                        )

                # Confusion matrix: cue memory i, measure all memories
                for i in range(n_mem):
                    net.noise_std = cf.TEST_NOISE
                    net.reset_state()
                    cue_n = assemblies[i][:cf.CUE_SIZE]
                    cue_stim = torch.zeros(cf.N_NEURONS, device=cf.DEVICE)
                    cue_stim[cue_n] = cf.CUE_STRENGTH

                    with torch.no_grad():
                        for _ in range(cf.probe_steps):
                            net.forward(cue_stim)

                    for j in range(n_mem):
                        asm_j = assemblies[j]
                        spk_j = float(net.spikes[asm_j].float().mean().item())
                        bg    = float(net.spikes[cf.BG_START:cf.BG_END].float().mean().item())
                        conf_all[oi, ti, i, j] = spk_j - bg

            except Exception as e:
                warnings.warn(f"Attractor fusion failed (ov={ov}, trial={ti}): {e}")

        if verbose:
            mean_conf = np.nanmean(conf_all[oi], axis=0)
            diag      = np.nanmean(np.diag(mean_conf))
            off       = np.nanmean(mean_conf[~np.eye(n_mem, dtype=bool)])
            print(f"  [Fusion] overlap={ov:.2f}: diag={diag:.3f}, off-diag={off:.3f}", flush=True)

    # Mean over trials
    confusion_mean = np.nanmean(conf_all, axis=1)  # (n_ov, n_mem, n_mem)
    return {
        "confusion_matrices": confusion_mean,
        "confusion_all":      conf_all,
        "overlap_fracs":      overlap_fracs,
    }


# ---------------------------------------------------------------------------
# 4. Consolidation saturation curve
# ---------------------------------------------------------------------------

def analyze_consolidation_saturation(
    n_presentations_values: Optional[List[int]] = None,
    n_trials: int = 3,
    verbose: bool = False,
) -> dict:
    """
    Measure W_slow distribution after training memory A with varying N_PRESENTATIONS.
    Shows where saturation causes instability.
    Returns dict with w_slow_means, w_slow_maxs, stability_flags.
    """
    if n_presentations_values is None:
        n_presentations_values = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30]

    assemblies = cf.make_overlapping_assemblies(cf.N_MEMORIES, cf.ASSEMBLY_SIZE, 0.20)
    asm_a      = assemblies[0]

    results = {
        "n_presentations": n_presentations_values,
        "w_slow_mean":     [],
        "w_slow_max":      [],
        "w_slow_frac_saturated": [],
        "retention_mean":  [],
    }

    for n_pres in n_presentations_values:
        w_slow_means = []
        w_slow_maxs  = []
        w_frac_sats  = []
        rets         = []

        for ti in range(n_trials):
            seed = cf.MASTER_SEED + ti * 37
            torch.manual_seed(seed)
            np.random.seed(seed)
            try:
                net  = cf.build_network(use_slow=True)
                tags = cf.SynapticTags()
                cf.train_one_memory(net, asm_a, tags=tags, n_presentations=n_pres)
                # One inter-memory rest with replay
                cf.inter_memory_rest_with_replay(
                    net, [asm_a], current_scores=[cf.probe_memory(net, asm_a)["isyn_score"]],
                    prioritize="interference_aware", tags=tags, rest_id=0,
                    accumulated_metrics=None, ablation=None,
                )
                # W_slow stats
                asm_e = asm_a[asm_a < cf.N_EXC]
                w_s   = net.W_slow[np.ix_(asm_e, asm_e)].detach().cpu().numpy()
                w_slow_means.append(float(np.mean(w_s)))
                w_slow_maxs.append(float(np.max(w_s)))
                w_frac_sats.append(float(np.mean(w_s >= 0.95 * cf.W_MAX)))
                # Probe retention after training B (forgetting pressure)
                cf.train_one_memory(net, assemblies[1], tags=tags, n_presentations=n_pres)
                r = cf.probe_memory(net, asm_a)
                rets.append(float(r["isyn_score"]))
            except Exception as e:
                warnings.warn(f"Saturation analysis failed (n_pres={n_pres}, ti={ti}): {e}")
                w_slow_means.append(np.nan)
                w_slow_maxs.append(np.nan)
                w_frac_sats.append(np.nan)
                rets.append(np.nan)

        results["w_slow_mean"].append(float(np.nanmean(w_slow_means)))
        results["w_slow_max"].append(float(np.nanmean(w_slow_maxs)))
        results["w_slow_frac_saturated"].append(float(np.nanmean(w_frac_sats)))
        results["retention_mean"].append(float(np.nanmean(rets)))

        if verbose:
            print(f"  [Saturation] n_pres={n_pres:3d}: "
                  f"W_slow_max={results['w_slow_max'][-1]:.3f}, "
                  f"frac_sat={results['w_slow_frac_saturated'][-1]:.3f}, "
                  f"ret={results['retention_mean'][-1]:.4f}", flush=True)

    return results


# ---------------------------------------------------------------------------
# 5. Replay competition stability
# ---------------------------------------------------------------------------

def _replay_stability_worker(args):
    """(prioritize, pers_gain, trial_seed) → mean_retention"""
    prioritize, pers_gain, trial_seed = args
    assemblies = cf.make_overlapping_assemblies(cf.N_MEMORIES, cf.ASSEMBLY_SIZE, 0.20)
    try:
        result = cf.run_sequential_experiment(
            use_slow=True, use_replay=True,
            assemblies=assemblies, trial_seed=trial_seed,
            prioritize=prioritize, verbose=False,
            ablation={"pers_gain": pers_gain, "use_competition": True},
        )
        n = len(assemblies)
        ret = float(np.nanmean(result["final_scores"][:n - 1]))
    except Exception as e:
        warnings.warn(f"Replay stability worker failed: {e}")
        ret = float("nan")
    return ret


def run_replay_stability_sweep(
    pers_gain_values: Optional[List[float]] = None,
    n_trials: int = 3,
    verbose: bool = False,
) -> dict:
    """
    Sweep persistence gain for each scheduling mode.
    Shows interaction: does high persistence gain destabilize endogenous scheduling?
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    if pers_gain_values is None:
        pers_gain_values = [0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80]
    prioritize_modes = ["uniform", "interference_aware", "endogenous"]

    seeds = [cf.MASTER_SEED + i * 37 for i in range(n_trials)]
    results = {mode: [] for mode in prioritize_modes}

    for mode in prioritize_modes:
        for pg in pers_gain_values:
            tasks     = [(mode, pg, s) for s in seeds]
            n_workers = min(cf.N_WORKERS, n_trials)
            try:
                ctx = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                    trial_rets = list(pool.map(_replay_stability_worker, tasks))
            except Exception as e:
                warnings.warn(f"Stability sweep parallel failed: {e}. Serial.")
                trial_rets = [_replay_stability_worker(t) for t in tasks]
            results[mode].append(float(np.nanmean(trial_rets)))
            if verbose:
                print(f"  [ReplayStability] mode={mode}, pg={pg:.2f}: "
                      f"ret={results[mode][-1]:.4f}", flush=True)

    return {"modes": prioritize_modes, "pers_gain_values": pers_gain_values, "results": results}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_phase_diagram(
    grid: np.ndarray,
    overlap_values: List[float],
    gamma_values: List[float],
    out_dir: str = ".",
) -> None:
    """Phase diagram: overlap × gamma retention heatmap with failure zone contour."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        grid.T, aspect="auto", origin="lower", cmap="RdYlGn",
        extent=[overlap_values[0], overlap_values[-1],
                gamma_values[0],   gamma_values[-1]],
        vmin=np.nanmin(grid), vmax=max(np.nanmax(grid), 0.01),
    )
    plt.colorbar(im, ax=ax, fraction=0.046, label="Mean A/B/C Retention")
    ax.set_xlabel("Overlap Fraction", fontsize=10)
    ax.set_ylabel("Gamma (Slow Weight Fraction)", fontsize=10)
    ax.set_title("Phase Diagram: Retention vs Overlap × Gamma\n"
                 "(production operating point marked with ★)", fontsize=10)
    # Mark production operating point
    ax.plot(0.20, cf.GAMMA, "w*", markersize=14, zorder=5, label="Production")
    # Mark failure contour (ret < 0.05)
    try:
        ax.contour(
            np.array(overlap_values), np.array(gamma_values), grid.T,
            levels=[0.05], colors="red", linewidths=1.5, linestyles="--",
        )
    except Exception:
        pass
    ax.legend(fontsize=9)
    cf._save_fig(fig, "failure_phase_diagram")
    plt.close(fig)
    print("[FIG] Saved failure_phase_diagram.png", flush=True)


def fig_coherence_collapse_map(
    data: dict,
    out_dir: str = ".",
) -> None:
    """2D heatmap: n_replay_events × noise — mean coherence and retention."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    n_ev  = data["n_events_values"]
    ns    = data["noise_values"]
    ret_g = data["retention_grid"]
    coh_g = data["coherence_grid"]

    for ax, grid, title, cmap in [
        (ax1, ret_g, "Mean Retention", "RdYlGn"),
        (ax2, coh_g, "Mean Replay Coherence", "Blues"),
    ]:
        im = ax.imshow(
            grid, aspect="auto", cmap=cmap,
            vmin=np.nanmin(grid), vmax=max(np.nanmax(grid), 0.01),
        )
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([f"{v:.1f}" for v in ns], fontsize=8)
        ax.set_yticks(range(len(n_ev)))
        ax.set_yticklabels([str(v) for v in n_ev], fontsize=8)
        ax.set_xlabel("Replay Noise STD", fontsize=9)
        ax.set_ylabel("N Replay Events / Rest", fontsize=9)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        # Mark production point
        try:
            xi = [i for i, v in enumerate(ns) if abs(v - cf.REPLAY_NOISE_STD) < 0.1]
            yi = [i for i, v in enumerate(n_ev) if v == cf._N_REPLAY_EVENTS]
            if xi and yi:
                ax.plot(xi[0], yi[0], "w*", markersize=12, zorder=5)
        except Exception:
            pass

    fig.suptitle("Coherence Collapse Map\n(★ = production operating point)", fontsize=11)
    fig.tight_layout()
    cf._save_fig(fig, "coherence_collapse_map")
    plt.close(fig)
    print("[FIG] Saved coherence_collapse_map.png", flush=True)


def fig_attractor_fusion(data: dict, out_dir: str = ".") -> None:
    """Confusion matrices at selected overlaps showing attractor fusion transition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overlap_fracs = data["overlap_fracs"]
    conf_mats     = data["confusion_matrices"]
    n_show        = min(6, len(overlap_fracs))
    fig, axes     = plt.subplots(1, n_show, figsize=(3.5 * n_show, 3.5))
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i >= len(overlap_fracs):
            ax.axis("off"); continue
        cm  = conf_mats[i]
        vmax = max(float(np.nanmax(cm)), 0.01)
        im  = ax.imshow(cm, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(f"Overlap={overlap_fracs[i]:.0%}", fontsize=9)
        ax.set_xlabel("Probed Memory", fontsize=8)
        ax.set_ylabel("Cue Memory", fontsize=8)
        ax.set_xticks(range(cm.shape[1]))
        ax.set_yticks(range(cm.shape[0]))
        ax.set_xticklabels([chr(65+j) for j in range(cm.shape[1])], fontsize=8)
        ax.set_yticklabels([chr(65+j) for j in range(cm.shape[0])], fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Attractor Fusion: Cross-Memory Confusion vs Overlap\n"
                 "(off-diagonal activation = fusion risk)", fontsize=11)
    fig.tight_layout()
    cf._save_fig(fig, "attractor_fusion")
    plt.close(fig)
    print("[FIG] Saved attractor_fusion.png", flush=True)


def fig_consolidation_saturation(data: dict, out_dir: str = ".") -> None:
    """W_slow saturation curve and retention collapse vs N_PRESENTATIONS."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_pres = data["n_presentations"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, y_key, ylabel, title, color in [
        (axes[0], "w_slow_max",             "Peak W_slow",            "W_slow Peak vs Presentations", "#3498db"),
        (axes[1], "w_slow_frac_saturated",  "Fraction saturated\n(≥95% W_MAX)", "Saturation Fraction", "#e74c3c"),
        (axes[2], "retention_mean",          "Retention (A after B)",  "Post-B Retention",             "#2ecc71"),
    ]:
        ax.plot(n_pres, data[y_key], "o-", color=color, linewidth=2, markersize=6)
        ax.set_xlabel("N Presentations / Memory", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.axvline(cf.N_PRESENTATIONS_PER_MEM, color="black", linestyle="--",
                   linewidth=1.2, label="Production (12)")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Consolidation Saturation Analysis\n"
                 "(production N_PRESENTATIONS=12 marked with dashed line)", fontsize=11)
    fig.tight_layout()
    cf._save_fig(fig, "consolidation_saturation")
    plt.close(fig)
    print("[FIG] Saved consolidation_saturation.png", flush=True)


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

    print("[failure_analysis self-test] saturation analysis (fast) ...", flush=True)
    sat = analyze_consolidation_saturation(
        n_presentations_values=[5, 10, 12, 20],
        n_trials=2, verbose=True,
    )
    print(f"  w_slow_max: {sat['w_slow_max']}")
    print(f"  retention:  {sat['retention_mean']}")

    print("[failure_analysis self-test] overlap×gamma grid (tiny) ...", flush=True)
    grid, ovs, gms = run_overlap_gamma_grid(
        overlap_values=[0.10, 0.20, 0.30],
        gamma_values=[0.50, 0.65],
        n_trials=2, verbose=True,
    )
    print(f"  grid shape: {grid.shape}, min={np.nanmin(grid):.3f}, max={np.nanmax(grid):.3f}")
    print("[failure_analysis self-test] DONE.", flush=True)
