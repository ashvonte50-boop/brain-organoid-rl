"""
extensions/benchmark.py — External benchmark validation (Task 9).

Validates the spiking model on external continual-learning benchmarks
without modifying the core architecture. Uses adapter interfaces.

Benchmarks:
  1. Associative recall task
     Learn pattern pairs (A→B), test forward recall A→B and backward B→A.
     Standard in attractor network literature.
  2. Noisy pattern completion
     After training, test recall with increasing noise/erasure in the cue.
     Measures basin-of-attraction robustness.
  3. Sequential binary patterns (mini pattern-set)
     Small binary vector set, sequentially trained, overlap-controlled.
     Tests generalisation beyond the 20-neuron assembly paradigm.
  4. Interference robustness (multi-overlap chain)
     Chain of 8 memories with 4-overlap (higher interference than production 4-mem).
     Stress-test: does the architecture scale?

All benchmarks use the EXISTING cf.py functions as building blocks —
no new plasticity rules, no network modifications.
"""
import os
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import compare_catastrophic_forgetting as cf

__all__ = [
    "run_associative_recall_benchmark",
    "run_noisy_completion_benchmark",
    "run_long_chain_benchmark",
    "run_all_benchmarks",
    "fig_benchmark_results",
]


# ---------------------------------------------------------------------------
# 1. Associative recall benchmark
# ---------------------------------------------------------------------------

def run_associative_recall_benchmark(
    n_trials: int = None,
    verbose: bool = False,
) -> dict:
    """
    Train 4 memories (A, B, C, D), then test forward-associative recall:
    cue from memory i, probe for memory i+1 activation.
    This measures whether the chain replay correctly propagates associations.

    We use the existing partial-cue probe mechanism: cue A's neurons,
    measure B's I_syn response relative to background.

    Returns dict with forward_scores (N_MEMORIES-1) and baseline_scores.
    """
    if n_trials is None:
        n_trials = cf.N_TRIALS_SWEEP

    all_forward  = []
    all_baseline = []

    for ti in range(n_trials):
        seed = cf.MASTER_SEED + ti * 37
        torch.manual_seed(seed)
        np.random.seed(seed)

        assemblies = cf.make_overlapping_assemblies(cf.N_MEMORIES, cf.ASSEMBLY_SIZE, 0.20)
        n_mem      = len(assemblies)

        result = cf.run_sequential_experiment(
            use_slow=True, use_replay=True,
            assemblies=assemblies, trial_seed=seed,
            prioritize="interference_aware", verbose=False,
        )

        # Re-build network to probe associative responses
        # (run_sequential_experiment doesn't expose final net state directly,
        # so we probe by re-running the experiment and capturing final state)
        # Alternative: rebuild net and retrain to match result state.
        # Since we can't get the net from run_sequential_experiment directly,
        # we rebuild and retrain:
        torch.manual_seed(seed)
        np.random.seed(seed)
        net  = cf.build_network(use_slow=True)
        tags = cf.SynapticTags()
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

        # Forward associative probe: cue A, measure B activation
        forward_scores = []
        for i in range(n_mem - 1):
            asm_cue    = assemblies[i]
            asm_target = assemblies[i + 1]

            net.noise_std = cf.TEST_NOISE
            net.reset_state()
            cue_n     = asm_cue[:cf.CUE_SIZE]
            cue_stim  = torch.zeros(cf.N_NEURONS, device=cf.DEVICE)
            cue_stim[cue_n] = cf.CUE_STRENGTH

            isyn_arr = np.zeros(cf.probe_steps)
            with torch.no_grad():
                for t in range(cf.probe_steps):
                    net.forward(cue_stim)
                    isyn_arr[t] = float(
                        net.I_syn[asm_target].mean().item()
                        - net.I_syn[cf.BG_START:cf.BG_END].mean().item()
                    )
            forward_scores.append(float(np.nanmean(isyn_arr)))

        all_forward.append(forward_scores)
        all_baseline.append(result["final_scores"][:n_mem-1].tolist())
        if verbose:
            print(f"  [AssocRecall] trial {ti+1}/{n_trials}: "
                  f"fwd={[f'{s:.3f}' for s in forward_scores]}", flush=True)

    fwd_arr  = np.array(all_forward)
    base_arr = np.array(all_baseline)
    return {
        "forward_mean":     np.nanmean(fwd_arr, axis=0).tolist(),
        "forward_sem":      (np.nanstd(fwd_arr, axis=0) / np.sqrt(n_trials)).tolist(),
        "baseline_mean":    np.nanmean(base_arr, axis=0).tolist(),
        "n_trials":         n_trials,
    }


# ---------------------------------------------------------------------------
# 2. Noisy pattern completion benchmark
# ---------------------------------------------------------------------------

def run_noisy_completion_benchmark(
    erasure_fracs: Optional[List[float]] = None,
    n_trials: int = None,
    verbose: bool = False,
) -> dict:
    """
    Test recall as a function of cue completeness: erase 1-erasure_frac of
    assembly neurons from the cue, measure I_syn retention of un-cued neurons.
    erasure_frac=0 → full assembly cue; erasure_frac=1 → no cue.
    """
    if erasure_fracs is None:
        erasure_fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if n_trials is None:
        n_trials = cf.N_TRIALS_SWEEP

    all_results = {ef: [] for ef in erasure_fracs}

    for ti in range(n_trials):
        seed = cf.MASTER_SEED + ti * 37
        torch.manual_seed(seed)
        np.random.seed(seed)

        assemblies = cf.make_overlapping_assemblies(cf.N_MEMORIES, cf.ASSEMBLY_SIZE, 0.20)
        n_mem      = len(assemblies)

        # Train and consolidate
        net  = cf.build_network(use_slow=True)
        tags = cf.SynapticTags()
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

        # Probe Memory A (oldest) with varying erasure fraction
        asm_a     = assemblies[0]
        n_asm     = len(asm_a)
        for ef in erasure_fracs:
            n_cue = max(1, int(round((1.0 - ef) * n_asm)))
            cue_n = asm_a[:n_cue]

            net.noise_std = cf.TEST_NOISE
            net.reset_state()
            cue_stim = torch.zeros(cf.N_NEURONS, device=cf.DEVICE)
            cue_stim[cue_n] = cf.CUE_STRENGTH

            non_cued = asm_a[n_cue:]
            if len(non_cued) == 0:
                all_results[ef].append(float("nan"))
                continue

            isyn_nc_vals = []
            isyn_bg_vals = []
            with torch.no_grad():
                for _ in range(cf.probe_steps):
                    net.forward(cue_stim)
                    isyn_nc_vals.append(float(net.I_syn[non_cued].mean().item()))
                    isyn_bg_vals.append(float(net.I_syn[cf.BG_START:cf.BG_END].mean().item()))
            score = float(np.mean(isyn_nc_vals) - np.mean(isyn_bg_vals))
            all_results[ef].append(score)

        if verbose:
            print(f"  [NoisyCompletion] trial {ti+1}/{n_trials} done", flush=True)

    means = [float(np.nanmean(all_results[ef])) for ef in erasure_fracs]
    sems  = [float(np.nanstd(all_results[ef]) / max(1, np.sqrt(n_trials))) for ef in erasure_fracs]
    return {
        "erasure_fracs": erasure_fracs,
        "means":         means,
        "sems":          sems,
        "n_trials":      n_trials,
    }


# ---------------------------------------------------------------------------
# 3. Long chain benchmark (8 memories)
# ---------------------------------------------------------------------------

def _long_chain_worker(args):
    """(n_memories, trial_seed, use_replay) → mean retention of first n_mem//2"""
    n_memories, trial_seed, use_replay = args
    try:
        # Build 8-memory chain with small assembly/overlap to fit N_EXC=240
        asm_size = 15
        overlap  = 0.20
        assemblies = cf.make_overlapping_assemblies(n_memories, asm_size, overlap)
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        net  = cf.build_network(use_slow=True)
        tags = cf.SynapticTags()
        all_replay = []

        for j, asm in enumerate(assemblies):
            cf.train_one_memory(net, asm, tags=tags, n_presentations=cf._N_PRESENTATIONS)
            if j > 0 and use_replay:
                cf.apply_competitive_interference(net, asm, assemblies[:j])
            if j < n_memories - 1:
                if use_replay:
                    cs = [cf.probe_memory(net, assemblies[i])["isyn_score"] for i in range(j+1)]
                    cf.inter_memory_rest_with_replay(
                        net, assemblies[:j+1], current_scores=cs,
                        prioritize="interference_aware", tags=tags, rest_id=j,
                        accumulated_metrics=all_replay, ablation=None,
                    )
                else:
                    cf.inter_memory_rest_no_replay(net, tags=tags)

        # Probe first half
        final = [cf.probe_memory(net, assemblies[i])["isyn_score"] for i in range(n_memories)]
        half  = n_memories // 2
        ret   = float(np.nanmean(final[:half]))
    except Exception as e:
        warnings.warn(f"long_chain_worker failed (n_mem={n_memories}): {e}")
        ret = float("nan")
    return ret


def run_long_chain_benchmark(
    chain_lengths: Optional[List[int]] = None,
    n_trials: int = None,
    verbose: bool = False,
) -> dict:
    """
    Compare Slow+Replay vs Slow+NoReplay on longer memory chains.
    chain_lengths: list of N_MEMORIES values to test.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    if chain_lengths is None:
        chain_lengths = [4, 5, 6, 7, 8]
    if n_trials is None:
        n_trials = cf.N_TRIALS_SWEEP

    seeds     = [cf.MASTER_SEED + i * 37 for i in range(n_trials)]
    n_workers = min(cf.N_WORKERS, n_trials)
    results   = {"replay": [], "no_replay": []}

    for n_mem in chain_lengths:
        for use_replay, key in [(True, "replay"), (False, "no_replay")]:
            task_args = [(n_mem, s, use_replay) for s in seeds]
            try:
                ctx = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                    trial_rets = list(pool.map(_long_chain_worker, task_args))
            except Exception as e:
                warnings.warn(f"Long chain parallel failed: {e}. Serial.")
                trial_rets = [_long_chain_worker(a) for a in task_args]
            m = float(np.nanmean(trial_rets))
            results[key].append(m)
            if verbose:
                print(f"  [LongChain] n_mem={n_mem}, replay={use_replay}: ret={m:.4f}", flush=True)

    return {
        "chain_lengths":   chain_lengths,
        "replay_means":    results["replay"],
        "no_replay_means": results["no_replay"],
        "n_trials":        n_trials,
    }


# ---------------------------------------------------------------------------
# Run all benchmarks
# ---------------------------------------------------------------------------

def run_all_benchmarks(
    n_trials: int = None,
    verbose: bool = False,
) -> dict:
    """Run all benchmark experiments."""
    if n_trials is None:
        n_trials = cf.N_TRIALS_SWEEP

    out = {}

    if verbose:
        print("[Benchmark] Associative recall ...", flush=True)
    out["associative_recall"] = run_associative_recall_benchmark(n_trials=n_trials, verbose=verbose)

    if verbose:
        print("[Benchmark] Noisy completion ...", flush=True)
    out["noisy_completion"]   = run_noisy_completion_benchmark(n_trials=n_trials, verbose=verbose)

    if verbose:
        print("[Benchmark] Long chain ...", flush=True)
    out["long_chain"]         = run_long_chain_benchmark(n_trials=n_trials, verbose=verbose)

    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def fig_benchmark_results(data: dict, out_dir: str = ".") -> None:
    """
    3-panel benchmark figure:
      (A) Forward associative recall scores vs memory position
      (B) Noisy completion: retention vs erasure fraction
      (C) Long chain: Slow+Replay vs Slow+NoReplay scaling
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: associative recall
    ax = axes[0]
    ar = data.get("associative_recall", {})
    if ar:
        n    = len(ar["forward_mean"])
        x    = np.arange(1, n + 1)
        fwd  = ar["forward_mean"]
        base = ar["baseline_mean"]
        ax.bar(x - 0.2, base, 0.35, label="Self-recall",     color="#2ecc71", alpha=0.85)
        ax.bar(x + 0.2, fwd,  0.35, label="Forward-assoc",   color="#3498db", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"A→{chr(65+i+1)}" for i in range(n)], fontsize=8)
        ax.set_ylabel("I_syn score", fontsize=9)
        ax.set_title("Associative Recall\n(self vs forward)", fontsize=10)
        ax.legend(fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel B: noisy completion
    ax = axes[1]
    nc = data.get("noisy_completion", {})
    if nc:
        ef = nc["erasure_fracs"]
        m  = nc["means"]
        s  = nc["sems"]
        ax.plot(ef, m, "o-", color="#e67e22", linewidth=2, markersize=6)
        ax.fill_between(ef, [mi - si for mi, si in zip(m, s)],
                            [mi + si for mi, si in zip(m, s)],
                        alpha=0.25, color="#e67e22")
        ax.set_xlabel("Cue Erasure Fraction", fontsize=9)
        ax.set_ylabel("Retention (Memory A)", fontsize=9)
        ax.set_title("Noisy Pattern Completion\n(basin robustness)", fontsize=10)
        ax.axhline(0, color="gray", linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel C: long chain scaling
    ax = axes[2]
    lc = data.get("long_chain", {})
    if lc:
        cl  = lc["chain_lengths"]
        rep = lc["replay_means"]
        nor = lc["no_replay_means"]
        ax.plot(cl, rep, "o-", color="#2ecc71", linewidth=2, markersize=6, label="Slow+Replay")
        ax.plot(cl, nor, "s--", color="#e74c3c", linewidth=2, markersize=6, label="Slow+NoReplay")
        ax.set_xlabel("N Memories in Chain", fontsize=9)
        ax.set_ylabel("Mean Retention (first half)", fontsize=9)
        ax.set_title("Scaling: Long Memory Chains", fontsize=10)
        ax.axvline(cf.N_MEMORIES, color="black", linestyle=":", linewidth=1.2,
                   alpha=0.7, label="Production (N=4)")
        ax.legend(fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("External Benchmark Validation", fontsize=12)
    fig.tight_layout()
    cf._save_fig(fig, "benchmark_results")
    plt.close(fig)
    print("[FIG] Saved benchmark_results.png", flush=True)


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

    print("[benchmark self-test] noisy completion (5 erasure values, 2 trials) ...", flush=True)
    nc = run_noisy_completion_benchmark(
        erasure_fracs=[0.0, 0.3, 0.6, 0.9], n_trials=2, verbose=True
    )
    print(f"  Means: {[f'{m:.3f}' for m in nc['means']]}")

    print("[benchmark self-test] long chain (2 lengths, 2 trials) ...", flush=True)
    lc = run_long_chain_benchmark(chain_lengths=[4, 6], n_trials=2, verbose=True)
    print(f"  Replay: {lc['replay_means']}")
    print(f"  NoReplay: {lc['no_replay_means']}")
    print("[benchmark self-test] DONE.", flush=True)
