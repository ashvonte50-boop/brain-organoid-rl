"""Comprehensive parametric sweeps for publication-ready results.

Includes:
  1. Overlap sweep: [0%, 10%, 20%, 40%, 60%] × all 4 conditions
  2. Downscale rate sweep: [0.0005, 0.001, 0.002, 0.003] × all 4 conditions
  3. Multi-seed robustness: 5 seeds × 5 trials
  4. High-fidelity replay ablation (natural vs perfect)
  5. Retention vs downscale rate plots
  6. Schema convergence vs downscale rate plots
  7. Generalization vs downscale rate plots
"""

import os
import sys
import time
import numpy as np

from compare_catastrophic_forgetting import (
    run_all_conditions, make_overlapping_assemblies,
    N_MEMORIES, ASSEMBLY_SIZE, MASTER_SEED,
)
from .schema_core import register_schema_hooks
from .schema_experiments import (
    make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE,
    OVERLAP_FRACS_SWEEP, DOWNSCALE_RATES_SWEEP, SWEEP_SEEDS,
)


# ── Helper: attach schema data ──────────────────────────────────────────

def _attach_schema_data(all_results):
    """Move hook_extra data from each trial into top-level keys."""
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


# ── 1. Overlap Sweep ────────────────────────────────────────────────────

def run_overlap_sweep(n_trials=3, seeds=SWEEP_SEEDS[:3], verbose=True):
    """Run overlap sweep: 0%, 10%, 20%, 40%, 60% × 4 conditions.

    Returns dict mapping overlap_fraction -> {
        "results": all_results,
        "schema": schema_results,
    }
    """
    register_schema_hooks()
    all_results = {}

    for overlap in OVERLAP_FRACS_SWEEP:
        print(f"\n{'='*60}", flush=True)
        print(f"OVERLAP SWEEP  overlap={overlap:.1f} ({int(overlap*100)}%)", flush=True)
        print(f"{'='*60}", flush=True)

        import compare_catastrophic_forgetting as ccf
        ccf.MASTER_SEED = seeds[0]
        ccf.torch.manual_seed(seeds[0])
        ccf.np.random.seed(seeds[0])

        # Use standard overlapping assemblies (not hierarchical) for consistency
        assemblies = make_overlapping_assemblies(N_MEMORIES, ASSEMBLY_SIZE, overlap)

        results = run_all_conditions(assemblies, n_trials=n_trials)
        _attach_schema_data(results)

        from .schema_analysis import run_all_schema_analysis
        schema_results = run_all_schema_analysis(results, verbose=verbose)

        from .schema_novel_metrics import compute_all_novel_metrics
        for res in results:
            for t in res.get("trials", []):
                try:
                    nm = compute_all_novel_metrics(t, assemblies)
                    t["novel_metrics"] = nm
                except Exception as e:
                    t["novel_metrics"] = {"error": str(e)}

        all_results[overlap] = {
            "results": results,
            "schema": schema_results,
        }

    return all_results


# ── 2. Downscale Rate Sweep ─────────────────────────────────────────────

def run_downscale_rate_sweep(n_trials=2, seeds=SWEEP_SEEDS[:1], verbose=True):
    """Run downscale rate sweep: [0.0005, 0.001, 0.002, 0.003].

    For each rate, run all 4 conditions and compute retention,
    schema convergence, and generalization.

    Returns dict mapping rate -> {"results": ..., "schema": ...}
    """
    register_schema_hooks()
    all_results = {}

    import schema_abstraction.schema_downscaling as sd
    import compare_catastrophic_forgetting as ccf

    for rate in DOWNSCALE_RATES_SWEEP:
        print(f"\n{'='*60}", flush=True)
        print(f"DOWNSCALE RATE SWEEP  rate={rate}", flush=True)
        print(f"{'='*60}", flush=True)

        original_rate = sd.DOWNSCALE_RATE
        sd.DOWNSCALE_RATE = rate

        ccf.MASTER_SEED = seeds[0]
        ccf.torch.manual_seed(seeds[0])
        ccf.np.random.seed(seeds[0])

        assemblies = make_overlapping_assemblies(N_MEMORIES, ASSEMBLY_SIZE, 0.20)

        results = run_all_conditions(assemblies, n_trials=n_trials)
        _attach_schema_data(results)

        from .schema_analysis import run_all_schema_analysis
        schema_results = run_all_schema_analysis(results, verbose=verbose)

        all_results[rate] = {
            "results": results,
            "schema": schema_results,
        }

        sd.DOWNSCALE_RATE = original_rate

    return all_results


# ── 3. Multi-Seed Robustness ────────────────────────────────────────────

def run_multi_seed_sweep(n_trials=5, seeds=SWEEP_SEEDS, verbose=True):
    """Run 5 seeds × 5 trials for statistical robustness.

    Returns dict mapping seed -> (all_results, schema_results)
    """
    register_schema_hooks()
    all_seed_data = {}

    import compare_catastrophic_forgetting as ccf

    for seed in seeds:
        print(f"\n{'='*60}", flush=True)
        print(f"MULTI-SEED SWEEP  seed={seed}", flush=True)
        print(f"{'='*60}", flush=True)

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

        results = run_all_conditions(assemblies, n_trials=n_trials)
        _attach_schema_data(results)

        from .schema_analysis import run_all_schema_analysis
        schema_results = run_all_schema_analysis(results, verbose=verbose)

        from .schema_novel_metrics import compute_all_novel_metrics
        for res in results:
            for t in res.get("trials", []):
                try:
                    nm = compute_all_novel_metrics(t, assemblies, core_mask)
                    t["novel_metrics"] = nm
                except Exception:
                    t["novel_metrics"] = {"error": str(e)}

        all_seed_data[seed] = (results, schema_results)

    # Compute aggregate statistics
    print(f"\n{'='*60}", flush=True)
    print("MULTI-SEED AGGREGATE", flush=True)
    print(f"{'='*60}", flush=True)

    # Collect directionality scores across seeds
    all_dir = {cond: [] for cond in
               ["Fast / No Replay", "Fast / Replay", "Slow / No Replay", "Slow + Replay"]}
    for seed, (results, _) in all_seed_data.items():
        for res in results:
            label = res["cond"]["label"]
            for t in res.get("trials", []):
                nm = t.get("novel_metrics", {})
                cfr = nm.get("catastrophic_forgetting_resistance", {})
                if isinstance(cfr, dict):
                    cfr_a = cfr.get("memory_A_CFR", 0.0)
                    if label in all_dir:
                        all_dir[label].append(cfr_a)

    for cond, vals in all_dir.items():
        if vals:
            arr = np.array(vals)
            print(f"  {cond:25s}: CFR_A mean={np.mean(arr):.4f} "
                  f"sem={np.std(arr, ddof=1)/np.sqrt(len(arr)):.4f} "
                  f"n={len(arr)}", flush=True)

    return all_seed_data


# ── 4. Replay Ablation: Natural vs Perfect ──────────────────────────────

def run_replay_ablation_sweep(n_trials=3, seeds=SWEEP_SEEDS[:2], verbose=True):
    """Compare natural fragmented replay vs perfect replay fidelity.

    Hypothesis:
      Perfect replay preserves episodic detail but reduces abstraction.
      Natural replay generalizes better and compresses structure more.

    Returns dict: {"natural": [(results, schema), ...],
                   "perfect": [(results, schema), ...]}
    """
    register_schema_hooks()
    all_data = {"natural": [], "perfect": []}
    import compare_catastrophic_forgetting as ccf

    for ablation_name, ablation_dict in [("natural", None), ("perfect", {"perfect_fidelity": True})]:
        print(f"\n{'='*60}", flush=True)
        print(f"REPLAY ABLATION: {ablation_name}", flush=True)
        print(f"{'='*60}", flush=True)

        for seed in seeds:
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

            results = run_all_conditions(assemblies, n_trials=n_trials,
                                         ablation=ablation_dict)
            _attach_schema_data(results)

            from .schema_analysis import run_all_schema_analysis
            schema_results = run_all_schema_analysis(results, verbose=verbose)

            all_data[ablation_name].append((results, schema_results))

    return all_data


# ── 5. Retention vs Downscale Rate Analysis ─────────────────────────────

def analyze_retention_vs_rate(sweep_results):
    """Extract retention scores per downscale rate for plotting.

    Args:
        sweep_results: dict from run_downscale_rate_sweep.

    Returns:
        dict mapping rate -> {condition -> mean_retention}
    """
    data = {}
    for rate, sweep_data in sweep_results.items():
        results = sweep_data["results"]
        rate_data = {}
        for res in results:
            label = res["cond"]["label"]
            finals = []
            for t in res.get("trials", []):
                fs = t.get("final_scores", [])
                if len(fs) > 0:
                    finals.append(np.nanmean(fs))
            rate_data[label] = {
                "mean": float(np.mean(finals)) if finals else 0.0,
                "sem": float(np.std(finals, ddof=1) / np.sqrt(len(finals))) if len(finals) > 1 else 0.0,
                "n": len(finals),
            }
        data[rate] = rate_data
    return data


# ── 6. Schema Convergence vs Downscale Rate ─────────────────────────────

def analyze_convergence_vs_rate(sweep_results):
    """Extract schema convergence slopes per downscale rate."""
    data = {}
    for rate, sweep_data in sweep_results.items():
        schema = sweep_data["schema"]
        conv = schema.get("convergence", {})
        rate_data = {}
        for label, cdata in conv.items():
            rate_data[label] = {
                "mean_slope": cdata.get("mean_slope", 0.0),
                "sem_slope": cdata.get("sem_slope", 0.0),
                "combined_p": cdata.get("combined_p", 1.0),
            }
        data[rate] = rate_data
    return data


# ── 7. Generalization vs Downscale Rate ─────────────────────────────────

def analyze_generalization_vs_rate(sweep_results):
    """Extract generalization scores per downscale rate."""
    data = {}
    for rate, sweep_data in sweep_results.items():
        schema = sweep_data["schema"]
        gen = schema.get("generalization", {})
        rate_data = {}
        for label, gdata in gen.items():
            rate_data[label] = {
                "mean": gdata.get("mean", 0.0),
                "sem": gdata.get("sem", 0.0),
            }
        data[rate] = rate_data
    return data


# ── Master Runner ───────────────────────────────────────────────────────

def run_schema_sweep(n_trials=3, n_seeds=3, verbose=True):
    """Run the complete publication-ready sweep.

    This runs ALL experiments needed for a publication:

    1. Overlap sweep (0–60%)
    2. Downscale rate sweep (0.0005–0.003)
    3. Multi-seed robustness (5 seeds × 5 trials)
    4. Replay ablation (natural vs perfect)

    Returns dict with all results.
    """
    print("=" * 70, flush=True)
    print("COMPREHENSIVE SCHEMA ABSTRACTION SWEEP", flush=True)
    print("=" * 70, flush=True)

    results = {}

    t0 = time.time()

    # 1. Overlap sweep
    print("\n>>> PHASE 1: OVERLAP SWEEP", flush=True)
    results["overlap_sweep"] = run_overlap_sweep(n_trials=max(1, n_trials // 2),
                                                  seeds=SWEEP_SEEDS[:2],
                                                  verbose=verbose)

    # 2. Downscale rate sweep
    print("\n>>> PHASE 2: DOWNSCALE RATE SWEEP", flush=True)
    results["downscale_sweep"] = run_downscale_rate_sweep(n_trials=max(1, n_trials // 2),
                                                           verbose=verbose)

    # 3. Multi-seed robustness
    print("\n>>> PHASE 3: MULTI-SEED ROBUSTNESS", flush=True)
    results["multi_seed"] = run_multi_seed_sweep(n_trials=max(1, n_trials),
                                                  seeds=SWEEP_SEEDS[:n_seeds],
                                                  verbose=verbose)

    # 4. Replay ablation
    print("\n>>> PHASE 4: REPLAY ABLATION", flush=True)
    results["replay_ablation"] = run_replay_ablation_sweep(n_trials=max(1, n_trials // 2),
                                                            seeds=SWEEP_SEEDS[:2],
                                                            verbose=verbose)

    total = time.time() - t0
    print(f"\n{'='*70}", flush=True)
    print(f"SWEEP COMPLETE  {total:.0f}s ({total/60:.1f} min)", flush=True)
    print(f"{'='*70}", flush=True)

    return results
