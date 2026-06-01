"""
run_extended.py — Master orchestrator for all 9 extension tasks.

Runs the full publication-grade analysis suite in order:
  Task 1: Baseline comparisons (EWC, Replay Buffer, Rehearsal, Hopfield)
  Task 2: Parameter robustness sweeps
  Task 3: Extended mechanism ablations
  Task 4: Failure regime characterization
  Task 5: Biological plausibility controls
  Task 6: Reproducibility manifest
  Task 7: Statistical rigor (bootstrap CI, permutation tests, FDR)
  Task 8: Computational efficiency analysis
  Task 9: External benchmark validation

Usage (detached via launch_prod.py):
  python launch_prod.py run_extended.py

Flags (environment variables):
  EXTENDED_TASKS=1,2,3     — comma-separated list of task numbers to run (default: all)
  EXTENDED_TRIALS=5        — override N_TRIALS for all tasks
  DEV_MODE=1               — enable DEV_MODE (fast, reduced N)

Estimated wall time (production, N_TRIALS=5 each): ~6–10 hours.
Estimated wall time (DEV_MODE, N_TRIALS=2 each):   ~30–60 minutes.

All outputs saved to current directory (OUT_DIR from cf.py = ".").
Manifest saved to extended_manifest.json.
Stats CSV saved to extended_stats.csv.
"""
import os
import sys
import time
import warnings

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch

import compare_catastrophic_forgetting as cf
from extensions import stats_utils, repro, baselines, robustness
from extensions import ablations_extended, failure_analysis, bio_controls
from extensions import efficiency, benchmark

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_dev = cf.DEV_MODE
_N   = 2 if _dev else int(os.environ.get("EXTENDED_TRIALS", cf.N_TRIALS_SWEEP))
_tasks_env = os.environ.get("EXTENDED_TASKS", "")
if _tasks_env.strip():
    _TASKS_TO_RUN = {int(t.strip()) for t in _tasks_env.split(",") if t.strip().isdigit()}
else:
    _TASKS_TO_RUN = set(range(1, 10))   # all 9 tasks

_TIMER_EXTENDED: dict = {}

def _hms(s: float) -> str:
    h = int(s // 3600); m = int((s % 3600) // 60); ss = int(s % 60)
    return f"{h:02d}h{m:02d}m{ss:02d}s"

def _section(title: str) -> float:
    t0 = time.perf_counter()
    print(f"\n{'='*65}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*65}", flush=True)
    return t0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_global = time.perf_counter()

    print(f"[run_extended] Starting extended analysis suite", flush=True)
    print(f"  DEV_MODE={_dev}, N={_N}, Tasks={sorted(_TASKS_TO_RUN)}", flush=True)

    # Build shared assemblies (same as production)
    assemblies = cf.make_overlapping_assemblies(cf.N_MEMORIES, cf.ASSEMBLY_SIZE, 0.20)
    print(f"  Assemblies: {cf.N_MEMORIES} memories, "
          f"size={cf.ASSEMBLY_SIZE}, overlap=20%", flush=True)

    all_results = {}

    # ─── Task 1: Baseline Comparisons ────────────────────────────────────────
    if 1 in _TASKS_TO_RUN:
        t0 = _section("Task 1: Baseline Comparisons")
        n1 = max(2, _N)
        baseline_results = baselines.run_all_baselines(assemblies, n_trials=n1, verbose=True)
        all_results["baselines"] = baseline_results

        # Build stats table (all baselines vs Slow+Replay reference)
        ref_label = "slow_replay"
        condition_list = []
        for cfg_b in baselines.BASELINE_CONFIGS:
            key = cfg_b["key"]
            trial_list = baseline_results.get(key, [])
            scores = [r["mean_retention"] for r in trial_list if np.isfinite(r.get("mean_retention", np.nan))]
            condition_list.append({"label": cfg_b["label"], "scores": scores})

        ref_idx = next((i for i, c in enumerate(condition_list)
                        if "Full" in c["label"] or "Slow" in c["label"]), 0)
        baseline_stats = stats_utils.stats_table(condition_list, reference_idx=ref_idx,
                                                  n_boot=500, seed=0)
        stats_utils.print_stats_table(baseline_stats, title="Baseline Comparison Statistics")
        stats_utils.export_stats_csv(baseline_stats, "baseline_stats.csv")
        baselines.fig_baseline_comparison(baseline_results, baseline_stats)
        _TIMER_EXTENDED["baselines"] = time.perf_counter() - t0
        print(f"  Task 1 done in {_hms(_TIMER_EXTENDED['baselines'])}", flush=True)

    # ─── Task 2: Robustness sweeps ────────────────────────────────────────────
    if 2 in _TASKS_TO_RUN:
        t0 = _section("Task 2: Robustness / Sensitivity Sweeps")
        n2 = max(2, _N)
        # In DEV_MODE run only 4 params; production runs all 10
        params_to_run = (["coherence_thr", "pers_gain", "n_replay_events", "overlap_frac"]
                         if _dev else None)
        sweep_data = robustness.run_full_sensitivity(
            assemblies, n_trials=n2, verbose=True, params=params_to_run
        )
        all_results["robustness"] = sweep_data
        robustness.fig_robustness_heatmap(sweep_data)
        robustness.fig_sensitivity_rankings(sweep_data)
        _TIMER_EXTENDED["robustness"] = time.perf_counter() - t0
        print(f"  Task 2 done in {_hms(_TIMER_EXTENDED['robustness'])}", flush=True)

    # ─── Task 3: Extended Ablations ───────────────────────────────────────────
    if 3 in _TASKS_TO_RUN:
        t0 = _section("Task 3: Extended Mechanism Ablations")
        n3 = max(2, _N)
        # In DEV_MODE use first 5 conditions only
        conds = (ablations_extended.EXTENDED_ABLATION_CONDITIONS[:5]
                 if _dev else ablations_extended.EXTENDED_ABLATION_CONDITIONS)
        ablation_results = ablations_extended.run_all_extended_ablations(
            assemblies, n_trials=n3, verbose=True, conditions=conds
        )
        all_results["ablations"] = ablation_results

        # Stats: vs Full Model
        cond_list = []
        for cond in conds:
            label = cond["label"].replace("\n", " ")
            trial_list = ablation_results.get(label, [])
            scores = [r["mean_ret"] for r in trial_list if np.isfinite(r.get("mean_ret", np.nan))]
            cond_list.append({"label": label, "scores": scores})
        ref_idx_ab = next((i for i, c in enumerate(cond_list) if "Full" in c["label"]), 0)
        abl_stats  = stats_utils.stats_table(cond_list, reference_idx=ref_idx_ab, n_boot=500)
        stats_utils.print_stats_table(abl_stats, title="Extended Ablation Statistics")
        stats_utils.export_stats_csv(abl_stats, "ablation_stats.csv")
        ablations_extended.fig_ablation_matrix(ablation_results, conds)
        ablations_extended.fig_mechanism_contributions(ablation_results, conds)
        _TIMER_EXTENDED["ablations"] = time.perf_counter() - t0
        print(f"  Task 3 done in {_hms(_TIMER_EXTENDED['ablations'])}", flush=True)

    # ─── Task 4: Failure Regime Analysis ─────────────────────────────────────
    if 4 in _TASKS_TO_RUN:
        t0 = _section("Task 4: Failure Regime Characterization")
        n4 = max(2, _N)

        if _dev:
            ov_vals = [0.10, 0.20, 0.30]
            gm_vals = [0.50, 0.65]
        else:
            ov_vals = None; gm_vals = None

        grid, ov_vals_out, gm_vals_out = failure_analysis.run_overlap_gamma_grid(
            overlap_values=ov_vals, gamma_values=gm_vals, n_trials=n4, verbose=True
        )
        failure_analysis.fig_phase_diagram(grid, ov_vals_out, gm_vals_out)

        sat_data = failure_analysis.analyze_consolidation_saturation(
            n_presentations_values=([5, 12, 20] if _dev else None),
            n_trials=n4, verbose=True,
        )
        failure_analysis.fig_consolidation_saturation(sat_data)

        if not _dev:
            fusion_data = failure_analysis.analyze_attractor_fusion(n_trials=n4, verbose=True)
            failure_analysis.fig_attractor_fusion(fusion_data)
            all_results["attractor_fusion"] = fusion_data

        all_results["failure"] = {"grid": grid.tolist(), "saturation": sat_data}
        _TIMER_EXTENDED["failure"] = time.perf_counter() - t0
        print(f"  Task 4 done in {_hms(_TIMER_EXTENDED['failure'])}", flush=True)

    # ─── Task 5: Biological Plausibility Controls ─────────────────────────────
    if 5 in _TASKS_TO_RUN:
        t0 = _section("Task 5: Biological Plausibility Controls")
        n5 = max(2, _N)

        burst_data   = bio_controls.run_burst_timing_sweep(n_trials=n5, verbose=True)
        cue_data     = bio_controls.run_cue_sparsity_sweep(n_trials=n5, verbose=True)
        window_data  = bio_controls.run_replay_window_sweep(n_trials=n5, verbose=True)
        ei_data      = bio_controls.run_ei_balance_sweep(n_trials=n5, verbose=True)
        latency_data = bio_controls.run_latency_sweep(n_trials=n5, verbose=True)

        bio_controls.fig_bio_controls_summary(
            burst_data=burst_data, cue_data=cue_data, window_data=window_data,
            ei_data=ei_data, latency_data=latency_data,
        )
        all_results["bio_controls"] = {
            "burst":   burst_data, "cue":     cue_data,
            "window":  window_data, "ei":      ei_data,
            "latency": latency_data,
        }
        _TIMER_EXTENDED["bio_controls"] = time.perf_counter() - t0
        print(f"  Task 5 done in {_hms(_TIMER_EXTENDED['bio_controls'])}", flush=True)

    # ─── Task 6: Reproducibility ──────────────────────────────────────────────
    if 6 in _TASKS_TO_RUN:
        t0 = _section("Task 6: Reproducibility Infrastructure")
        manifest = repro.save_manifest(
            results=all_results,
            out_path="extended_manifest.json",
            notes=f"Extended analysis suite, DEV_MODE={_dev}, N={_N}",
        )
        print(f"  Manifest saved: run_id={manifest.run_id}, "
              f"results_hash={manifest.results_hash}", flush=True)
        print(f"  git_hash={manifest.git_hash}, dirty={manifest.git_dirty}", flush=True)
        cfg_keys = list(manifest.config.keys())
        print(f"  Config captured ({len(cfg_keys)} keys): {cfg_keys[:5]} ...", flush=True)
        all_results["manifest"] = manifest.to_dict()
        _TIMER_EXTENDED["repro"] = time.perf_counter() - t0
        print(f"  Task 6 done in {_hms(_TIMER_EXTENDED['repro'])}", flush=True)

    # ─── Task 7: Statistical Rigor (comprehensive table) ─────────────────────
    if 7 in _TASKS_TO_RUN:
        t0 = _section("Task 7: Statistical Rigor")
        # Aggregate all condition scores for a comprehensive stats table
        all_conds_stats = []

        # From baselines
        if "baselines" in all_results:
            for cfg_b in baselines.BASELINE_CONFIGS:
                trial_list = all_results["baselines"].get(cfg_b["key"], [])
                scores = [r["mean_retention"] for r in trial_list if np.isfinite(r.get("mean_retention", np.nan))]
                if scores:
                    all_conds_stats.append({"label": cfg_b["label"].replace("\n", " "), "scores": scores})

        if all_conds_stats:
            ref_i = next((i for i, c in enumerate(all_conds_stats) if "Full" in c["label"] or "Slow+Replay" in c["label"]), 0)
            full_stats = stats_utils.stats_table(all_conds_stats, reference_idx=ref_i, n_boot=1000, seed=42)
            stats_utils.print_stats_table(full_stats, title="Full Extended Analysis Statistics")
            stats_utils.export_stats_csv(full_stats, "extended_stats.csv")
            print(f"  Stats CSV: extended_stats.csv ({len(full_stats)} conditions)", flush=True)

        _TIMER_EXTENDED["stats"] = time.perf_counter() - t0
        print(f"  Task 7 done in {_hms(_TIMER_EXTENDED['stats'])}", flush=True)

    # ─── Task 8: Efficiency Analysis ──────────────────────────────────────────
    if 8 in _TASKS_TO_RUN:
        t0 = _section("Task 8: Computational Efficiency Analysis")
        n8 = max(2, _N)
        eff_data = efficiency.run_efficiency_sweep(n_trials=n8, verbose=True)
        all_results["efficiency"] = eff_data
        efficiency.fig_efficiency_curves(eff_data)
        efficiency.fig_efficiency_pareto(eff_data)
        _TIMER_EXTENDED["efficiency"] = time.perf_counter() - t0
        print(f"  Task 8 done in {_hms(_TIMER_EXTENDED['efficiency'])}", flush=True)

    # ─── Task 9: External Benchmarks ─────────────────────────────────────────
    if 9 in _TASKS_TO_RUN:
        t0 = _section("Task 9: External Benchmark Validation")
        n9 = max(2, _N)
        bench_data = benchmark.run_all_benchmarks(n_trials=n9, verbose=True)
        all_results["benchmarks"] = bench_data
        benchmark.fig_benchmark_results(bench_data)
        _TIMER_EXTENDED["benchmarks"] = time.perf_counter() - t0
        print(f"  Task 9 done in {_hms(_TIMER_EXTENDED['benchmarks'])}", flush=True)

    # ─── Final summary ────────────────────────────────────────────────────────
    total_s = time.perf_counter() - t_global
    print(f"\n{'='*65}", flush=True)
    print(f"  run_extended COMPLETE in {_hms(total_s)}", flush=True)
    print(f"{'='*65}", flush=True)
    for task_key, elapsed in _TIMER_EXTENDED.items():
        print(f"    {task_key:<20}: {_hms(elapsed)}", flush=True)

    if "manifest" not in all_results and 6 not in _TASKS_TO_RUN:
        # Save a minimal manifest if task 6 was skipped
        try:
            manifest = repro.save_manifest(
                results=all_results, out_path="extended_manifest.json",
                notes=f"Extended run (tasks={sorted(_TASKS_TO_RUN)})"
            )
            print(f"\n  Manifest saved: {manifest.run_id}", flush=True)
        except Exception as e:
            warnings.warn(f"Manifest save failed: {e}")

    print(f"\n[run_extended] All done. Total: {_hms(total_s)}", flush=True)
