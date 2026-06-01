"""
gen_pubsummary.py — regenerate publication_summary.png from scratch.

Runs only Phase 1 (run_all_conditions, N=15 trials) and Phase 3
(run_prioritization_comparison, N=5 trials), then calls
fig_publication_summary().  All other phases are skipped.

Estimated wall time: ~20 minutes on production hardware.

NOTE: the if __name__ == "__main__" guard is REQUIRED on Windows.
Python's multiprocessing uses 'spawn' by default, which re-imports
the __main__ module in every worker.  Without the guard, each worker
would re-run this entire script recursively.
"""
import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"

import compare_catastrophic_forgetting as cf
import time


if __name__ == "__main__":
    t0 = time.perf_counter()

    print("[gen_pubsummary] Rebuilding assemblies ...", flush=True)
    assemblies = cf.make_overlapping_assemblies(
        cf.N_MEMORIES, cf.ASSEMBLY_SIZE, 0.20  # MAIN_OVERLAP = 0.20 (same as main())
    )

    print(f"[gen_pubsummary] Phase 1: {cf.N_TRIALS} trials x {len(cf.CONDITIONS)} conditions ...",
          flush=True)
    all_results = cf.run_all_conditions(assemblies, n_trials=cf.N_TRIALS, verbose=True)

    print(f"\n[gen_pubsummary] Phase 3: prioritization ({cf.N_TRIALS_SWEEP} trials) ...",
          flush=True)
    prio = cf.run_prioritization_comparison(assemblies, n_trials=cf.N_TRIALS_SWEEP, verbose=True)

    print("\n[gen_pubsummary] Generating publication_summary.png ...", flush=True)
    cf.fig_publication_summary(all_results, prio, {})

    elapsed = time.perf_counter() - t0
    print(f"\n[gen_pubsummary] Done in {elapsed/60:.1f} min", flush=True)
