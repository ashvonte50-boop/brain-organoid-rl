"""Quick validation: Slow+Replay with hierarchical schema.

Runs 1 seed, 1 trial, prints results + saves figures.
Should complete in <15 min.
"""
import os, sys, time
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

import multiprocessing as mp
mp.freeze_support()

import numpy as np
import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True

from schema_abstraction.schema_core import register_schema_hooks, _SCHEMA_CORE_MASK
from schema_abstraction.schema_experiments import (
    make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE, _attach_schema_data,
)
from schema_abstraction.schema_novel_metrics import compute_all_novel_metrics
from schema_abstraction.schema_analysis import run_all_schema_analysis
from schema_abstraction.schema_visualization import generate_all_schema_figures


def _run():
    t0 = time.time()
    print("=== Schema Abstraction Validation ===", flush=True)

    register_schema_hooks()

    seed = ccf.MASTER_SEED
    ccf.torch.manual_seed(seed)
    ccf.np.random.seed(seed)

    assemblies, core_mask = make_schema_assemblies(
        ccf.N_MEMORIES, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
    import schema_abstraction.schema_core as sc
    sc._SCHEMA_CORE_MASK = core_mask

    print(f"Assemblies: {len(assemblies)} memories, core={len(core_mask)} neurons", flush=True)

    # Run all 4 conditions, 1 trial each
    all_results = ccf.run_all_conditions(assemblies, n_trials=1)

    _attach_schema_data(all_results)

    for res in all_results:
        for t in res.get("trials", []):
            try:
                t["novel_metrics"] = compute_all_novel_metrics(t, assemblies, core_mask)
            except Exception as e:
                t["novel_metrics"] = {"error": str(e)}
            # Print summary
            cond = res["cond"]["label"]
            fs = t.get("final_scores", [])
            nm = t.get("novel_metrics", {})
            sci = nm.get("schema_crystallization_index", {}).get("final_SCI", "?")
            cfr = nm.get("catastrophic_forgetting_resistance", {}).get("memory_A_CFR", "?")
            gen = t.get("generalization", {})
            antic = t.get("anti_prediction", {})
            print(f"  {cond:20s} final={np.round(fs, 4).tolist()}  SCI={sci}  CFR_A={cfr}", flush=True)

    # Analysis
    schema_results = run_all_schema_analysis(all_results, verbose=True)

    # Figures
    try:
        generate_all_schema_figures(all_results, schema_results)
    except Exception as e:
        print(f"  Figure error: {e}", flush=True)
        import traceback; traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"Figures: figures/schema/", flush=True)


if __name__ == '__main__':
    _run()
