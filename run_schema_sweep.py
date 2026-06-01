"""Run full schema sweep and save results."""
import sys, os, time, json, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["DEV_MODE"] = "1"

import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True
ccf.N_WORKERS = 1

from schema_abstraction.schema_core import register_schema_hooks
from schema_abstraction.schema_analysis import run_all_schema_analysis
from schema_abstraction.schema_visualization import generate_all_schema_figures

register_schema_hooks()

n_mem = ccf.N_MEMORIES
asm_size = ccf.ASSEMBLY_SIZE

asm = ccf.make_overlapping_assemblies(n_mem, asm_size, 0.20)

conditions = [
    (False, False, "Fast / No Replay"),
    (False, True,  "Fast / Replay"),
    (True,  False, "Slow / No Replay"),
    (True,  True,  "Slow + Replay"),
]

all_results = []
for ci, (us, ur, label) in enumerate(conditions):
    trial_results = []
    for t in range(2):
        seed = ccf.MASTER_SEED + ci * 1000 + t
        t0 = time.time()
        r = ccf.run_sequential_experiment(us, ur, asm, seed)
        dt = time.time() - t0
        finals = r["final_scores"]
        hook_extra = r.get("hook_extra", {})
        print(f"[{label:20s}] trial {t+1}/2  {dt:6.1f}s  final={np.round(finals, 3)}", flush=True)
        trial_results.append({**r, "hook_extra": hook_extra})
    all_results.append({"cond": {"label": label, "use_slow": us, "use_replay": ur}, "trials": trial_results})

# Attach schema data
for res in all_results:
    for t in res.get("trials", []):
        extra = t.pop("hook_extra", None)
        if extra:
            for k, v in extra.items():
                t[k] = v
        else:
            for k in ["centroid_snapshots", "distance_trajectories", "schema_convergence",
                       "downscale_summary", "generative_layer", "generalization",
                       "anti_prediction", "metaplasticity", "hidden_state"]:
                t[k] = None if k != "centroid_snapshots" else []

# Analysis + figures
schema_results = run_all_schema_analysis(all_results, verbose=True)
try:
    generate_all_schema_figures(all_results, schema_results)
except Exception as e:
    print(f"[SKIP] figures: {e}", flush=True)

# Save results
out = {"all_results": all_results, "schema_results": schema_results}
with open("schema_sweep_results.pkl", "wb") as f:
    pickle.dump(out, f)
print("\n=== DONE ===", flush=True)
print("Results saved to schema_sweep_results.pkl", flush=True)
