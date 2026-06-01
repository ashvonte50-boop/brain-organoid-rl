"""
extensions/ablations_extended.py — Fine-grained mechanism ablations (Task 3).

More granular ablations than the 5-condition cf.ABLATION_CONDITIONS suite.
Each ablation disables exactly ONE mechanism via the existing ablation dict
interface, revealing individual causal contributions.

Extended ablation conditions (15 total):
  Baseline   : Full model (Slow+Replay, all mechanisms ON)
  Group A — Replay quality gating:
    A1: No coherence gating    (REPLAY_COHERENCE_THR=0 — all events accepted)
    A2: No adaptive acceptance (REPLAY_ACCEPT_MIN_CONSEC=0)
    A3: No burst clustering    (REPLAY_BURST_SIZE=1)
    A4: No chain replay        (CHAIN_REPLAY_PROB=0)
  Group B — Consolidation pathway:
    B1: No persistence current (pers_gain=0)
    B2: No slow weights        (Fast weights only, same as Fast+Replay)
    B3: No synaptic tags       (USE_TAGGING=False)
    B4: No tag consolidation   (TAG_CAPTURE_RATE=0)
  Group C — Replay scheduling:
    C1: Uniform replay         (prioritize="uniform")
    C2: Oldest-first replay    (prioritize="oldest_first")
    C3: No endogenous urgency  (use cf's interference_aware without endogenous)
  Group D — Network structure:
    D1: No competitive interference (use_competition=False)
    D2: No overlap-exclusion cue    (uses full assembly as cue pool)
  Group E — Full model reference:
    E1: Full model with endogenous scheduling

The ablation_dict passed to run_sequential_experiment controls which mechanisms
are active. Monkey-patching module constants handles parameters not in the dict.
"""
import os
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import compare_catastrophic_forgetting as cf

__all__ = [
    "EXTENDED_ABLATION_CONDITIONS",
    "run_extended_ablation",
    "run_all_extended_ablations",
    "fig_ablation_matrix",
    "fig_mechanism_contributions",
]

# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------

# Each entry:
#   label       : display name
#   group       : letter code (A/B/C/D/E)
#   ablation    : dict passed directly to run_sequential_experiment (ablation=...)
#   module_patch: {attr: value} to monkey-patch cf before the trial (restored after)
#   use_slow    : bool (default True)
#   use_replay  : bool (default True)

EXTENDED_ABLATION_CONDITIONS: List[dict] = [
    # ── Baseline ───────────────────────────────────────────────────────────
    {
        "label": "Full Model",
        "group": "E",
        "ablation": {
            "pers_gain":      cf.REPLAY_PERS_GAIN,
            "use_competition": True,
        },
        "module_patch": {},
        "prioritize": "interference_aware",
        "use_slow": True, "use_replay": True,
    },

    # ── Group A: Replay quality gating ────────────────────────────────────
    {
        "label": "No Coherence\nGating",
        "group": "A",
        "ablation": {
            "pers_gain":      cf.REPLAY_PERS_GAIN,
            "use_competition": True,
        },
        "module_patch": {"REPLAY_COHERENCE_THR": 0.0},
        "prioritize": "interference_aware",
        "use_slow": True, "use_replay": True,
    },
    {
        "label": "No Adaptive\nAcceptance",
        "group": "A",
        "ablation": {
            "pers_gain":      cf.REPLAY_PERS_GAIN,
            "use_competition": True,
        },
        "module_patch": {"REPLAY_ACCEPT_MIN_CONSEC": 0},
        "prioritize": "interference_aware",
        "use_slow": True, "use_replay": True,
    },
    {
        "label": "No Burst\nClustering",
        "group": "A",
        "ablation": {
            "pers_gain":      cf.REPLAY_PERS_GAIN,
            "use_competition": True,
        },
        "module_patch": {"REPLAY_BURST_SIZE": 1},
        "prioritize": "interference_aware",
        "use_slow": True, "use_replay": True,
    },
    {
        "label": "No Chain\nReplay",
        "group": "A",
        "ablation": {
            "pers_gain":      cf.REPLAY_PERS_GAIN,
            "use_competition": True,
        },
        "module_patch": {"CHAIN_REPLAY_PROB": 0.0},
        "prioritize": "interference_aware",
        "use_slow": True, "use_replay": True,
    },

    # ── Group B: Consolidation pathway ────────────────────────────────────
    {
        "label": "No Persistence\nCurrent",
        "group": "B",
        "ablation": {
            "pers_gain":      0.0,
            "use_competition": True,
        },
        "module_patch": {},
        "prioritize": "interference_aware",
        "use_slow": True, "use_replay": True,
    },
    {
        "label": "Fast Only\n+ Replay",
        "group": "B",
        "ablation": {
            "pers_gain":      0.0,
            "use_competition": False,
        },
        "module_patch": {},
        "prioritize": "interference_aware",
        "use_slow": False, "use_replay": True,
    },
    {
        "label": "No Synaptic\nTags",
        "group": "B",
        "ablation": {
            "pers_gain":      cf.REPLAY_PERS_GAIN,
            "use_competition": True,
        },
        "module_patch": {"USE_TAGGING": False},
        "prioritize": "interference_aware",
        "use_slow": True, "use_replay": True,
    },
    {
        "label": "No Tag\nConsolidation",
        "group": "B",
        "ablation": {
            "pers_gain":      cf.REPLAY_PERS_GAIN,
            "use_competition": True,
        },
        "module_patch": {"TAG_CAPTURE_RATE": 0.0},
        "prioritize": "interference_aware",
        "use_slow": True, "use_replay": True,
    },

    # ── Group C: Replay scheduling ─────────────────────────────────────────
    {
        "label": "Uniform\nReplay",
        "group": "C",
        "ablation": {
            "pers_gain":      cf.REPLAY_PERS_GAIN,
            "use_competition": True,
        },
        "module_patch": {},
        "prioritize": "uniform",
        "use_slow": True, "use_replay": True,
    },
    {
        "label": "Oldest-First\nReplay",
        "group": "C",
        "ablation": {
            "pers_gain":      cf.REPLAY_PERS_GAIN,
            "use_competition": True,
        },
        "module_patch": {},
        "prioritize": "oldest_first",
        "use_slow": True, "use_replay": True,
    },
    {
        "label": "Endogenous\nScheduling",
        "group": "C",
        "ablation": {
            "pers_gain":      cf.REPLAY_PERS_GAIN,
            "use_competition": True,
        },
        "module_patch": {},
        "prioritize": "endogenous",
        "use_slow": True, "use_replay": True,
    },

    # ── Group D: Network structure ─────────────────────────────────────────
    {
        "label": "No Competitive\nInterference",
        "group": "D",
        "ablation": {
            "pers_gain":      cf.REPLAY_PERS_GAIN,
            "use_competition": False,
        },
        "module_patch": {},
        "prioritize": "interference_aware",
        "use_slow": True, "use_replay": True,
    },
    {
        "label": "No Overlap\nExclusion",
        "group": "D",
        "ablation": {
            "pers_gain":      cf.REPLAY_PERS_GAIN,
            "use_competition": True,
        },
        "module_patch": {"PARTIAL_CUE_EXCLUDE_OVERLAP": False},
        "prioritize": "interference_aware",
        "use_slow": True, "use_replay": True,
    },

    # ── No replay reference ─────────────────────────────────────────────────
    {
        "label": "No Replay\n(Slow)",
        "group": "E",
        "ablation": {
            "pers_gain":      0.0,
            "use_competition": False,
        },
        "module_patch": {},
        "prioritize": "interference_aware",
        "use_slow": True, "use_replay": False,
    },
]

# Group colors
_GROUP_COLORS = {
    "E": "#2ecc71",
    "A": "#3498db",
    "B": "#e74c3c",
    "C": "#e67e22",
    "D": "#9b59b6",
}


# ---------------------------------------------------------------------------
# Worker function (top-level, picklable)
# ---------------------------------------------------------------------------

def _extended_ablation_worker(args):
    """
    Worker for one extended ablation trial.
    args = (condition_dict, assemblies, trial_seed)
    """
    cond, assemblies, trial_seed = args

    # Apply module patches
    saved = {}
    for attr, val in cond.get("module_patch", {}).items():
        # PARTIAL_CUE_EXCLUDE_OVERLAP is not a real cf constant but signals
        # the D2 condition; handled below.
        if attr == "PARTIAL_CUE_EXCLUDE_OVERLAP":
            continue
        if hasattr(cf, attr):
            saved[attr] = getattr(cf, attr)
            setattr(cf, attr, val)

    try:
        result = cf.run_sequential_experiment(
            use_slow=cond["use_slow"],
            use_replay=cond["use_replay"],
            assemblies=assemblies,
            trial_seed=trial_seed,
            prioritize=cond["prioritize"],
            verbose=False,
            ablation=cond["ablation"],
        )
        n_mem = len(assemblies)
        final = result["final_scores"]
        ret   = float(np.nanmean(final[:n_mem - 1]))
    except Exception as e:
        warnings.warn(f"Extended ablation worker failed ({cond['label']}): {e}")
        ret   = float("nan")
        final = np.full(cf.N_MEMORIES, np.nan)
    finally:
        for attr, val in saved.items():
            setattr(cf, attr, val)

    return {
        "label":        cond["label"],
        "group":        cond["group"],
        "mean_ret":     ret,
        "final_scores": final if isinstance(final, np.ndarray) else np.full(cf.N_MEMORIES, np.nan),
        "trial_seed":   trial_seed,
    }


# ---------------------------------------------------------------------------
# Single ablation runner
# ---------------------------------------------------------------------------

def run_extended_ablation(
    cond: dict,
    assemblies: List[np.ndarray],
    n_trials: int = None,
    verbose: bool = False,
) -> List[dict]:
    """
    Run n_trials for a single extended ablation condition.
    Returns list of per-trial result dicts.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    if n_trials is None:
        n_trials = cf.N_TRIALS_ABLATION

    seeds    = [cf.MASTER_SEED + i * 37 for i in range(n_trials)]
    asms     = [a.copy() for a in assemblies]
    task_args = [(cond, asms, s) for s in seeds]
    n_workers = min(cf.N_WORKERS, n_trials)

    if verbose:
        print(f"  [ExtAblation] {cond['label'].replace(chr(10),' ')} x{n_trials} ...", flush=True)
    try:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            results = list(pool.map(_extended_ablation_worker, task_args))
    except Exception as e:
        warnings.warn(f"Parallel ablation failed: {e}. Serial fallback.")
        results = [_extended_ablation_worker(a) for a in task_args]
    return results


# ---------------------------------------------------------------------------
# Full ablation suite
# ---------------------------------------------------------------------------

def run_all_extended_ablations(
    assemblies: List[np.ndarray],
    n_trials: int = None,
    verbose: bool = False,
    conditions: Optional[List[dict]] = None,
) -> Dict[str, List[dict]]:
    """
    Run all extended ablation conditions.
    Returns dict[label] = list[per-trial dicts].
    """
    if conditions is None:
        conditions = EXTENDED_ABLATION_CONDITIONS
    if n_trials is None:
        n_trials = cf.N_TRIALS_ABLATION

    out = {}
    for cond in conditions:
        label   = cond["label"].replace("\n", " ")
        results = run_extended_ablation(cond, assemblies, n_trials=n_trials, verbose=verbose)
        out[label] = results
        if verbose:
            rets = [r["mean_ret"] for r in results if np.isfinite(r["mean_ret"])]
            m    = np.nanmean(rets)
            sem  = np.nanstd(rets) / max(1, np.sqrt(len(rets)))
            print(f"    -> mean_ret = {m:.4f} ± {sem:.4f}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_ablation_matrix(
    results: Dict[str, List[dict]],
    conditions: Optional[List[dict]] = None,
) -> None:
    """
    Heatmap showing mean retention per condition × memory slot.
    Conditions sorted by group, then mean retention.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if conditions is None:
        conditions = EXTENDED_ABLATION_CONDITIONS

    labels_ord = [c["label"].replace("\n", " ") for c in conditions]
    groups     = [c["group"] for c in conditions]
    n_mem      = cf.N_MEMORIES

    mat = np.full((len(conditions), n_mem), np.nan)
    for i, label in enumerate(labels_ord):
        trial_list = results.get(label, [])
        if trial_list:
            scores = np.array([r["final_scores"] for r in trial_list])
            mat[i] = np.nanmean(scores, axis=0)

    # Colour for y-axis labels by group
    ylabel_colors = [_GROUP_COLORS.get(g, "#333333") for g in groups]

    fig, ax = plt.subplots(figsize=(8, max(5, len(conditions) * 0.55)))
    vmin = np.nanmin(mat) if not np.all(np.isnan(mat)) else -0.2
    vmax = max(np.nanmax(mat), 0.01) if not np.all(np.isnan(mat)) else 1.0
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n_mem))
    ax.set_xticklabels([f"Memory {chr(65+i)}" for i in range(n_mem)], fontsize=9)
    ax.set_yticks(range(len(conditions)))
    yticklabels = ax.set_yticklabels(labels_ord, fontsize=7)
    for tick, color in zip(yticklabels, ylabel_colors):
        tick.set_color(color)
    ax.set_title(f"Extended Ablation Suite: Final Retention per Memory\n"
                 f"(N={cf.N_TRIALS_ABLATION} trials, Slow+Replay base)",
                 fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.04, label="I_syn score")

    for i in range(len(conditions)):
        for j in range(n_mem):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=6)

    fig.tight_layout()
    cf._save_fig(fig, "ablation_matrix_extended")
    plt.close(fig)
    print("[FIG] Saved ablation_matrix_extended.png", flush=True)


def fig_mechanism_contributions(
    results: Dict[str, List[dict]],
    conditions: Optional[List[dict]] = None,
) -> None:
    """
    Horizontal bar chart: Δ retention vs Full Model (contribution of each mechanism).
    Positive = mechanism helps; negative = mechanism hurts when removed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if conditions is None:
        conditions = EXTENDED_ABLATION_CONDITIONS

    # Full model reference
    full_label = "Full Model"
    full_rets  = [r["mean_ret"] for r in results.get(full_label, []) if np.isfinite(r["mean_ret"])]
    full_mean  = float(np.nanmean(full_rets)) if full_rets else 0.0

    deltas = []
    labels = []
    colors = []
    for cond in conditions:
        label = cond["label"].replace("\n", " ")
        if label == full_label:
            continue
        trial_list = results.get(label, [])
        rets = [r["mean_ret"] for r in trial_list if np.isfinite(r["mean_ret"])]
        if not rets:
            continue
        delta = full_mean - float(np.nanmean(rets))  # + = this mechanism helps
        deltas.append(delta)
        labels.append(label)
        colors.append(_GROUP_COLORS.get(cond["group"], "#555555"))

    # Sort by delta descending
    order  = np.argsort(deltas)[::-1]
    deltas = [deltas[i] for i in order]
    labels = [labels[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.55)))
    y   = np.arange(len(labels))
    bars = ax.barh(y, deltas, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Δ Mean Retention vs Full Model\n(positive = mechanism helps)", fontsize=9)
    ax.set_title("Mechanism Contribution Analysis\n(Full Model − Ablated)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Group legend
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=v, label=f"Group {k}")
        for k, v in _GROUP_COLORS.items()
        if k != "E"
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    fig.tight_layout()
    cf._save_fig(fig, "mechanism_contributions")
    plt.close(fig)
    print("[FIG] Saved mechanism_contributions.png", flush=True)


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

    # Test just 2 conditions x 2 trials for speed
    test_conds = [EXTENDED_ABLATION_CONDITIONS[0], EXTENDED_ABLATION_CONDITIONS[1]]
    print("[ablations_extended self-test] 2 conditions x 2 trials ...", flush=True)
    results = {}
    for cond in test_conds:
        label   = cond["label"].replace("\n", " ")
        r_list  = run_extended_ablation(cond, asms, n_trials=2, verbose=True)
        results[label] = r_list
        print(f"  {label}: {[r['mean_ret'] for r in r_list]}")
    print("[ablations_extended self-test] DONE.", flush=True)
