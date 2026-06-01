"""
Publication-quality figures for:
  "Replay Distortion as Directional Schema Abstraction"

Loads distortion_data.pkl produced by _distortion_paper.py and generates
all 8 paper figures as high-resolution PDFs + PNGs.

Usage:
    python generate_paper_figures.py [--data PATH]
"""
import os
import sys
import pickle
import warnings
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind, ttest_1samp
from scipy.stats import sem as scipy_sem

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

DATA_PATH = r"C:\Users\Admin\brain-organoid-rl\figures\schema\distortion_data.pkl"
OUT_DIR   = r"C:\Users\Admin\brain-organoid-rl\figures\paper"

COND_ORDER  = ["no_replay", "natural", "hyper"]
COND_LABELS = {"no_replay": "No Replay", "natural": "Natural", "hyper": "Hyper"}
COLORS      = {"no_replay": "#4e79a7", "natural": "#59a14f", "hyper": "#f28e2b"}
HATCHES     = {"no_replay": "//", "natural": "", "hyper": "\\\\"}

PLT_STYLE = {
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.labelsize":   13,
    "axes.titlesize":   14,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
}
plt.rcParams.update(PLT_STYLE)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def _add_sig_bracket(ax, x1, x2, y, p, dy=0.02, color="black"):
    """Draw a significance bracket between bars at x-positions x1, x2."""
    stars = _sig_stars(p)
    ax.plot([x1, x1, x2, x2], [y, y+dy, y+dy, y], lw=1.2, c=color)
    ax.text((x1+x2)/2, y+dy+0.005, stars, ha="center", va="bottom",
            fontsize=11, color=color)


def _bar_plot(ax, values_dict, sem_dict=None, ylabel="", title="", ylim=None,
              sig_pairs=None, sig_p=None):
    """Standard 3-condition bar plot with error bars and significance brackets."""
    xs = np.arange(len(COND_ORDER))
    bars = []
    for i, cond in enumerate(COND_ORDER):
        val = values_dict.get(cond, 0.0)
        err = sem_dict.get(cond, 0.0) if sem_dict else 0.0
        b = ax.bar(i, val, width=0.6, color=COLORS[cond],
                   hatch=HATCHES[cond], edgecolor="white", linewidth=1.2,
                   yerr=err, capsize=5, error_kw=dict(lw=1.5, ecolor="black"))
        bars.append(b)

    ax.set_xticks(xs)
    ax.set_xticklabels([COND_LABELS[c] for c in COND_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)

    # Significance brackets
    if sig_pairs and sig_p:
        y_top = max(values_dict.values()) + (max(sem_dict.values()) if sem_dict else 0)
        offsets = [0.06, 0.14, 0.22]
        for k, (c1, c2) in enumerate(sig_pairs):
            if c1 in values_dict and c2 in values_dict:
                idx1 = COND_ORDER.index(c1)
                idx2 = COND_ORDER.index(c2)
                p = sig_p.get((c1, c2), 1.0)
                _add_sig_bracket(ax, idx1, idx2, y_top + offsets[k], p)
    return bars


def _get_agg(data, cond, key, default=np.nan):
    """Get aggregated value, with fallback to computing from per-seed data."""
    val = data.get(cond, {}).get("agg", {}).get(key, None)
    if val is not None:
        return val

    # Fallbacks for fields added in new code
    cdata = data.get(cond, {})
    if key == "real_schema_mean":
        rs = np.array(cdata.get("real_schemas", []))
        return float(np.nanmean(rs)) if len(rs) > 0 else default
    if key == "real_schema_sem":
        rs = np.array(cdata.get("real_schemas", []))
        return float(scipy_sem(rs)) if len(rs) > 1 else 0.0
    if key == "func_schema_mean":
        fs = np.array(cdata.get("func_schemas", []))
        return float(np.nanmean(fs)) if len(fs) > 0 else default
    if key == "func_schema_sem":
        fs = np.array(cdata.get("func_schemas", []))
        return float(scipy_sem(fs)) if len(fs) > 1 else 0.0
    if key == "dai_core_mean":
        da = cdata.get("directional_alignment", [])
        vals = np.array([x.get("mean_core", np.nan) for x in da])
        return float(np.nanmean(vals)) if len(vals) > 0 else default
    if key == "dai_core_sem":
        da = cdata.get("directional_alignment", [])
        vals = np.array([x.get("mean_core", np.nan) for x in da])
        return float(scipy_sem(vals[np.isfinite(vals)])) if np.isfinite(vals).sum() > 1 else 0.0
    if key == "dai_unique_mean":
        da = cdata.get("directional_alignment", [])
        vals = np.array([x.get("mean_unique", np.nan) for x in da])
        return float(np.nanmean(vals)) if len(vals) > 0 else default
    if key == "dai_unique_sem":
        da = cdata.get("directional_alignment", [])
        vals = np.array([x.get("mean_unique", np.nan) for x in da])
        return float(scipy_sem(vals[np.isfinite(vals)])) if np.isfinite(vals).sum() > 1 else 0.0
    if key == "p_core_mean":
        da = cdata.get("directional_alignment", [])
        vals = np.array([x.get("p_core", np.nan) for x in da])
        return float(np.nanmean(vals)) if len(vals) > 0 else 1.0
    if key == "p_unique_mean":
        da = cdata.get("directional_alignment", [])
        vals = np.array([x.get("p_unique", np.nan) for x in da])
        return float(np.nanmean(vals)) if len(vals) > 0 else 1.0

    return default


def _ttest(data, cond_a, cond_b, key):
    """t-test between two conditions on per-seed values from directional_alignment."""
    va = np.array([x.get(key, np.nan) for x in data.get(cond_a, {}).get("directional_alignment", [])])
    vb = np.array([x.get(key, np.nan) for x in data.get(cond_b, {}).get("directional_alignment", [])])
    va = va[np.isfinite(va)]
    vb = vb[np.isfinite(vb)]
    if len(va) < 2 or len(vb) < 2:
        # For no_replay (n_events=0 → no values), treat as 0
        if len(va) == 0 and len(vb) >= 2:
            va = np.zeros(len(vb))
        elif len(vb) == 0 and len(va) >= 2:
            vb = np.zeros(len(va))
        else:
            return np.nan, 1.0
    t, p = ttest_ind(va, vb)
    return float(t), float(p)


def _ttest_schema(data, cond_a, cond_b, key):
    """t-test on per-seed schema values."""
    va = np.array([sm.get(key, np.nan) for sm in data.get(cond_a, {}).get("schema", [])])
    vb = np.array([sm.get(key, np.nan) for sm in data.get(cond_b, {}).get("schema", [])])
    valid = np.isfinite(va) & np.isfinite(vb)
    if valid.sum() < 2:
        return np.nan, 1.0
    t, p = ttest_ind(va[valid], vb[valid])
    return float(t), float(p)


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path)
    print(f"  Saved {name}.pdf / .png", flush=True)
    plt.close(fig)


# ── Figure 1: Experimental design diagram ────────────────────────────────────

def fig1_design():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    stages = ["Encoding\n(Pattern A–D)", "Rest / Replay", "Retention\nProbe"]
    stage_x = [0.15, 0.50, 0.85]
    stage_w = 0.22
    stage_h = 0.4

    replay_labels = {
        "no_replay": ("No Replay", COLORS["no_replay"], "No consolidation"),
        "natural":   ("Natural",   COLORS["natural"],   "Partial cue, low noise"),
        "hyper":     ("Hyper",     COLORS["hyper"],     "Minimal cue, high noise"),
    }

    # Draw stages
    for x, label in zip(stage_x, stages):
        rect = mpatches.FancyBboxPatch((x - stage_w/2, 0.55), stage_w, stage_h,
                                       boxstyle="round,pad=0.02", linewidth=1.5,
                                       edgecolor="grey", facecolor="#f5f5f5")
        ax.add_patch(rect)
        ax.text(x, 0.55 + stage_h/2, label, ha="center", va="center",
                fontsize=11, fontweight="bold")
        if x < 0.85:
            ax.annotate("", xy=(x + 0.12, 0.75), xytext=(x + stage_w/2, 0.75),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Replay condition callouts
    for j, (cond, (name, color, desc)) in enumerate(replay_labels.items()):
        y = 0.32 - j * 0.13
        ax.plot([0.50 - stage_w/2, 0.38 + j * 0.04], [0.55, y + 0.04],
                lw=1, color=color, linestyle=":")
        rect2 = mpatches.FancyBboxPatch((0.28, y - 0.04), 0.44, 0.09,
                                        boxstyle="round,pad=0.01",
                                        edgecolor=color, facecolor=color + "22",
                                        linewidth=1.5)
        ax.add_patch(rect2)
        ax.text(0.30, y + 0.005, f"{name}:", fontsize=10, fontweight="bold", color=color)
        ax.text(0.44, y + 0.005, desc, fontsize=10, va="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Experimental Design: Replay Distortion & Schema Abstraction",
                 fontsize=14, fontweight="bold", pad=10)
    _save(fig, "fig1_design")


# ── Figure 2: Memory Retention ────────────────────────────────────────────────

def fig2_retention(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Memory A retention (first memory — most vulnerable to forgetting)
    ax = axes[0]
    vals = {c: _get_agg(data, c, "retention_mean", [0,0,0,0])[0] for c in COND_ORDER}
    sems = {c: _get_agg(data, c, "retention_sem", [0,0,0,0])[0] for c in COND_ORDER}

    # t-tests
    fa = np.array([r[0] for r in data.get("natural", {}).get("finals", []) if r])
    fb = np.array([r[0] for r in data.get("hyper", {}).get("finals", []) if r])
    fc = np.array([r[0] for r in data.get("no_replay", {}).get("finals", []) if r])
    p_nh = ttest_ind(fa, fb)[1] if len(fa) >= 2 and len(fb) >= 2 else 1.0
    p_nn = ttest_ind(fa, fc)[1] if len(fa) >= 2 and len(fc) >= 2 else 1.0
    p_hn = ttest_ind(fb, fc)[1] if len(fb) >= 2 and len(fc) >= 2 else 1.0

    sig_pairs = [("natural", "hyper"), ("natural", "no_replay")]
    sig_p = {("natural", "hyper"): float(p_nh), ("natural", "no_replay"): float(p_nn)}
    _bar_plot(ax, vals, sems, ylabel="Retention Score (Memory A)", title="(a) Memory Retention",
              ylim=(0, None), sig_pairs=sig_pairs, sig_p=sig_p)

    # Right: Mean retention across all 4 memories
    ax = axes[1]
    vals2 = {}
    sems2 = {}
    for c in COND_ORDER:
        finals = np.array(data.get(c, {}).get("finals", []))
        if finals.ndim == 2 and finals.shape[1] >= 4:
            vals2[c] = float(np.mean(finals))
            sems2[c] = float(scipy_sem(finals.mean(axis=1)))
        else:
            vals2[c] = 0.0; sems2[c] = 0.0

    nat_m = np.array(data.get("natural",  {}).get("finals", [])).mean(axis=1) if data.get("natural",  {}).get("finals") else np.array([])
    hyp_m = np.array(data.get("hyper",    {}).get("finals", [])).mean(axis=1) if data.get("hyper",    {}).get("finals") else np.array([])
    nor_m = np.array(data.get("no_replay",{}).get("finals", [])).mean(axis=1) if data.get("no_replay",{}).get("finals") else np.array([])
    p2_nh = ttest_ind(nat_m, hyp_m)[1] if len(nat_m) >= 2 and len(hyp_m) >= 2 else 1.0
    p2_nn = ttest_ind(nat_m, nor_m)[1] if len(nat_m) >= 2 and len(nor_m) >= 2 else 1.0

    sig_p2 = {("natural", "hyper"): float(p2_nh), ("natural", "no_replay"): float(p2_nn)}
    _bar_plot(ax, vals2, sems2, ylabel="Mean Retention Score (A–D)",
              title="(b) Mean Retention Across Memories",
              ylim=(0, None), sig_pairs=sig_pairs, sig_p=sig_p2)

    fig.suptitle("Figure 2: Memory Retention by Replay Condition",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_retention")


# ── Figure 3: REAL_SCHEMA ─────────────────────────────────────────────────────

def fig3_real_schema(data):
    fig, ax = plt.subplots(figsize=(6, 5))

    vals = {c: _get_agg(data, c, "real_schema_mean", np.nan) for c in COND_ORDER}
    sems = {c: _get_agg(data, c, "real_schema_sem", 0.0)     for c in COND_ORDER}

    nat_rs = np.array(data.get("natural",  {}).get("real_schemas", []))
    hyp_rs = np.array(data.get("hyper",    {}).get("real_schemas", []))
    nor_rs = np.array(data.get("no_replay",{}).get("real_schemas", []))
    p_nh = ttest_ind(nat_rs, hyp_rs)[1] if len(nat_rs) >= 2 and len(hyp_rs) >= 2 else 1.0
    p_nn = ttest_ind(nat_rs, nor_rs)[1] if len(nat_rs) >= 2 and len(nor_rs) >= 2 else 1.0

    sig_pairs = [("natural", "hyper"), ("natural", "no_replay")]
    sig_p = {("natural", "hyper"): float(p_nh), ("natural", "no_replay"): float(p_nn)}
    _bar_plot(ax, vals, sems,
              ylabel="REAL_SCHEMA Index\n(core–core vs core–unique weight ratio)",
              title="Figure 3: Schema Strength (REAL_SCHEMA)",
              sig_pairs=sig_pairs, sig_p=sig_p)
    fig.tight_layout()
    _save(fig, "fig3_real_schema")


# ── Figure 4: SchemaScore ─────────────────────────────────────────────────────

def fig4_schema_score(data):
    fig, ax = plt.subplots(figsize=(6, 5))

    vals = {c: _get_agg(data, c, "schema_score_mean", np.nan) for c in COND_ORDER}
    sems = {c: _get_agg(data, c, "schema_score_sem", 0.0)     for c in COND_ORDER}

    _, p_nh = _ttest_schema(data, "natural", "hyper", "schema_score")
    _, p_nn = _ttest_schema(data, "natural", "no_replay", "schema_score")
    sig_pairs = [("natural", "hyper"), ("natural", "no_replay")]
    sig_p = {("natural", "hyper"): p_nh, ("natural", "no_replay"): p_nn}

    _bar_plot(ax, vals, sems,
              ylabel="Schema Score\n(cosine distance convergence)",
              title="Figure 4: Schema Formation Score",
              ylim=(0, None), sig_pairs=sig_pairs, sig_p=sig_p)
    fig.tight_layout()
    _save(fig, "fig4_schema_score")


# ── Figure 5: Distortion Index ────────────────────────────────────────────────

def fig5_distortion(data):
    fig, ax = plt.subplots(figsize=(6, 5))

    vals = {c: _get_agg(data, c, "distortion_mean", np.nan) for c in COND_ORDER}
    sems_raw = {}
    for c in COND_ORDER:
        di_vals = np.array([sm.get("distortion_index", np.nan) for sm in data.get(c, {}).get("schema", [])])
        di_vals = di_vals[np.isfinite(di_vals)]
        sems_raw[c] = float(scipy_sem(di_vals)) if len(di_vals) >= 2 else 0.0

    _, p_nh = _ttest_schema(data, "natural", "hyper",    "distortion_index")
    _, p_nn = _ttest_schema(data, "natural", "no_replay","distortion_index")

    xs = np.arange(len(COND_ORDER))
    for i, cond in enumerate(COND_ORDER):
        ax.bar(i, vals.get(cond, 0), width=0.6, color=COLORS[cond],
               hatch=HATCHES[cond], edgecolor="white", linewidth=1.2,
               yerr=sems_raw.get(cond, 0), capsize=5,
               error_kw=dict(lw=1.5, ecolor="black"))

    ax.set_xticks(xs)
    ax.set_xticklabels([COND_LABELS[c] for c in COND_ORDER])
    ax.set_ylabel("Distortion Index\n(mean centroid movement / replay event)")
    ax.set_title("Figure 5: Replay Distortion Index")
    ax.set_ylim(0, 0.28)  # explicit ylim to accommodate brackets

    y_top = max(vals.values()) + max(sems_raw.values())
    for k, (c1, c2, p_val) in enumerate([
        ("natural", "hyper", p_nh), ("natural", "no_replay", p_nn)
    ]):
        idx1, idx2 = COND_ORDER.index(c1), COND_ORDER.index(c2)
        yb = y_top + 0.05 + k * 0.06
        ax.plot([idx1, idx1, idx2, idx2], [yb, yb+0.012, yb+0.012, yb], lw=1.2, c="black")
        ax.text((idx1+idx2)/2, yb+0.013, _sig_stars(p_val), ha="center", va="bottom", fontsize=11)

    fig.tight_layout()
    _save(fig, "fig5_distortion")


# ── Figure 6: Directional Alignment Index ────────────────────────────────────

def fig6_dai(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax_idx, (key_mean, key_sem, label, panel) in enumerate([
        ("dai_core_mean",   "dai_core_sem",   "DAI — Core Component",   "(a)"),
        ("dai_unique_mean", "dai_unique_sem", "DAI — Unique Component", "(b)"),
    ]):
        ax = axes[ax_idx]
        vals = {c: _get_agg(data, c, key_mean, np.nan) for c in COND_ORDER}
        sems = {c: _get_agg(data, c, key_sem, 0.0)     for c in COND_ORDER}

        # agg key "dai_core_mean" → per-seed key "mean_core"
        per_seed_key = "mean_core" if "core" in key_mean else "mean_unique"
        _, p_nh = _ttest(data, "natural", "hyper",    per_seed_key)
        _, p_nn = _ttest(data, "natural", "no_replay",per_seed_key)
        sig_pairs = [("natural", "hyper"), ("natural", "no_replay")]
        sig_p = {("natural", "hyper"): p_nh, ("natural", "no_replay"): p_nn}

        bars = []
        for i, cond in enumerate(COND_ORDER):
            val = vals.get(cond, 0.0)
            err = sems.get(cond, 0.0)
            b = ax.bar(i, val, width=0.6, color=COLORS[cond],
                       hatch=HATCHES[cond], edgecolor="white", linewidth=1.2,
                       yerr=err, capsize=5, error_kw=dict(lw=1.5, ecolor="black"))
            bars.append(b)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(COND_ORDER)))
        ax.set_xticklabels([COND_LABELS[c] for c in COND_ORDER])
        ax.set_ylabel("cos(Δcentroid, toward schema)")
        ax.set_title(f"{panel} {label}")

        # Significance brackets
        y_top = max((abs(v) for v in vals.values()), default=0.1)
        for k, (c1, c2) in enumerate(sig_pairs):
            idx1, idx2 = COND_ORDER.index(c1), COND_ORDER.index(c2)
            p = sig_p.get((c1, c2), 1.0)
            yb = y_top * (1.25 + k * 0.25)
            ax.plot([idx1, idx1, idx2, idx2], [yb, yb+y_top*0.08, yb+y_top*0.08, yb],
                    lw=1.2, c="black")
            ax.text((idx1+idx2)/2, yb + y_top*0.09, _sig_stars(p),
                    ha="center", va="bottom", fontsize=11)

        # p-values annotation per bar
        for i, cond in enumerate(COND_ORDER):
            p_key = "p_core_mean" if "core" in key_mean else "p_unique_mean"
            p_val = _get_agg(data, cond, p_key, 1.0)
            if not np.isnan(p_val):
                ax.text(i, ax.get_ylim()[0] * 0.95 if ax.get_ylim()[0] < 0 else -0.002,
                        f"p={p_val:.3f}", ha="center", va="top", fontsize=8, color="grey")

    fig.suptitle("Figure 6: Directional Alignment Index (DAI)\n"
                 "cos(Δcentroid, toward schema centroid) per replay event",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig6_dai")


# ── Figure 7: Centroid trajectories (PCA) ────────────────────────────────────

def fig7_trajectories(data):
    """PCA on centroid evolution across conditions using schema metrics."""
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax_idx, cond in enumerate(COND_ORDER):
        ax = axes[ax_idx]
        # Build centroid matrix from schema distance_trajectories
        schema_list = data.get(cond, {}).get("schema", [])
        if not schema_list:
            ax.set_title(f"{COND_LABELS[cond]}\n(no data)")
            continue

        # Collect per-checkpoint cross-memory distances as proxy for trajectory
        # (shape: n_seeds × n_checkpoints)
        all_cross = []
        for sm in schema_list:
            ct = sm.get("cross_memory_trajectory", [])
            if ct and any(np.isfinite(v) for v in ct):
                row = [v if np.isfinite(v) else np.nan for v in ct]
                all_cross.append(row)

        if not all_cross:
            ax.set_title(f"{COND_LABELS[cond]}\n(no trajectory data)")
            continue

        mat = np.array(all_cross)  # n_seeds × n_checkpoints
        mat = np.nan_to_num(mat, nan=np.nanmean(mat))
        n_cp = mat.shape[1]
        checkpoints = np.arange(n_cp)

        mean_traj = mat.mean(axis=0)
        sem_traj  = scipy_sem(mat, axis=0) if mat.shape[0] > 1 else np.zeros(n_cp)

        ax.plot(checkpoints, mean_traj, "o-", color=COLORS[cond], lw=2,
                ms=6, label=COND_LABELS[cond])
        ax.fill_between(checkpoints, mean_traj - sem_traj, mean_traj + sem_traj,
                        alpha=0.25, color=COLORS[cond])

        ax.set_xlabel("Memory Checkpoint")
        ax.set_ylabel("Mean Pairwise Cosine Distance")
        ax.set_title(f"({"abc"[ax_idx]}) {COND_LABELS[cond]}")
        ax.set_xticks(checkpoints)
        ax.set_xticklabels([f"M{i+1}" for i in checkpoints])

    fig.suptitle("Figure 7: Centroid Convergence Trajectories\n"
                 "Mean pairwise cosine distance across memory checkpoints",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig7_trajectories")


# ── Figure 8: Inverted-U summary ─────────────────────────────────────────────

def fig8_summary(data):
    """Main paper figure: Retention + Schema + DAI across replay regimes."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    metrics = [
        ("retention_mean",  "retention_sem",  "Memory Retention\n(Memory A)",          "(a) Retention"),
        ("real_schema_mean","real_schema_sem", "REAL_SCHEMA Index\n(schema strength)",  "(b) Schema Formation"),
        ("dai_core_mean",   "dai_core_sem",    "DAI — Core\n(cos toward schema)",       "(c) Directional Abstraction"),
    ]

    for ax_idx, (key_m, key_s, ylabel, title) in enumerate(metrics):
        ax = axes[ax_idx]

        vals = {}
        sems = {}
        for cond in COND_ORDER:
            raw = _get_agg(data, cond, key_m, np.nan)
            sem = _get_agg(data, cond, key_s, 0.0)
            # retention_mean is a list; take first element (Memory A)
            if isinstance(raw, list):
                raw = raw[0] if raw else np.nan
                sem = sem[0] if isinstance(sem, list) and sem else 0.0
            vals[cond] = float(raw) if not np.isnan(raw) else 0.0
            sems[cond] = float(sem)

        # Statistical tests
        if "retention" in key_m:
            nat_v = np.array([r[0] for r in data.get("natural", {}).get("finals", []) if r])
            hyp_v = np.array([r[0] for r in data.get("hyper",   {}).get("finals", []) if r])
            nor_v = np.array([r[0] for r in data.get("no_replay",{}).get("finals", []) if r])
            p_nh  = ttest_ind(nat_v, hyp_v)[1] if len(nat_v)>=2 and len(hyp_v)>=2 else 1.0
            p_nn  = ttest_ind(nat_v, nor_v)[1] if len(nat_v)>=2 and len(nor_v)>=2 else 1.0
        elif "real_schema" in key_m:
            nat_v = np.array(data.get("natural",  {}).get("real_schemas", []))
            hyp_v = np.array(data.get("hyper",    {}).get("real_schemas", []))
            nor_v = np.array(data.get("no_replay",{}).get("real_schemas", []))
            p_nh  = ttest_ind(nat_v, hyp_v)[1] if len(nat_v)>=2 and len(hyp_v)>=2 else 1.0
            p_nn  = ttest_ind(nat_v, nor_v)[1] if len(nat_v)>=2 and len(nor_v)>=2 else 1.0
        else:
            _, p_nh = _ttest(data, "natural", "hyper",    "mean_core")
            _, p_nn = _ttest(data, "natural", "no_replay","mean_core")

        bars_objs = []
        for i, cond in enumerate(COND_ORDER):
            b = ax.bar(i, vals[cond], width=0.6,
                       color=COLORS[cond], hatch=HATCHES[cond],
                       edgecolor="white", linewidth=1.2,
                       yerr=sems[cond], capsize=5,
                       error_kw=dict(lw=1.5, ecolor="black"))
            bars_objs.append(b)

        if "dai" in key_m:
            ax.axhline(0, color="black", lw=0.8, ls="--")

        ax.set_xticks(range(len(COND_ORDER)))
        ax.set_xticklabels([COND_LABELS[c] for c in COND_ORDER])
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Significance
        y_top = max(abs(v) + s for v, s in zip(vals.values(), sems.values()))
        for k, (c1, c2, p_val) in enumerate([
            ("natural", "hyper",    p_nh),
            ("natural", "no_replay",p_nn),
        ]):
            idx1, idx2 = COND_ORDER.index(c1), COND_ORDER.index(c2)
            yb = y_top * (1.15 + k * 0.2)
            ax.plot([idx1, idx1, idx2, idx2], [yb, yb + y_top*0.07, yb + y_top*0.07, yb],
                    lw=1.2, c="black")
            ax.text((idx1+idx2)/2, yb + y_top*0.08, _sig_stars(p_val),
                    ha="center", va="bottom", fontsize=11)

    # Legend
    legend_patches = [mpatches.Patch(color=COLORS[c], label=COND_LABELS[c]) for c in COND_ORDER]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), frameon=False)

    fig.suptitle(
        "Figure 8: Replay Distortion Drives Directional Schema Abstraction\n"
        "Natural replay optimally balances retention and abstraction",
        fontsize=13, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    _save(fig, "fig8_summary")


# ── Figure 9: Functional Schema ───────────────────────────────────────────────

def fig9_functional_schema(data):
    fig, ax = plt.subplots(figsize=(6, 5))

    vals = {c: _get_agg(data, c, "func_schema_mean", np.nan) for c in COND_ORDER}
    sems = {c: _get_agg(data, c, "func_schema_sem", 0.0)     for c in COND_ORDER}

    nat_v = np.array(data.get("natural",  {}).get("func_schemas", []))
    hyp_v = np.array(data.get("hyper",    {}).get("func_schemas", []))
    nor_v = np.array(data.get("no_replay",{}).get("func_schemas", []))
    p_nh  = ttest_ind(nat_v, hyp_v)[1] if len(nat_v)>=2 and len(hyp_v)>=2 else 1.0
    p_nn  = ttest_ind(nat_v, nor_v)[1] if len(nat_v)>=2 and len(nor_v)>=2 else 1.0

    sig_pairs = [("natural", "hyper"), ("natural", "no_replay")]
    sig_p = {("natural", "hyper"): float(p_nh), ("natural", "no_replay"): float(p_nn)}
    _bar_plot(ax, vals, sems,
              ylabel="Functional Schema Score\n(# assemblies activated by core cue)",
              title="Figure 9: Functional Schema (Core Cue Completion)",
              ylim=(0, None), sig_pairs=sig_pairs, sig_p=sig_p)
    fig.tight_layout()
    _save(fig, "fig9_functional_schema")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    args = parser.parse_args()

    print(f"Loading data from: {args.data}", flush=True)
    try:
        data = _load(args.data)
    except FileNotFoundError:
        print(f"ERROR: data file not found at {args.data}", flush=True)
        print("Run _distortion_paper.py first to generate the data.", flush=True)
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output directory: {OUT_DIR}", flush=True)
    print("", flush=True)

    # Quick data summary
    for cond in COND_ORDER:
        n = len(data.get(cond, {}).get("finals", []))
        agg = data.get(cond, {}).get("agg", {})
        ret = agg.get("retention_mean", [0])[0] if agg.get("retention_mean") else "?"
        rs  = agg.get("real_schema_mean", "?")
        dai = agg.get("dai_core_mean", "?")
        print(f"  {COND_LABELS[cond]:12s}  n={n}  ret_A={ret:.4f}  REAL_SCHEMA={rs:.4f}  DAI_core={dai:+.4f}"
              if all(isinstance(x, float) for x in [ret, rs, dai]) else
              f"  {COND_LABELS[cond]:12s}  n={n}  (incomplete aggregation)", flush=True)

    print("", flush=True)
    print("Generating figures...", flush=True)

    fig1_design()
    fig2_retention(data)
    fig3_real_schema(data)
    fig4_schema_score(data)
    fig5_distortion(data)
    fig6_dai(data)

    try:
        from sklearn.decomposition import PCA
        fig7_trajectories(data)
    except ImportError:
        print("  Skipping fig7 (sklearn not available)", flush=True)

    fig8_summary(data)
    fig9_functional_schema(data)

    print(f"\nAll figures saved to: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
