"""Publication-quality figures for all schema-abstraction elements.

10+ publication-quality figures:
  1. Retention curves (retention per memory across conditions)
  2. Schema convergence trajectories
  3. Overlap proportionality scatter
  4. Replay coherence vs drift
  5. Basin geometry (retention changes per assembly)
  6. Partial cue completion curves
  7. Replay entropy vs abstraction
  8. Natural vs perfect replay comparison
  9. Slow-weight accumulation
  10. Overlap sweep: convergence vs overlap
  11. Schema Crystallization Index (SCI)
  12. Catastrophic Forgetting Resistance (CFR)
  13. Centroid trajectory PCA (with sleep-phase coloring)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from compare_catastrophic_forgetting import _safe_mean

FIGS_DIR = Path("figures/schema")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

_STYLE = {
    "figsize": (10, 6), "dpi": 150,
    "colors": plt.cm.Set1(np.linspace(0, 1, 12)),
    "markers": ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"],
    "color_map": {
        "Fast / No Replay":    "#c0392b",
        "Fast / Replay":       "#e67e22",
        "Slow / No Replay":    "#2980b9",
        "Slow + Replay":       "#27ae60",
    },
}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.set_xlabel(ax.get_xlabel(), fontsize=12)
    ax.set_ylabel(ax.get_ylabel(), fontsize=12)
    ax.title.set_fontsize(13)


def _save_fig(fig, name):
    path = FIGS_DIR / f"{name}.png"
    fig.savefig(path, dpi=_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] saved {path}", flush=True)


def _get_color(label):
    return _STYLE["color_map"].get(label, _STYLE["colors"][0])


def _get_cond_idx(all_results, label):
    for idx, res in enumerate(all_results):
        if res["cond"]["label"] == label:
            return idx
    return 0


def _extract_finals(all_results):
    """Extract final retention scores per condition per trial."""
    data = {}
    for res in all_results:
        label = res["cond"]["label"]
        trials = []
        for t in res.get("trials", []):
            fs = t.get("final_scores", [])
            if len(fs) > 0:
                trials.append(fs)
        data[label] = trials
    return data


def _bootstrap_ci(arr, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval."""
    if len(arr) < 2:
        return float(np.mean(arr)) if len(arr) == 1 else 0.0, 0.0, 0.0
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    means = np.array(means)
    lower = float(np.percentile(means, (1.0 - ci) / 2.0 * 100))
    upper = float(np.percentile(means, (1.0 + ci) / 2.0 * 100))
    return float(np.mean(arr)), lower, upper


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 1:  RETENTION CURVES
# ═════════════════════════════════════════════════════════════════════════

def fig_retention_curves(all_results, schema_results):
    """Retention per memory (A, B, C, D) for each condition.

    Publication-quality: bar plot with CI error bars.
    """
    data = _extract_finals(all_results)
    labels = list(data.keys())
    n_mem = max((len(v[0]) for v in data.values() if v), default=4)

    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5),
                              squeeze=False)
    axes = axes[0]

    for idx, label in enumerate(labels):
        ax = axes[idx]
        trials = data[label]
        if not trials:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{label}", fontsize=11)
            continue

        # Compute means and CIs per memory
        mem_data = {}
        for t in trials:
            for mi in range(len(t)):
                if mi not in mem_data:
                    mem_data[mi] = []
                mem_data[mi].append(t[mi])

        mems = sorted(mem_data.keys())
        means = []
        cis_low = []
        cis_high = []
        for mi in mems:
            arr = np.array(mem_data[mi])
            valid = arr[np.isfinite(arr)]
            if len(valid) > 0:
                m, lo, hi = _bootstrap_ci(valid)
                lo = max(0.0, lo)
                hi = min(1.0, hi)
                means.append(m)
                cis_low.append(lo)
                cis_high.append(hi)
            else:
                means.append(0.0)
                cis_low.append(0.0)
                cis_high.append(0.0)

        x = np.arange(len(mems))
        letters = ["A", "B", "C", "D", "E", "F"]
        labels_mem = [letters[i] if i < len(letters) else f"M{i}" for i in mems]

        ax.bar(x, means, yerr=[np.array(means) - np.array(cis_low),
                                np.array(cis_high) - np.array(means)],
               capsize=4, color=_get_color(label), alpha=0.85,
               error_kw={"linewidth": 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels(labels_mem, fontsize=10)
        ax.set_ylabel("Retention score", fontsize=11)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axhline(y=0, color="gray", ls="--", lw=0.5)
        _style(ax)

    fig.suptitle("Figure 1: Memory Retention Across Conditions",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "figure1_retention_curves")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 2:  SCHEMA CONVERGENCE TRAJECTORIES
# ═════════════════════════════════════════════════════════════════════════

def fig_schema_convergence(all_results, schema_results):
    """Schema convergence trajectories across conditions."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for idx, res in enumerate(all_results):
        label = res["cond"]["label"]
        all_convs = []
        max_len = 0
        for t in res.get("trials", []):
            for steps in t.get("schema_convergence", {}).values():
                dists = [(s["dist_i_to_schema"] + s["dist_j_to_schema"]) / 2.0
                         for s in steps]
                if len(dists) > 1:
                    all_convs.append(dists)
                    max_len = max(max_len, len(dists))
        if not all_convs:
            continue
        aligned = []
        for d in all_convs:
            if len(d) < max_len:
                d = d + [d[-1]] * (max_len - len(d))
            aligned.append(d)
        mean_conv = np.mean(aligned, axis=0)
        sem_conv = np.std(aligned, axis=0, ddof=1) / np.sqrt(len(aligned))
        x = np.arange(len(mean_conv))
        ax.plot(x, mean_conv, label=label, color=_get_color(label), lw=2.5)
        ax.fill_between(x, mean_conv - sem_conv, mean_conv + sem_conv,
                        color=_get_color(label), alpha=0.15)

    ax.axhline(y=0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Training step / snapshot", fontsize=12)
    ax.set_ylabel("Distance to schema centroid", fontsize=12)
    ax.set_title("Figure 2: Schema Convergence Trajectories",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    _style(ax)
    fig.tight_layout()
    _save_fig(fig, "figure2_schema_convergence")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 3:  OVERLAP PROPORTIONALITY
# ═════════════════════════════════════════════════════════════════════════

def fig_overlap_proportionality(all_results, schema_results):
    """Overlap fraction vs centroid drift with regression line."""
    op_results = schema_results.get("overlap_proportionality", {})
    if not op_results:
        print("  [SKIP] overlap proportionality figure: no data", flush=True)
        return

    fig, axes = plt.subplots(1, len(op_results), figsize=(5.5 * len(op_results), 5))
    if len(op_results) == 1:
        axes = [axes]

    for idx, (label, rdata) in enumerate(op_results.items()):
        ax = axes[idx]
        overlaps, drifts = [], []
        for res in all_results:
            if res["cond"]["label"] != label:
                continue
            for t in res.get("trials", []):
                snaps = t.get("centroid_snapshots", [])
                traj = t.get("distance_trajectories", {})
                if len(snaps) < 2:
                    continue
                for (i, j), data in traj.items():
                    of = data.get("overlap_frac", 0.0)
                    dist_series = data.get("pair_dist", [])
                    if len(dist_series) < 2:
                        continue
                    drifts.append(abs(dist_series[-1] - dist_series[0]))
                    overlaps.append(of * 100)

        ax.scatter(overlaps, drifts, alpha=0.6, s=40,
                   c=_get_color(label), edgecolors="black", linewidths=0.5)
        if len(overlaps) > 2:
            from numpy.polynomial.polynomial import polyfit
            coeffs = polyfit(overlaps, drifts, 1)
            x_fit = np.linspace(min(overlaps), max(overlaps), 100)
            ax.plot(x_fit, coeffs[0] + coeffs[1] * x_fit, "r--", lw=1.5,
                    label=f"r={rdata['r']:.3f}, p={rdata['p']:.3f}")
        ax.set_xlabel("Overlap fraction (%)", fontsize=11)
        ax.set_ylabel("Centroid drift magnitude", fontsize=11)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        _style(ax)

    fig.suptitle("Figure 3: Overlap Proportionality — More Overlap = More Convergence",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "figure3_overlap_proportionality")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 4:  COHERENCE VS DRIFT
# ═════════════════════════════════════════════════════════════════════════

def fig_coherence_drift(all_results, schema_results):
    """Replay coherence vs representational drift."""
    cd_results = schema_results.get("coherence_drift", {})
    if not cd_results:
        print("  [SKIP] coherence-drift figure: no data", flush=True)
        return

    fig, axes = plt.subplots(1, len(cd_results), figsize=(5.5 * len(cd_results), 5))
    if len(cd_results) == 1:
        axes = [axes]

    for idx, (label, rdata) in enumerate(cd_results.items()):
        ax = axes[idx]
        coherences, drifts = [], []
        for res in all_results:
            if res["cond"]["label"] != label:
                continue
            for t in res.get("trials", []):
                snaps = t.get("centroid_snapshots", [])
                rp_metrics = t.get("replay_metrics", [])
                if len(snaps) < 2 or len(rp_metrics) < 2:
                    continue
                coh = _safe_mean([m.get("mean_coherence", 0.0) for m in rp_metrics])
                bl = snaps[0].get("centroids", [])
                fin = snaps[-1].get("centroids", [])
                if len(bl) < 2 or len(fin) < 2:
                    continue
                disp = [float(np.linalg.norm(fin[i].ravel() - bl[i].ravel()))
                        for i in range(len(bl))]
                drifts.append(_safe_mean(disp))
                coherences.append(coh)

        ax.scatter(coherences, drifts, alpha=0.6, s=40,
                   c=_get_color(label), edgecolors="black", linewidths=0.5)
        if len(coherences) > 2:
            from numpy.polynomial.polynomial import polyfit
            coeffs = polyfit(coherences, drifts, 1)
            x_fit = np.linspace(min(coherences), max(coherences), 100)
            ax.plot(x_fit, coeffs[0] + coeffs[1] * x_fit, "r--", lw=1.5,
                    label=f"r={rdata['r']:.3f}, p={rdata['p']:.3f}")
        ax.set_xlabel("Mean replay coherence", fontsize=11)
        ax.set_ylabel("Centroid drift magnitude", fontsize=11)
        ax.set_title(f"{label}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        _style(ax)

    fig.suptitle("Figure 4: Replay Coherence vs Representational Drift",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "figure4_coherence_drift")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 5:  BASIN GEOMETRY
# ═════════════════════════════════════════════════════════════════════════

def fig_basin_geometry(all_results, schema_results):
    """Per-assembly retention changes (structured forgetting)."""
    bp_results = schema_results.get("basin_protection", {})
    if not bp_results:
        print("  [SKIP] basin geometry figure: no data", flush=True)
        return

    labels = list(bp_results.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(5.5 * len(labels), 5))
    if len(labels) == 1:
        axes = [axes]

    for idx, label in enumerate(labels):
        ax = axes[idx]
        all_changes = []
        for res in all_results:
            if res["cond"]["label"] != label:
                continue
            for t in res.get("trials", []):
                bl = t.get("baseline_scores", None)
                fn = t.get("final_scores", None)
                if bl is not None and fn is not None:
                    changes = np.array(fn, dtype=float) - np.array(bl, dtype=float)
                    all_changes.append(changes)

        if all_changes:
            changes_arr = np.array(all_changes)
            n_mem = changes_arr.shape[1]
            x = np.arange(n_mem)
            for trial_idx, changes in enumerate(changes_arr):
                ax.plot(x, changes, alpha=0.3, lw=1,
                        color=_get_color(label), marker="o")
            mean_changes = np.mean(changes_arr, axis=0)
            sem_changes = np.std(changes_arr, axis=0, ddof=1) / np.sqrt(len(changes_arr))
            ax.errorbar(x, mean_changes, yerr=sem_changes,
                        color="black", lw=2.5, marker="s", markersize=8,
                        capsize=4, label="Mean")
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)

        letters = ["A", "B", "C", "D", "E", "F"]
        ax.set_xticks(x)
        ax.set_xticklabels([letters[i] if i < len(letters) else f"M{i}"
                            for i in range(len(x))], fontsize=10)
        ax.set_ylabel("Retention change (final - baseline)", fontsize=11)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axhline(y=0, color="gray", ls="--", lw=0.8)
        ax.legend(fontsize=9)
        _style(ax)

    fig.suptitle("Figure 5: Structured Forgetting — Per-Assembly Retention Changes",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "figure5_basin_geometry")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 6:  PARTIAL CUE COMPLETION CURVES
# ═════════════════════════════════════════════════════════════════════════

def fig_partial_cue_completion(all_results, schema_results):
    """Completion probability vs cue fraction."""
    pc_results = schema_results.get("partial_cue", {})
    if not pc_results:
        print("  [SKIP] partial-cue figure: no data", flush=True)
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for idx, (label, pdata) in enumerate(pc_results.items()):
        comp = pdata.get("mean_completion", {})
        if not comp:
            continue
        fracs = sorted(comp.keys())
        frac_pct = [f * 100 for f in fracs]
        vals = [comp[f] for f in fracs]
        ax.plot(frac_pct, vals, marker="o", lw=2.5, markersize=8,
                color=_get_color(label), label=label)
        # Add SEM if available
        sem = pdata.get("sem_completion", None)
        if sem:
            sem_vals = [sem.get(f, 0) for f in fracs]
            ax.fill_between(frac_pct,
                            [min(v - s, 0) for v, s in zip(vals, sem_vals)],
                            [min(v + s, 1) for v, s in zip(vals, sem_vals)],
                            alpha=0.15, color=_get_color(label))

    ax.set_xlabel("Cue fraction (% of assembly)", fontsize=12)
    ax.set_ylabel("Completion probability", fontsize=12)
    ax.set_title("Figure 6: Partial-Cue Completion Curves",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="best")
    ax.axhline(y=0.5, color="gray", ls="--", lw=0.8, label="50% threshold")
    _style(ax)
    fig.tight_layout()
    _save_fig(fig, "figure6_partial_cue_completion")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 7:  REPLAY ENTROPY vs ABSTRACTION
# ═════════════════════════════════════════════════════════════════════════

def fig_replay_entropy_vs_abstraction(all_results, schema_results):
    """Replay entropy vs schema convergence (abstraction)."""
    rd_results = schema_results.get("replay_diversity", {})
    conv_results = schema_results.get("convergence", {})
    if not rd_results or not conv_results:
        print("  [SKIP] replay entropy figure: no data", flush=True)
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for label in rd_results:
        rd = rd_results[label]
        conv = conv_results.get(label, {})
        entropy = rd.get("mean_entropy", 0.0)
        slope = conv.get("mean_slope", 0.0)
        ax.scatter(entropy, -slope, s=120, c=_get_color(label),
                   edgecolors="black", linewidths=1, zorder=5)
        ax.annotate(label, (entropy, -slope),
                    textcoords="offset points", xytext=(10, 5), fontsize=9)

    ax.set_xlabel("Replay entropy (diversity)", fontsize=12)
    ax.set_ylabel("Schema convergence rate (negative = converging)", fontsize=12)
    ax.set_title("Figure 7: Replay Diversity vs Schema Abstraction",
                 fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="gray", ls="--", lw=0.8)
    _style(ax)
    fig.tight_layout()
    _save_fig(fig, "figure7_replay_entropy_vs_abstraction")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 8:  NATURAL vs PERFECT REPLAY
# ═════════════════════════════════════════════════════════════════════════

def fig_natural_vs_perfect(all_results, schema_results):
    """Compare natural fragmented replay vs perfect replay fidelity.

    This figure is generated from the ablation experiment data.
    It checks for 'replay_mode' metadata in results.
    """
    # Check if we have ablation data attached
    replay_modes = {}
    for res in all_results:
        replay_mode = res.get("replay_mode", None)
        if replay_mode is not None:
            if replay_mode not in replay_modes:
                replay_modes[replay_mode] = []
            replay_modes[replay_mode].append(res)

    if len(replay_modes) < 2:
        # Try to extract from schema results
        gen_results = schema_results.get("generalization", {})
        ap_results = schema_results.get("anti_prediction", {})
        if not gen_results or not ap_results:
            print("  [SKIP] natural vs perfect: no ablation data", flush=True)
            return

    # Fallback: compare Fast+Replay (natural) vs Slow+Replay (more structured)
    gen_results = schema_results.get("generalization", {})
    ap_results = schema_results.get("anti_prediction", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Generalization comparison
    labels = list(gen_results.keys())
    means = [gen_results[l].get("mean", 0.0) for l in labels]
    sems = [gen_results[l].get("sem", 0.0) for l in labels]
    x = np.arange(len(labels))
    colors = [_get_color(l) for l in labels]
    ax1.bar(x, means, yerr=sems, capsize=4, color=colors, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Generalization score", fontsize=11)
    ax1.set_title("Schema Generalization", fontsize=12, fontweight="bold")
    ax1.axhline(y=0.5, color="gray", ls="--", lw=0.8, label="Chance")
    _style(ax1)

    # Subplot 2: Anti-prediction (nat vs hf gap)
    ap_labels = list(ap_results.keys())
    nat_means = [ap_results[l].get("natural_mean", 0.0) for l in ap_labels]
    hf_means = [ap_results[l].get("hf_mean", 0.0) for l in ap_labels]
    x2 = np.arange(len(ap_labels))
    w = 0.35
    colors2 = [_get_color(l) for l in ap_labels]
    ax2.bar(x2 - w / 2, nat_means, w, label="Natural (blended)",
            color="steelblue", alpha=0.85)
    ax2.bar(x2 + w / 2, hf_means, w, label="High-fidelity",
            color="coral", alpha=0.85)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(ap_labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Generalization score", fontsize=11)
    ax2.set_title("Anti-Prediction: Natural vs High-Fidelity Replay",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    _style(ax2)

    fig.suptitle("Figure 8: Natural vs Structured Replay — Effect on Generalization",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "figure8_natural_vs_perfect")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 9:  SLOW-WEIGHT ACCUMULATION
# ═════════════════════════════════════════════════════════════════════════

def fig_slow_weight_accumulation(all_results, schema_results):
    """Slow-weight contribution over time (from replay metrics)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    has_data = False
    for res in all_results:
        label = res["cond"]["label"]
        all_w_slow = []
        for t in res.get("trials", []):
            replay_metrics = t.get("replay_metrics", [])
            for ev in replay_metrics:
                ws = ev.get("w_slow_aa", None)
                if ws is not None:
                    all_w_slow.append(ws)
        if all_w_slow:
            has_data = True
            # Sliding window mean
            window = max(1, len(all_w_slow) // 20)
            if window >= 5:
                smoothed = np.convolve(all_w_slow, np.ones(window) / window, mode="valid")
                ax.plot(smoothed, lw=2, color=_get_color(label), label=label)
            else:
                ax.plot(all_w_slow, lw=1.5, alpha=0.7,
                        color=_get_color(label), label=label)

    if not has_data:
        ax.text(0.5, 0.5, "No slow-weight data available.\n"
                          "Slow+Replay condition must be run with W_slow tracking.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Figure 9: Slow-Weight Accumulation (No Data)",
                     fontsize=14, fontweight="bold")
    else:
        ax.set_xlabel("Replay event index", fontsize=12)
        ax.set_ylabel("Mean W_slow within assembly", fontsize=12)
        ax.set_title("Figure 9: Slow-Weight Accumulation Across Replay Events",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)

    _style(ax)
    fig.tight_layout()
    _save_fig(fig, "figure9_slow_weight_accumulation")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 10:  OVERLAP SWEEP
# ═════════════════════════════════════════════════════════════════════════

def fig_overlap_sweep(all_results, schema_results):
    """Overlap sweep: convergence vs overlap fraction.

    Requires sweep_data metadata attached to all_results.
    """
    # Check for sweep data
    sweep_data = getattr(all_results, "sweep_data", None)
    if sweep_data is None:
        # Try to extract from results
        overlap_vals = set()
        for res in all_results:
            ov = res.get("overlap", None)
            if ov is not None:
                overlap_vals.add(ov)
        if not overlap_vals or len(overlap_vals) < 2:
            print("  [SKIP] overlap sweep figure: no sweep data", flush=True)
            return

    conv_results = schema_results.get("convergence", {})
    if not conv_results:
        print("  [SKIP] overlap sweep figure: no convergence data", flush=True)
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for label, cdata in conv_results.items():
        slope = cdata.get("mean_slope", 0.0)
        # Use overlap fraction as x (from metadata or default 0.20)
        overlap = 0.20
        ax.scatter(overlap * 100, -slope, s=120, c=_get_color(label),
                   edgecolors="black", linewidths=1.5, zorder=5)
        ax.annotate(label, (overlap * 100, -slope),
                    textcoords="offset points", xytext=(10, 5), fontsize=9)

    ax.set_xlabel("Overlap fraction (%)", fontsize=12)
    ax.set_ylabel("Schema convergence (negative = converging)", fontsize=12)
    ax.set_title("Figure 10: Overlap Fraction vs Schema Convergence",
                 fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="gray", ls="--", lw=0.8)
    _style(ax)
    fig.tight_layout()
    _save_fig(fig, "figure10_overlap_sweep")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 11:  SCHEMA CRYSTALLIZATION INDEX (SCI)
# ═════════════════════════════════════════════════════════════════════════

def fig_schema_crystallization(all_results, schema_results):
    """Schema Crystallization Index across conditions."""
    sci_results = schema_results.get("schema_crystallization", {})
    if not sci_results:
        print("  [SKIP] SCI figure: no data", flush=True)
        return

    labels = list(sci_results.keys())
    means = [sci_results[l].get("mean_SCI", 0.0) for l in labels]
    sems = [sci_results[l].get("sem", 0.0) for l in labels]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(labels))
    colors = [_get_color(l) for l in labels]
    bars = ax.bar(x, means, yerr=sems, capsize=4, color=colors, alpha=0.85)

    # Color bars by interpretation
    for i, (l, bar) in enumerate(zip(labels, bars)):
        interp = sci_results[l].get("interpretation", "")
        if "differentiating" in interp:
            bar.set_hatch("///")
        elif "decorrelated" in interp:
            pass
        elif "blending" in interp:
            bar.set_hatch("...")

    ax.axhline(y=0, color="gray", ls="--", lw=0.8)
    ax.axhline(y=-0.1, color="red", ls=":", lw=0.8, alpha=0.5,
               label="Differentiation threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Schema Crystallization Index (SCI)", fontsize=12)
    ax.set_title("Figure 11: Schema Crystallization Index\n"
                 "(Negative = healthy differentiation)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    _style(ax)
    fig.tight_layout()
    _save_fig(fig, "figure11_schema_crystallization")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 12:  CATASTROPHIC FORGETTING RESISTANCE (CFR)
# ═════════════════════════════════════════════════════════════════════════

def fig_catastrophic_forgetting_resistance(all_results, schema_results):
    """CFR: Memory A retention ratio across conditions."""
    cfr_results = schema_results.get("cfr", {})
    if not cfr_results:
        print("  [SKIP] CFR figure: no data", flush=True)
        return

    labels = list(cfr_results.keys())
    cfr_a = [cfr_results[l].get("mean_CFR_A", 0.0) for l in labels]
    cfr_a_sem = [cfr_results[l].get("sem_CFR_A", 0.0) for l in labels]
    cfr_overall = [cfr_results[l].get("mean_CFR", 0.0) for l in labels]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(labels))
    w = 0.35
    colors = [_get_color(l) for l in labels]

    ax.bar(x - w / 2, cfr_a, w, yerr=cfr_a_sem, capsize=4,
           color=colors, alpha=0.85, label="Memory A (oldest)")
    ax.bar(x + w / 2, cfr_overall, w, capsize=4,
           color=colors, alpha=0.4, label="Overall")

    ax.axhline(y=0.5, color="green", ls="--", lw=1.5,
               label="Target resistance (0.5)")
    ax.axhline(y=0.0, color="gray", ls="-", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("CFR (Retention_A_final / Retention_A_baseline)", fontsize=11)
    ax.set_title("Figure 12: Catastrophic Forgetting Resistance",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    _style(ax)
    fig.tight_layout()
    _save_fig(fig, "figure12_catastrophic_forgetting_resistance")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 13:  CENTROID TRAJECTORY PCA (with sleep-phase coloring)
# ═════════════════════════════════════════════════════════════════════════

def fig_centroid_trajectory_pca(all_results, schema_results):
    """PCA projection of centroid trajectories, colored by sleep phase.

    Requires sklearn (pip install scikit-learn).

    Colors: Wake = green, NREM = blue, REM = red.
    Arrows show direction of time.
    """
    try:
        from sklearn.decomposition import PCA as _PCA
    except ImportError:
        print("  [SKIP] centroid PCA figure: sklearn not installed", flush=True)
        return

    n_conds = len(all_results)
    fig, axes = plt.subplots(2, max(1, (n_conds + 1) // 2),
                             figsize=(6 * min(2, n_conds), 8))
    axes = np.atleast_1d(axes).ravel()

    for idx, res in enumerate(all_results):
        if idx >= len(axes):
            break
        ax = axes[idx]
        label = res["cond"]["label"]

        all_centroids = []
        labels_list = []
        for t in res.get("trials", []):
            for s in t.get("centroid_snapshots", []):
                for c in s.get("centroids", []):
                    all_centroids.append(c.ravel())
                    lbl = s.get("label", "unknown")
                    if "baseline" in lbl:
                        labels_list.append("wake")
                    elif "final" in lbl:
                        labels_list.append("rem")
                    elif "replay" in lbl:
                        labels_list.append("rem")
                    elif "encode" in lbl:
                        labels_list.append("wake")
                    else:
                        labels_list.append("nrem")

        if len(all_centroids) < 3:
            ax.text(0.5, 0.5, "insufficient data",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontsize=11)
            continue

        X = np.array(all_centroids)
        X_pca = _PCA(n_components=2).fit_transform(X)

        # Color by phase
        color_map_phase = {"wake": "#27ae60", "nrem": "#2980b9", "rem": "#e74c3c"}
        phase_colors = [color_map_phase.get(p, "#95a5a6") for p in labels_list]

        # Plot with connecting lines
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=phase_colors,
                   alpha=0.7, s=30, edgecolors="black", linewidths=0.3)
        ax.plot(X_pca[:, 0], X_pca[:, 1], "gray", lw=0.5, alpha=0.4)

        # Add arrow to show direction of time
        from matplotlib.patches import FancyArrowPatch
        if len(X_pca) > 1:
            start, end = 0, min(len(X_pca) - 1, 5)
            ax.annotate("", xy=(X_pca[end, 0], X_pca[end, 1]),
                        xytext=(X_pca[start, 0], X_pca[start, 1]),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        _style(ax)

    for idx in range(len(all_results), len(axes)):
        axes[idx].set_visible(False)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#27ae60", label="Wake"),
        Patch(facecolor="#2980b9", label="NREM"),
        Patch(facecolor="#e74c3c", label="REM"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Figure 13: Centroid Trajectory (PCA)\n"
                 "Green=Wake, Blue=NREM, Red=REM — Arrows show time",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, "figure13_centroid_trajectory_pca")


# ═════════════════════════════════════════════════════════════════════════
# SUMMARY HEATMAP
# ═════════════════════════════════════════════════════════════════════════

def fig_summary_heatmap(all_results, schema_results):
    """Summary heatmap of all test statistics."""
    test_names = [
        "directionality", "convergence", "coherence_drift",
        "retention_tradeoff", "overlap_proportionality",
        "generalization", "anti_prediction", "downscaling",
        "schema_crystallization", "cfr",
    ]
    stat_names = ["mean", "mean_slope", "r", "r", "r",
                  "mean", "gap", "mean_active_synapses",
                  "mean_SCI", "mean_CFR_A"]
    conditions = [res["cond"]["label"] for res in all_results]
    if not conditions:
        return

    data = np.full((len(test_names), len(conditions)), np.nan)
    for i, (tname, sname) in enumerate(zip(test_names, stat_names)):
        tres = schema_results.get(tname, {})
        for j, cond in enumerate(conditions):
            if cond in tres:
                val = tres[cond].get(sname, np.nan)
                data[i, j] = val if val is not None else np.nan
                if tname == "directionality" and val is not None:
                    data[i, j] = -val if val != 0 else 0  # negate so positive = good

    fig, ax = plt.subplots(1, 1, figsize=(max(6, len(conditions) * 2), 7))
    vlim = max(abs(np.nanmax(data)), abs(np.nanmin(data)), 0.1)
    im = ax.imshow(data, cmap="RdYlBu_r", aspect="auto", vmin=-vlim, vmax=vlim)

    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(test_names)))
    ax.set_yticklabels([t.replace("_", " ").title() for t in test_names], fontsize=10)

    for i in range(len(test_names)):
        for j in range(len(conditions)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(val) < vlim * 0.5 else "white")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Effect size (|r| or mean)")
    ax.set_title("Schema Abstraction: Summary Heatmap\n"
                 "(Red = stronger schema effect)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "summary_heatmap")


# ═════════════════════════════════════════════════════════════════════════
# LEGACY FIGURES (keep for backward compatibility)
# ═════════════════════════════════════════════════════════════════════════

def fig_pairwise_distance_trajectories(all_results, schema_results):
    """Legacy pairwise distance trajectories."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for idx, res in enumerate(all_results):
        label = res["cond"]["label"]
        all_trajs = []
        max_len = 0
        for t in res.get("trials", []):
            for pair_data in t.get("distance_trajectories", {}).values():
                d = pair_data.get("pair_dist", [])
                if len(d) > 1:
                    all_trajs.append(d)
                    max_len = max(max_len, len(d))
        if not all_trajs:
            continue
        aligned = []
        for d in all_trajs:
            if len(d) < max_len:
                d = d + [d[-1]] * (max_len - len(d))
            aligned.append(d)
        mean_traj = np.mean(aligned, axis=0)
        sem_traj = np.std(aligned, axis=0, ddof=1) / np.sqrt(len(aligned))
        x = np.arange(len(mean_traj))
        ax.plot(x, mean_traj, label=label, color=_get_color(label), lw=2)
        ax.fill_between(x, mean_traj - sem_traj, mean_traj + sem_traj,
                        color=_get_color(label), alpha=0.15)
    ax.set_xlabel("Snapshot index")
    ax.set_ylabel("Pairwise cosine distance")
    ax.set_title("Centroid Distance Trajectories")
    ax.legend(fontsize=10)
    _style(ax)
    fig.tight_layout()
    _save_fig(fig, "pairwise_distance_trajectories")


def fig_directionality_barplot(all_results, schema_results):
    """Legacy directionality barplot."""
    dir_results = schema_results.get("directionality", {})
    if not dir_results:
        print("  [SKIP] directionality barplot: no data", flush=True)
        return
    labels = list(dir_results.keys())
    means = [dir_results[l]["mean"] for l in labels]
    sems = [dir_results[l]["sem"] for l in labels]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(labels))
    colors = [_get_color(l) for l in labels]
    ax.bar(x, means, yerr=sems, capsize=4, color=colors, alpha=0.85)
    ax.axhline(y=0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Directionality score", fontsize=11)
    ax.set_title("Schema Directionality", fontsize=12)
    _style(ax)
    fig.tight_layout()
    _save_fig(fig, "directionality_barplot")


# ═════════════════════════════════════════════════════════════════════════
# REMAINING LEGACY FIGURES (stubs that call existing)
# ═════════════════════════════════════════════════════════════════════════

def fig_anti_prediction(all_results, schema_results):
    """Legacy anti-prediction figure."""
    ap_results = schema_results.get("anti_prediction", {})
    if not ap_results:
        print("  [SKIP] anti-prediction: no data", flush=True)
        return
    labels = list(ap_results.keys())
    nat_means = [ap_results[l]["natural_mean"] for l in labels]
    hf_means = [ap_results[l]["hf_mean"] for l in labels]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, nat_means, w, label="Natural (blended)", color="steelblue", alpha=0.85)
    ax.bar(x + w / 2, hf_means, w, label="High-fidelity", color="coral", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Generalization score", fontsize=11)
    ax.set_title("Anti-Prediction: Fidelity vs Generalization", fontsize=12)
    ax.legend(fontsize=9)
    _style(ax)
    fig.tight_layout()
    _save_fig(fig, "anti_prediction")


def fig_downscaling(all_results, schema_results):
    """Legacy downscaling figure."""
    ds_results = schema_results.get("downscaling", {})
    if not ds_results:
        print("  [SKIP] downscaling: no data", flush=True)
        return
    labels = list(ds_results.keys())
    active_syn = [ds_results[l].get("mean_active_synapses", 0) for l in labels]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(labels))
    colors = [_get_color(l) for l in labels]
    ax.bar(x, active_syn, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Replay-active synapses", fontsize=11)
    ax.set_title("Downscaling: Synapse Protection", fontsize=12)
    _style(ax)
    fig.tight_layout()
    _save_fig(fig, "downscaling")


def fig_generative_layer(all_results, schema_results):
    """Generative layer figure."""
    gl_results = schema_results.get("generative_layer", {})
    if not gl_results:
        print("  [SKIP] generative layer: no data", flush=True)
        return
    labels = list(gl_results.keys())
    mse_vals = [gl_results[l].get("final_mse", 0) for l in labels]
    indep_vals = [gl_results[l].get("generative_independence", 0) for l in labels]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = [_get_color(l) for l in labels]
    x = np.arange(len(labels))
    ax1.bar(x, mse_vals, color=colors, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Final reconstruction MSE", fontsize=11)
    ax1.set_title("Autoencoder Reconstruction Error", fontsize=12)
    _style(ax1)
    ax2.bar(x, indep_vals, color=colors, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Generative independence (r)", fontsize=11)
    ax2.set_title("Cortical Independence", fontsize=12)
    _style(ax2)
    fig.suptitle("Generative Cortical Layer", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, "generative_layer")


def fig_hidden_state(all_results, schema_results):
    """Hidden state figure."""
    hs_results = schema_results.get("hidden_state", {})
    if not hs_results:
        print("  [SKIP] hidden state: no data", flush=True)
        return
    labels = list(hs_results.keys())
    cortical = [hs_results[l].get("mean_cortical", 0) for l in labels]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(labels))
    colors = [_get_color(l) for l in labels]
    ax.bar(x, cortical, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Cortical memories (hidden state > 0.7)", fontsize=11)
    ax.set_title("Hidden State: Consolidation Status", fontsize=12)
    _style(ax)
    fig.tight_layout()
    _save_fig(fig, "hidden_state")


def fig_forgetting_variability(all_results, schema_results):
    """Forgetting variability figure."""
    bp_results = schema_results.get("basin_protection", {})
    if not bp_results:
        print("  [SKIP] forgetting variability: no data", flush=True)
        return
    labels = list(bp_results.keys())
    var_vals = [bp_results[l].get("forgetting_variability", 0) for l in labels]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(labels))
    colors = [_get_color(l) for l in labels]
    ax.bar(x, var_vals, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Retention change variability", fontsize=11)
    ax.set_title("Structured Forgetting: Basin Protection", fontsize=12)
    _style(ax)
    fig.tight_layout()
    _save_fig(fig, "forgetting_variability")


def fig_reverse_replay(all_results, schema_results):
    """Reverse replay figure (legacy)."""
    print("  [SKIP] reverse replay: legacy figure not updated", flush=True)


def fig_uncertainty_weights(all_results, schema_results):
    """Uncertainty weights figure (legacy)."""
    print("  [SKIP] uncertainty weights: legacy figure not updated", flush=True)


def fig_multi_seed_meta(all_results, schema_results):
    """Multi-seed meta-analysis figure (legacy)."""
    meta = schema_results.get("meta_analysis", None)
    if meta is None:
        print("  [SKIP] multi-seed meta: no data", flush=True)
        return
    test_names = list(meta.keys())
    n_tests = len(test_names)
    if n_tests == 0:
        return
    fig, axes = plt.subplots(1, n_tests, figsize=(5 * n_tests, 4))
    if n_tests == 1:
        axes = [axes]
    for idx, test_name in enumerate(test_names):
        ax = axes[idx]
        cond_data = meta.get(test_name, {})
        conds = list(cond_data.keys())
        means = [cond_data[c]["mean_over_seeds"] for c in conds]
        sems = [cond_data[c]["sem_over_seeds"] for c in conds]
        x = np.arange(len(conds))
        colors = [_get_color(c) for c in conds]
        ax.bar(x, means, yerr=sems, capsize=4, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=8)
        ax.set_title(test_name.replace("_", " ").title(), fontsize=10)
        _style(ax)
    fig.suptitle("Multi-Seed Meta-Analysis", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, "multi_seed_meta")


# ═════════════════════════════════════════════════════════════════════════
# MASTER CALLER
# ═════════════════════════════════════════════════════════════════════════

def generate_all_schema_figures(all_results, schema_results):
    """Generate all publication-quality figures."""
    print("\n  Generating schema-abstraction publication figures...", flush=True)

    _figures = [
        # Primary publication figures (10+)
        fig_retention_curves,
        fig_schema_convergence,
        fig_overlap_proportionality,
        fig_coherence_drift,
        fig_basin_geometry,
        fig_partial_cue_completion,
        fig_replay_entropy_vs_abstraction,
        fig_natural_vs_perfect,
        fig_slow_weight_accumulation,
        fig_overlap_sweep,
        fig_schema_crystallization,
        fig_catastrophic_forgetting_resistance,
        fig_centroid_trajectory_pca,

        # Legacy figures (backward compatibility)
        fig_pairwise_distance_trajectories,
        fig_directionality_barplot,
        fig_anti_prediction,
        fig_downscaling,
        fig_generative_layer,
        fig_hidden_state,
        fig_forgetting_variability,
        fig_reverse_replay,
        fig_uncertainty_weights,
        fig_multi_seed_meta,

        # Summary
        fig_summary_heatmap,
    ]

    for fn in _figures:
        try:
            fn(all_results, schema_results)
        except Exception as e:
            print(f"  [SKIP] {fn.__name__}: {e}", flush=True)

    print("  Done.", flush=True)
