"""Score all PNG figures and select the top 15 for the paper."""
import json, shutil, time
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/Admin/brain-organoid-rl")
OUTPUT_DIR = PROJECT_ROOT / "important_paper_materials_and_results"

with open(OUTPUT_DIR / "00_full_file_inventory.json", "r") as f:
    inventory = json.load(f)

HIGH_PRIORITY_NAMES = [
    "serial_position_flagship",
    "serial_position",
    "e2_task105_30seeds",
    "e2_causal",
    "major1_scheduling",
    "major1_decoupling",
    "major1_equalized",
    "mod2_consolidation_law",
    "mod2_law",
    "mod3_learned_schema",
    "mod3_schema",
    "mod4_istdp",
    "mod4_inhibitory",
    "major3_wslow",
    "wslow_panel",
    "fig1_early_replay",
    "task10_fig1",
    "Figure1_MechanisticArchitecture",
    "Figure2_ExperimentalValidation",
    "e3_learned_schema",
    "m4_null_model",
    "q3_seed_scatter",
    "fig_Q3",
    "q6_parameter",
    "mod5_bio_param",
    "q5_dose_response",
    "mod1_scaling",
    "task2",
    "fig4_full_vs_noreplay",
    "fig3_block",
    "fig2_weight",
    "task7",
]

MEDIUM_PRIORITY_NAMES = [
    "retention", "replay", "consolidation", "memory",
    "attenuation", "bootstrap", "comparison", "primacy",
    "recency", "position", "cascade", "schema", "core",
    "suppression", "boost", "interaction", "dose",
    "ablation", "sensitivity", "w_slow", "heatmap",
]

LOW_PRIORITY_NAMES = [
    "debug", "test_", "temp", "old", "backup", "draft",
    "heartbeat", "checkpoint", "diagnostic_",
]


def score_figure(entry):
    name = entry["name"].lower().replace(".png", "")
    score = 0
    reasons = []

    for hp in HIGH_PRIORITY_NAMES:
        if hp.lower() in name:
            score += 8
            reasons.append(f"HIGH: {hp}")
            break

    if score < 8:
        for mp in MEDIUM_PRIORITY_NAMES:
            if mp in name:
                score += 3
                reasons.append(f"MED: {mp}")
                break

    for lp in LOW_PRIORITY_NAMES:
        if lp in name:
            score -= 5
            reasons.append(f"LOW: {lp}")

    if entry["size_kb"] > 500:
        score += 3
        reasons.append("large file")
    elif entry["size_kb"] > 200:
        score += 2
    elif entry["size_kb"] > 50:
        score += 1
    elif entry["size_kb"] < 10:
        score -= 3
        reasons.append("very small")

    days_old = (time.time() - entry["modified"]) / 86400
    if days_old < 3:
        score += 2
        reasons.append("recent")
    elif days_old < 7:
        score += 1

    path_lower = entry["path"].lower()
    if any(folder in path_lower for folder in
           ["serial_position", "e2_results", "major1", "mod_results",
            "mod2", "mod3", "mod4", "task11", "ablation_results",
            "m4_results", "m5_results", "major3", "major5"]):
        score += 2
        reasons.append("key results folder")

    # Bonus for being in figures/paper
    if "figures\\paper" in path_lower or "figures/paper" in path_lower:
        score += 4
        reasons.append("in figures/paper")

    return score, reasons


all_figures_scored = []
for fig_entry in inventory["figures_png"]:
    score, reasons = score_figure(fig_entry)
    all_figures_scored.append({
        **fig_entry,
        "score": score,
        "reasons": reasons
    })

all_figures_scored.sort(key=lambda x: -x["score"])

print(f"=== FIGURE SCORING RESULTS ===")
print(f"Total PNG figures: {len(all_figures_scored)}")
print(f"\nTop 30 candidates:")
for i, fig in enumerate(all_figures_scored[:30]):
    print(f"  {i+1:2d}. Score={fig['score']:3d} | {fig['name'][:60]:60s} | {fig['size_kb']:.0f}KB")
    if fig["reasons"]:
        print(f"       Reasons: {', '.join(fig['reasons'][:3])}")

selected_15 = all_figures_scored[:15]
rejected = all_figures_scored[15:]

print(f"\n=== SELECTED 15 FIGURES ===")
for i, fig in enumerate(selected_15):
    print(f"{i+1:2d}. {fig['name']} (score={fig['score']}, {fig['size_kb']:.0f}KB)")

print("\nCopying selected figures...")
for i, fig in enumerate(selected_15):
    src = Path(fig["path"])
    dst = OUTPUT_DIR / "figures_selected" / f"{i+1:02d}_{fig['name']}"
    shutil.copy2(src, dst)
    print(f"  Copied {i+1}: {fig['name']}")

# Copy ALL figures sorted by relevance
all_figs_dir = OUTPUT_DIR / "figures_all_sorted_by_relevance"
all_figs_dir.mkdir(exist_ok=True)
copied = 0
for i, fig in enumerate(all_figures_scored):
    src = Path(fig["path"])
    try:
        safe_name = fig["name"].replace(" ", "_")
        dst = all_figs_dir / f"{i+1:03d}_score{fig['score']}_{safe_name}"
        shutil.copy2(src, dst)
        copied += 1
    except Exception as e:
        pass

print(f"\nAll {copied} figures copied to figures_all_sorted_by_relevance/")

# Save the scoring data
with open(OUTPUT_DIR / "figures_selected" / "00_figure_scoring.json", "w") as f:
    scoring_data = [{
        "rank": i+1,
        "name": fig["name"],
        "score": fig["score"],
        "size_kb": fig["size_kb"],
        "relative_path": fig["relative"],
        "reasons": fig["reasons"],
        "selected": i < 15
    } for i, fig in enumerate(all_figures_scored)]
    json.dump(scoring_data, f, indent=2)

# Write rejection reasons for figures not selected
with open(OUTPUT_DIR / "figures_rejected_with_reason" / "00_rejection_reasons.txt", "w") as f:
    f.write("REJECTED FIGURES WITH REASONS\n")
    f.write("=" * 60 + "\n\n")
    for i, fig in enumerate(rejected[:50]):
        f.write(f"Rank {i+16}: {fig['name']} (score={fig['score']}, {fig['size_kb']:.0f}KB)\n")
        f.write(f"  Path: {fig['relative']}\n")
        f.write(f"  Reasons: {', '.join(fig['reasons']) if fig['reasons'] else 'no specific match'}\n\n")

print("Rejection reasons saved.")
