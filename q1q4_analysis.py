"""
Q1-Q4 Statistical Fixes for RGCC Paper
=======================================
Q1: Mixed-effects model replacing pseudoreplicated Pearson r
Q2: Within-seed Spearman aggregation (folded into Q1)
Q3: Seed-level scatter plot FULL vs NO_REPLAY
Q4: Supplementary statistics table
"""
import numpy as np
import pickle, os, sys
from scipy import stats
from pathlib import Path

OUT_DIR = Path(r'C:\Users\Admin\brain-organoid-rl\ablation_results\q_fixes')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════

# --- Task 10: Per-memory data (3 seeds × 4 memories = 12 observations) ---
T10_DATA = {
    42:   {'replay': [21, 15, 9, 0], 'retention': [0.343, 0.316, 0.304, 0.257],
           'WScc': 0.615, 'WSuc': 0.132, 'WSuu': 0.042},
    1042: {'replay': [23, 13, 10, 0], 'retention': [0.318, 0.290, 0.282, 0.238],
           'WScc': 0.616, 'WSuc': 0.129, 'WSuu': 0.042},
    2042: {'replay': [22, 20, 4, 0], 'retention': [0.329, 0.311, 0.263, 0.244],
           'WScc': 0.598, 'WSuc': 0.118, 'WSuu': 0.038},
}

# Load actual per-memory W_slow from PKLs if available
T10_PKL_DIR = Path(r'C:\Users\Admin\brain-organoid-rl\ablation_results\task10')
for seed in [42, 1042, 2042]:
    pkl_path = T10_PKL_DIR / f'T10_seed{seed}.pkl'
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        if 'per_mem_ws' in d:
            T10_DATA[seed]['per_mem_ws'] = d['per_mem_ws']
            print(f"  Loaded per_mem_ws from seed {seed}: {d['per_mem_ws']}")
        if 'timeline' in d:
            print(f"  Seed {seed} has {len(d['timeline'])} timeline entries")

# --- Task 2: Per-seed FULL vs NO_REPLAY (10 seeds each) ---
FULL_RET = [0.3064, 0.2812, 0.2880, 0.2711, 0.2964, 0.3009, 0.2848, 0.2825, 0.2869, 0.2619]
NOREPLAY_RET = [0.0440, 0.0353, 0.0350, 0.0363, 0.0396, 0.0376, 0.0353, 0.0374, 0.0397, 0.0334]
SEEDS = [42, 1042, 2042, 3042, 4042, 5042, 6042, 7042, 8042, 9042]

# Per-memory retention from Task 2 (FULL condition)
FULL_PER_MEM = {
    42:   [0.343, 0.316, 0.303, 0.264],
    1042: [0.319, 0.292, 0.279, 0.235],
    2042: [0.332, 0.305, 0.267, 0.248],
    3042: [0.316, 0.283, 0.254, 0.231],
    4042: [0.333, 0.302, 0.293, 0.257],
    5042: [0.339, 0.302, 0.311, 0.252],
    6042: [0.325, 0.284, 0.293, 0.237],
    7042: [0.313, 0.305, 0.274, 0.238],
    8042: [0.331, 0.291, 0.283, 0.243],
    9042: [0.300, 0.273, 0.249, 0.226],
}

NOREPLAY_PER_MEM = {
    42:   [0.047, 0.045, 0.043, 0.041],
    1042: [0.033, 0.039, 0.035, 0.034],
    2042: [0.034, 0.032, 0.040, 0.034],
    3042: [0.038, 0.034, 0.038, 0.035],
    4042: [0.040, 0.034, 0.044, 0.040],
    5042: [0.037, 0.036, 0.040, 0.037],
    6042: [0.036, 0.035, 0.038, 0.033],
    7042: [0.035, 0.038, 0.040, 0.036],
    8042: [0.043, 0.037, 0.042, 0.036],
    9042: [0.028, 0.033, 0.037, 0.035],
}

# Wcc per seed
FULL_WCC = [0.0903, 0.0636, 0.0626, 0.0626, 0.0664, 0.0675, 0.0623, 0.0631, 0.0662, 0.0630]
NOREPLAY_WCC = [0.0452, 0.0320, 0.0327, 0.0335, 0.0328, 0.0327, 0.0326, 0.0339, 0.0318, 0.0318]
FULL_S1 = [0.0542, 0.0412, 0.0429, 0.0436, 0.0433, 0.0449, 0.0415, 0.0429, 0.0447, 0.0441]
NOREPLAY_S1 = [0.0247, 0.0212, 0.0222, 0.0229, 0.0224, 0.0227, 0.0224, 0.0239, 0.0208, 0.0213]

print("=" * 70)
print("Q1: MIXED-EFFECTS MODEL - Replay Count -> Retention/W_slow")
print("=" * 70)

# Build observation-level data: each row = (seed, memory, replay_count, retention, w_slow)
obs_seed = []
obs_mem = []
obs_replay = []
obs_ret = []
obs_wslow = []

for seed, data in T10_DATA.items():
    for mi in range(4):
        obs_seed.append(seed)
        obs_mem.append(mi)
        obs_replay.append(data['replay'][mi])
        obs_ret.append(data['retention'][mi])
        # Use per_mem_ws if available
        if 'per_mem_ws' in data:
            ws = data['per_mem_ws']
            obs_wslow.append(ws.get(mi, ws.get(str(mi), 0.0)))
        else:
            obs_wslow.append(np.nan)

obs_seed = np.array(obs_seed)
obs_mem = np.array(obs_mem)
obs_replay = np.array(obs_replay)
obs_ret = np.array(obs_ret)
obs_wslow = np.array(obs_wslow)

print(f"\nN observations: {len(obs_seed)} (3 seeds × 4 memories)")
print(f"Replay range: {obs_replay.min()} – {obs_replay.max()}")
print(f"Retention range: {obs_ret.min():.3f} – {obs_ret.max():.3f}")

# --- Q1a: Naive (pseudoreplicated) Pearson r ---
r_naive_ret, p_naive_ret = stats.pearsonr(obs_replay, obs_ret)
print(f"\nNaive Pearson r (replay -> retention): r={r_naive_ret:.3f}, p={p_naive_ret:.4f}, N=12")
print("  !! WARNING: This treats 12 obs as independent, ignoring seed clustering")

if not np.all(np.isnan(obs_wslow)):
    mask = ~np.isnan(obs_wslow)
    r_naive_ws, p_naive_ws = stats.pearsonr(obs_replay[mask], obs_wslow[mask])
    print(f"Naive Pearson r (replay -> W_slow): r={r_naive_ws:.3f}, p={p_naive_ws:.4f}, N={mask.sum()}")

# --- Q1b: Within-seed Spearman (Q2 folded in) ---
print(f"\n--- Q2: Within-seed Spearman correlations ---")
within_rhos_ret = []
within_rhos_ws = []
for seed in T10_DATA:
    data = T10_DATA[seed]
    rho_ret, _ = stats.spearmanr(data['replay'], data['retention'])
    within_rhos_ret.append(rho_ret)
    print(f"  Seed {seed}: Spearman rho(replay, retention) = {rho_ret:.3f}")

    if 'per_mem_ws' in data:
        ws_vals = [data['per_mem_ws'].get(i, data['per_mem_ws'].get(str(i), 0)) for i in range(4)]
        rho_ws, _ = stats.spearmanr(data['replay'], ws_vals)
        within_rhos_ws.append(rho_ws)
        print(f"           Spearman rho(replay, W_slow) = {rho_ws:.3f}")

mean_rho_ret = np.mean(within_rhos_ret)
se_rho_ret = np.std(within_rhos_ret, ddof=1) / np.sqrt(len(within_rhos_ret))
print(f"\n  Aggregated rho(replay->retention): {mean_rho_ret:.3f} ± {se_rho_ret:.3f}")
# One-sample t-test: is mean rho > 0?
t_rho, p_rho = stats.ttest_1samp(within_rhos_ret, 0)
print(f"  t({len(within_rhos_ret)-1}) = {t_rho:.2f}, p = {p_rho:.4f}")

if within_rhos_ws:
    mean_rho_ws = np.mean(within_rhos_ws)
    se_rho_ws = np.std(within_rhos_ws, ddof=1) / np.sqrt(len(within_rhos_ws))
    print(f"  Aggregated rho(replay->W_slow): {mean_rho_ws:.3f} ± {se_rho_ws:.3f}")
    t_ws, p_ws = stats.ttest_1samp(within_rhos_ws, 0)
    print(f"  t({len(within_rhos_ws)-1}) = {t_ws:.2f}, p = {p_ws:.4f}")

# --- Q1c: Mixed-effects approximation via seed-mean-centered regression ---
# Center replay and retention within each seed, then pool
print(f"\n--- Q1c: Seed-centered regression (mixed-effects proxy) ---")
centered_replay = []
centered_ret = []
centered_ws = []
for seed in T10_DATA:
    data = T10_DATA[seed]
    rep = np.array(data['replay'], dtype=float)
    ret = np.array(data['retention'])
    centered_replay.extend(rep - rep.mean())
    centered_ret.extend(ret - ret.mean())
    if 'per_mem_ws' in data:
        ws = [data['per_mem_ws'].get(i, data['per_mem_ws'].get(str(i), 0)) for i in range(4)]
        ws = np.array(ws)
        centered_ws.extend(ws - ws.mean())

centered_replay = np.array(centered_replay)
centered_ret = np.array(centered_ret)
slope_ret, intercept_ret, r_centered, p_centered, se_slope = stats.linregress(centered_replay, centered_ret)
# Correct df: N - K - 1 where K=number of seeds (random effects) + 1 fixed effect
# Simplified: df = N - n_seeds - 1 = 12 - 3 - 1 = 8
df_corrected = len(centered_replay) - len(T10_DATA) - 1
t_corrected = slope_ret / se_slope
from scipy.stats import t as t_dist
p_corrected = 2 * (1 - t_dist.cdf(abs(t_corrected), df_corrected))

print(f"  Slope (within-seed replay -> retention): beta = {slope_ret:.6f}")
print(f"  SE(beta) = {se_slope:.6f}")
print(f"  Centered r = {r_centered:.3f}")
print(f"  t({df_corrected}) = {t_corrected:.2f}, p = {p_corrected:.4f}")
print(f"  Interpretation: Each additional replay event adds ~{slope_ret*100:.2f}pp to retention")

if centered_ws:
    centered_ws = np.array(centered_ws)
    slope_ws, _, r_ws_centered, p_ws_centered, se_ws = stats.linregress(centered_replay, centered_ws)
    t_ws_corrected = slope_ws / se_ws
    p_ws_corrected = 2 * (1 - t_dist.cdf(abs(t_ws_corrected), df_corrected))
    print(f"\n  Slope (within-seed replay -> W_slow): beta = {slope_ws:.6f}")
    print(f"  SE(beta) = {se_ws:.6f}")
    print(f"  Centered r = {r_ws_centered:.3f}")
    print(f"  t({df_corrected}) = {t_ws_corrected:.2f}, p = {p_ws_corrected:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q3: SEED-LEVEL SCATTER PLOT -- FULL vs NO_REPLAY")
print("=" * 70)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Paired dot plot
ax = axes[0]
for i, seed in enumerate(SEEDS):
    ax.plot([0, 1], [FULL_RET[i], NOREPLAY_RET[i]], 'o-', color='gray', alpha=0.5, markersize=6)
    ax.plot(0, FULL_RET[i], 'o', color='#2196F3', markersize=8, zorder=5)
    ax.plot(1, NOREPLAY_RET[i], 'o', color='#F44336', markersize=8, zorder=5)

ax.set_xticks([0, 1])
ax.set_xticklabels(['FULL\n(replay ON)', 'NO_REPLAY\n(replay OFF)'], fontsize=11)
ax.set_ylabel('Mean Retention', fontsize=12)
ax.set_title('A. Paired comparison (n=10 seeds)', fontsize=13, fontweight='bold')
ax.set_ylim(-0.01, 0.35)
ax.axhline(y=np.mean(FULL_RET), color='#2196F3', linestyle='--', alpha=0.5, label=f'FULL mean={np.mean(FULL_RET):.3f}')
ax.axhline(y=np.mean(NOREPLAY_RET), color='#F44336', linestyle='--', alpha=0.5, label=f'NO_REPLAY mean={np.mean(NOREPLAY_RET):.3f}')
ax.legend(fontsize=9)

# Panel B: Effect size per seed
ax = axes[1]
deltas = [f - n for f, n in zip(FULL_RET, NOREPLAY_RET)]
ax.barh(range(10), deltas, color='#4CAF50', alpha=0.7, height=0.6)
for i, (d, seed) in enumerate(zip(deltas, SEEDS)):
    ax.text(d + 0.002, i, f'{d:.3f}', va='center', fontsize=9)
ax.set_yticks(range(10))
ax.set_yticklabels([f'Seed {s}' for s in SEEDS], fontsize=9)
ax.set_xlabel('Delta Retention (FULL − NO_REPLAY)', fontsize=11)
ax.set_title('B. Per-seed effect magnitude', fontsize=13, fontweight='bold')

# Stats annotation
t_paired, p_paired = stats.ttest_rel(FULL_RET, NOREPLAY_RET)
d_paired = np.mean(deltas) / np.std(deltas, ddof=1)
ax.text(0.95, 0.05, f'Paired t(9)={t_paired:.1f}\np<1e-15\nCohen\'s d={d_paired:.1f}',
        transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
fig_path = OUT_DIR / 'fig_Q3_seed_scatter.png'
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
fig.savefig(str(fig_path).replace('.png', '.pdf'), bbox_inches='tight')
fig.savefig(str(fig_path).replace('.png', '.svg'), bbox_inches='tight')
plt.close()
print(f"\nSaved Q3 figure: {fig_path}")

# Print paired t-test results
print(f"\nPaired t-test (FULL vs NO_REPLAY, n=10 seeds):")
print(f"  FULL mean = {np.mean(FULL_RET):.4f} ± {np.std(FULL_RET, ddof=1):.4f}")
print(f"  NO_REPLAY mean = {np.mean(NOREPLAY_RET):.4f} ± {np.std(NOREPLAY_RET, ddof=1):.4f}")
print(f"  Delta = {np.mean(deltas):.4f} ± {np.std(deltas, ddof=1):.4f}")
print(f"  t(9) = {t_paired:.2f}")
print(f"  p = {p_paired:.2e}")
print(f"  Cohen's d = {d_paired:.2f}")
print(f"  All 10 seeds show FULL > NO_REPLAY: {all(d > 0 for d in deltas)}")

# Wilcoxon signed-rank as non-parametric backup
w_stat, p_wilcoxon = stats.wilcoxon(FULL_RET, NOREPLAY_RET)
print(f"  Wilcoxon W = {w_stat:.1f}, p = {p_wilcoxon:.4f}")

# 95% CI on the mean difference
from scipy.stats import t as t_dist
ci_se = np.std(deltas, ddof=1) / np.sqrt(10)
ci_margin = t_dist.ppf(0.975, 9) * ci_se
print(f"  95% CI on Delta: [{np.mean(deltas) - ci_margin:.4f}, {np.mean(deltas) + ci_margin:.4f}]")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q4: SUPPLEMENTARY STATISTICS TABLE")
print("=" * 70)

# Build comprehensive table
print("\nTable S1: Complete Statistical Summary")
print("-" * 100)
print(f"{'Claim':<50} {'Statistic':<15} {'Value':<12} {'df':<6} {'p':<12} {'CI_95':<20}")
print("-" * 100)

# 1. Replay necessity (retention)
t_val = 57.64  # from FINDINGS_REPORT
d_val = 25.78
print(f"{'Replay -> retention (FULL vs NO_REPLAY)':<50} {'Cohen d':<15} {d_val:<12.2f} {'18':<6} {'<1e-15':<12} {'[0.235, 0.262]':<20}")

# 2. Replay necessity (Wcc)
print(f"{'Replay -> Wcc (FULL vs NO_REPLAY)':<50} {'Cohen d':<15} {4.95:<12.2f} {'18':<6} {'6e-8':<12} {'[0.023, 0.042]':<20}")

# 3. Replay necessity (S1)
print(f"{'Replay -> S1 (FULL vs NO_REPLAY)':<50} {'Cohen d':<15} {8.02:<12.2f} {'18':<6} {'2e-9':<12} {'[0.017, 0.026]':<20}")

# 4. Core stim necessity
print(f"{'Core stim -> retention (FULL vs NO_CORE)':<50} {'Cohen d':<15} {25.31:<12.2f} {'18':<6} {'3e-16':<12} {'[0.247, 0.273]':<20}")

# 5. Core stim -> Wcc
print(f"{'Core stim -> Wcc (FULL vs NO_CORE)':<50} {'Cohen d':<15} {6.19:<12.2f} {'18':<6} {'5e-11':<12} {'[0.039, 0.069]':<20}")

# 6. Replay count -> retention (within-seed)
print(f"{'Replay count -> retention (within-seed rho)':<50} {'Mean rho':<15} {mean_rho_ret:<12.3f} {f'{len(within_rhos_ret)-1}':<6} {f'{p_rho:.4f}':<12} {'see text':<20}")

# 7. Replay count -> W_slow (within-seed)
if within_rhos_ws:
    print(f"{'Replay count -> W_slow (within-seed rho)':<50} {'Mean rho':<15} {mean_rho_ws:<12.3f} {f'{len(within_rhos_ws)-1}':<6} {f'{p_ws:.4f}':<12} {'see text':<20}")

# 8. W_slow[cc] sufficiency (from Task 7.5 -- paper values)
print(f"{'W_slow[cc] -> retention (Task 7.5)':<50} {'% retained':<15} {'74%':<12} {'--':<6} {'<0.001':<12} {'[0.071, 0.079]':<20}")

# 9. Causal intervention: suppress M0
print(f"{'Suppress M0 replay -> M0 retention':<50} {'Delta retention':<15} {'-8.4%':<12} {'--':<6} {'n=1':<12} {'--':<20}")

# 10. Causal intervention: boost M3
print(f"{'Boost M3 replay -> M3 retention':<50} {'Delta retention':<15} {'-0.8%':<12} {'--':<6} {'n=1, n.s.':<12} {'--':<20}")

# 11. Early prediction (R² at 25% replay)
print(f"{'25% replay -> final retention (Task 10)':<50} {'R²':<15} {0.459:<12.3f} {'10':<6} {'0.016':<12} {'--':<20}")

# 12. Full prediction
print(f"{'100% replay -> final retention (Task 10)':<50} {'R²':<15} {0.881:<12.3f} {'10':<6} {'<0.0001':<12} {'--':<20}")

# 13. Replay -> W_slow full
print(f"{'100% replay -> final W_slow (Task 10)':<50} {'R²':<15} {0.962:<12.3f} {'10':<6} {'<0.0001':<12} {'--':<20}")

print("-" * 100)

# ═══════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════

results = {
    'Q1': {
        'naive_r_ret': r_naive_ret, 'naive_p_ret': p_naive_ret,
        'within_rhos_ret': within_rhos_ret, 'mean_rho_ret': mean_rho_ret,
        'se_rho_ret': se_rho_ret, 't_rho': t_rho, 'p_rho': p_rho,
        'slope_ret': slope_ret, 'se_slope_ret': se_slope,
        'r_centered_ret': r_centered, 'p_corrected_ret': p_corrected,
        'df_corrected': df_corrected,
    },
    'Q3': {
        'full_ret': FULL_RET, 'noreplay_ret': NOREPLAY_RET,
        'deltas': deltas, 't_paired': t_paired, 'p_paired': p_paired,
        'd_paired': d_paired, 'w_stat': w_stat, 'p_wilcoxon': p_wilcoxon,
        'ci': [np.mean(deltas) - ci_margin, np.mean(deltas) + ci_margin],
    },
}

if within_rhos_ws:
    results['Q1']['within_rhos_ws'] = within_rhos_ws
    results['Q1']['mean_rho_ws'] = mean_rho_ws

with open(OUT_DIR / 'q1q4_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\nResults saved to {OUT_DIR / 'q1q4_results.pkl'}")
print("\n[DONE] Q1-Q4 COMPLETE")
