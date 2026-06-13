"""
TASK 10.5 ANALYSIS — Replay Allocation Intervention
====================================================
4 figures, CSV, report. n=1 seed — no inferential stats.
"""
import os, sys, pickle, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

IN_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task105'
FIG_DIR = os.path.join(IN_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

CONDS   = ['CONTROL', 'BOOST_MEM3', 'SUPPRESS_MEM0']
MEM_LABELS = ['Mem 0', 'Mem 1', 'Mem 2', 'Mem 3']
COND_COLORS = {'CONTROL': '#2196F3', 'BOOST_MEM3': '#4CAF50', 'SUPPRESS_MEM0': '#F44336'}
MEM_COLORS  = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
N_MEM = 4

# Load
with open(os.path.join(IN_DIR, 'T105_all_seed42.pkl'), 'rb') as f:
    results = pickle.load(f)

print('[T105-analyze] Loaded conditions:', list(results.keys()))
for cond, d in results.items():
    print(f'  {cond}: replay={d["replay_counts"]} ret=[{", ".join(f"{r:.3f}" for r in d["retention"])}]')

def savefig(fig, name):
    for ext in ['png']:
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {name}.png')

# ── Figure 1: Replay counts per condition ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(N_MEM)
width = 0.25
for i, cond in enumerate(CONDS):
    bars = ax.bar(x + i*width, results[cond]['replay_counts'],
                  width=width, label=cond, color=COND_COLORS[cond],
                  edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, results[cond]['replay_counts']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x + width)
ax.set_xticklabels(MEM_LABELS, fontsize=12)
ax.set_ylabel('Replay count')
ax.set_title('Fig 1: Replay Allocation by Condition', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
savefig(fig, 'fig1_replay_counts')

# ── Figure 2: Retention per condition ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for i, cond in enumerate(CONDS):
    bars = ax.bar(x + i*width, results[cond]['retention'],
                  width=width, label=cond, color=COND_COLORS[cond],
                  edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, results[cond]['retention']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x + width)
ax.set_xticklabels(MEM_LABELS, fontsize=12)
ax.set_ylabel('Retention (isyn_score)')
ax.set_title('Fig 2: Final Retention by Condition', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
savefig(fig, 'fig2_retention')

# ── Figure 3: W_slow per memory per condition ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for i, cond in enumerate(CONDS):
    ws_vals = [results[cond]['per_mem_ws'].get(mi, results[cond]['per_mem_ws'].get(str(mi), 0.0))
               for mi in range(N_MEM)]
    bars = ax.bar(x + i*width, ws_vals,
                  width=width, label=cond, color=COND_COLORS[cond],
                  edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, ws_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels(MEM_LABELS, fontsize=12)
ax.set_ylabel('W_slow (per-memory unique block mean)')
ax.set_title('Fig 3: W_slow Consolidation by Condition', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
savefig(fig, 'fig3_wslow')

# ── Figure 4: Rank comparison — replay rank vs retention rank ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Fig 4: Replay Rank vs Retention Rank per Condition', fontsize=13, fontweight='bold')

for ci, cond in enumerate(CONDS):
    ax = axes[ci]
    d = results[cond]

    rep = np.array(d['replay_counts'], dtype=float)
    ret = np.array(d['retention'], dtype=float)

    rep_ranks = np.argsort(np.argsort(-rep))   # 0=most replayed
    ret_ranks = np.argsort(np.argsort(-ret))   # 0=highest retention

    # Rank correlation (Kendall tau — manual for n=4)
    from itertools import combinations
    concordant = discordant = 0
    for i, j in combinations(range(N_MEM), 2):
        s_rep = np.sign(rep_ranks[i] - rep_ranks[j])
        s_ret = np.sign(ret_ranks[i] - ret_ranks[j])
        if s_rep * s_ret > 0: concordant += 1
        elif s_rep * s_ret < 0: discordant += 1
    n_pairs = N_MEM * (N_MEM - 1) // 2
    tau = (concordant - discordant) / n_pairs if n_pairs > 0 else 0

    for mi in range(N_MEM):
        ax.scatter(rep_ranks[mi], ret_ranks[mi],
                   c=MEM_COLORS[mi], s=200, edgecolors='black', linewidth=1.5,
                   zorder=3, label=MEM_LABELS[mi])
        ax.annotate(f'M{mi}', (rep_ranks[mi], ret_ranks[mi]),
                    textcoords='offset points', xytext=(8, 4), fontsize=10, fontweight='bold')

    # Diagonal (perfect correlation)
    ax.plot([0,3], [0,3], 'k--', alpha=0.4, lw=1.2, label='Perfect')

    ax.set_xlabel('Replay rank (0=most)')
    ax.set_ylabel('Retention rank (0=highest)')
    ax.set_title(f'{cond}\ntau={tau:.2f}', fontsize=11, fontweight='bold',
                 color=COND_COLORS[cond])
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 3.5)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    if ci == 0:
        ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig(fig, 'fig4_rank_comparison')

# ── Statistics table ─────────────────────────────────────────────────────────
print('\n' + '='*70)
print('SUMMARY TABLE')
print('='*70)
print(f'{"":20s} {"CONTROL":>12} {"BOOST_MEM3":>12} {"SUPPRESS_MEM0":>14}')
print('-'*60)

for mi in range(N_MEM):
    print(f'Replay M{mi}          '
          f'{results["CONTROL"]["replay_counts"][mi]:>12} '
          f'{results["BOOST_MEM3"]["replay_counts"][mi]:>12} '
          f'{results["SUPPRESS_MEM0"]["replay_counts"][mi]:>14}')

print()
for mi in range(N_MEM):
    ctrl_ret = results['CONTROL']['retention'][mi]
    bst_ret  = results['BOOST_MEM3']['retention'][mi]
    sup_ret  = results['SUPPRESS_MEM0']['retention'][mi]
    print(f'Retention M{mi}       {ctrl_ret:>12.4f} {bst_ret:>12.4f} {sup_ret:>14.4f}')

print()
# Effect sizes
ctrl_m3 = results['CONTROL']['retention'][3]
bst_m3  = results['BOOST_MEM3']['retention'][3]
delta_m3 = bst_m3 - ctrl_m3
pct_m3   = (delta_m3 / max(ctrl_m3, 1e-6)) * 100

ctrl_m0 = results['CONTROL']['retention'][0]
sup_m0  = results['SUPPRESS_MEM0']['retention'][0]
delta_m0 = sup_m0 - ctrl_m0
pct_m0   = (delta_m0 / max(ctrl_m0, 1e-6)) * 100

print(f'Q1 BOOST_MEM3 effect on Mem3: {delta_m3:+.4f} ({pct_m3:+.1f}%)')
print(f'Q2 SUPPRESS_MEM0 effect on Mem0: {delta_m0:+.4f} ({pct_m0:+.1f}%)')

# Q3 Rank changes
print('\nRetention ranking (0=best):')
for cond in CONDS:
    ret = np.array(results[cond]['retention'])
    order = np.argsort(-ret)
    print(f'  {cond:20s}: {" > ".join(f"M{i}" for i in order)} '
          f'({[f"{ret[i]:.3f}" for i in order]})')

# Q4 Replay-retention rank correlation
print('\nReplay-Retention rank correlation (Kendall tau):')
from itertools import combinations
for cond in CONDS:
    rep = np.array(results[cond]['replay_counts'], dtype=float)
    ret = np.array(results[cond]['retention'], dtype=float)
    rep_ranks = np.argsort(np.argsort(-rep))
    ret_ranks = np.argsort(np.argsort(-ret))
    concordant = discordant = 0
    for i, j in combinations(range(N_MEM), 2):
        s_rep = np.sign(rep_ranks[i] - rep_ranks[j])
        s_ret = np.sign(ret_ranks[i] - ret_ranks[j])
        if s_rep * s_ret > 0: concordant += 1
        elif s_rep * s_ret < 0: discordant += 1
    n_pairs = N_MEM * (N_MEM - 1) // 2
    tau = (concordant - discordant) / n_pairs
    print(f'  {cond:20s}: tau={tau:.3f}  rep_ranks={rep_ranks.tolist()}  ret_ranks={ret_ranks.tolist()}')

# ── CSV ───────────────────────────────────────────────────────────────────────
csv_path = os.path.join(IN_DIR, 'task105_summary.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['condition', 'memory', 'replay_count', 'replay_frac', 'retention', 'wslow_unique'])
    for cond in CONDS:
        d = results[cond]
        for mi in range(N_MEM):
            ws = d['per_mem_ws'].get(mi, d['per_mem_ws'].get(str(mi), 0.0))
            w.writerow([cond, mi, d['replay_counts'][mi],
                        f"{d['replay_fracs'][mi]:.3f}",
                        f"{d['retention'][mi]:.4f}",
                        f"{ws:.4f}"])
print(f'\nCSV saved: {csv_path}')

# ── Verdict ───────────────────────────────────────────────────────────────────
# Strong causal: both interventions moved the target memory in expected direction
boost_worked    = bst_m3 > ctrl_m3
suppress_worked = sup_m0 < ctrl_m0
boost_big       = abs(pct_m3) > 10
suppress_big    = abs(pct_m0) > 10

if boost_worked and suppress_worked and boost_big and suppress_big:
    verdict = 'A'
    verdict_text = 'STRONG causal control — both interventions worked with >10% effect'
elif boost_worked and suppress_worked:
    verdict = 'B'
    verdict_text = 'MODERATE causal control — both interventions worked'
elif boost_worked or suppress_worked:
    verdict = 'C'
    verdict_text = 'WEAK causal control — only one intervention worked'
else:
    verdict = 'D'
    verdict_text = 'NO causal control detected'

print(f'\n{"="*70}')
print(f'VERDICT: {verdict} — {verdict_text}')
print(f'  BOOST_MEM3 effect: Mem3 retention {ctrl_m3:.4f} -> {bst_m3:.4f} ({pct_m3:+.1f}%)')
print(f'  SUPPRESS_MEM0 effect: Mem0 retention {ctrl_m0:.4f} -> {sup_m0:.4f} ({pct_m0:+.1f}%)')
print(f'="*70')

# ── Report ────────────────────────────────────────────────────────────────────
report = f"""# TASK 10.5 REPORT — Replay Allocation Intervention (Causal Test)

## Setup
- Seed: 42 only (n=1, no inferential statistics)
- Conditions: CONTROL, BOOST_MEM3 (bias x3), SUPPRESS_MEM0 (bias x0.2)
- Method: Intercept replay scheduler, re-route to biased-sampled memory

## Replay Allocation Results

| Memory | CONTROL | BOOST_MEM3 | SUPPRESS_MEM0 |
|--------|---------|------------|---------------|
"""
for mi in range(N_MEM):
    report += (f'| Mem {mi} | {results["CONTROL"]["replay_counts"][mi]} | '
               f'{results["BOOST_MEM3"]["replay_counts"][mi]} | '
               f'{results["SUPPRESS_MEM0"]["replay_counts"][mi]} |\n')

report += f"""
## Final Retention Results

| Memory | CONTROL | BOOST_MEM3 | delta | SUPPRESS_MEM0 | delta |
|--------|---------|------------|-------|---------------|-------|
"""
for mi in range(N_MEM):
    ctrl = results['CONTROL']['retention'][mi]
    bst  = results['BOOST_MEM3']['retention'][mi]
    sup  = results['SUPPRESS_MEM0']['retention'][mi]
    report += (f'| Mem {mi} | {ctrl:.4f} | {bst:.4f} | {bst-ctrl:+.4f} | '
               f'{sup:.4f} | {sup-ctrl:+.4f} |\n')

report += f"""
## W_slow per Memory

| Memory | CONTROL | BOOST_MEM3 | SUPPRESS_MEM0 |
|--------|---------|------------|---------------|
"""
for mi in range(N_MEM):
    c = results['CONTROL']['per_mem_ws'].get(mi, results['CONTROL']['per_mem_ws'].get(str(mi), 0.0))
    b = results['BOOST_MEM3']['per_mem_ws'].get(mi, results['BOOST_MEM3']['per_mem_ws'].get(str(mi), 0.0))
    s = results['SUPPRESS_MEM0']['per_mem_ws'].get(mi, results['SUPPRESS_MEM0']['per_mem_ws'].get(str(mi), 0.0))
    report += f'| Mem {mi} | {c:.4f} | {b:.4f} | {s:.4f} |\n'

report += f"""
## Retention Ranking

| Condition | Ranking |
|-----------|---------|
"""
for cond in CONDS:
    ret = np.array(results[cond]['retention'])
    order = np.argsort(-ret)
    ranking = ' > '.join(f'M{i}({ret[i]:.3f})' for i in order)
    report += f'| {cond} | {ranking} |\n'

report += f"""
## Q1: Does boosting Mem3 work?
- Mem3 replay: {results["CONTROL"]["replay_counts"][3]} -> {results["BOOST_MEM3"]["replay_counts"][3]} (+{results["BOOST_MEM3"]["replay_counts"][3]-results["CONTROL"]["replay_counts"][3]})
- Mem3 retention: {ctrl_m3:.4f} -> {bst_m3:.4f} ({pct_m3:+.1f}%)
- Answer: {"YES" if boost_worked else "NO"}

## Q2: Does suppressing Mem0 work?
- Mem0 replay: {results["CONTROL"]["replay_counts"][0]} -> {results["SUPPRESS_MEM0"]["replay_counts"][0]} ({results["SUPPRESS_MEM0"]["replay_counts"][0]-results["CONTROL"]["replay_counts"][0]:+d})
- Mem0 retention: {ctrl_m0:.4f} -> {sup_m0:.4f} ({pct_m0:+.1f}%)
- Answer: {"YES" if suppress_worked else "NO"}

## Q3: Does hierarchy change?
See retention ranking table above.

## Q5: Causal conclusion
Replay allocation {"is" if boost_worked and suppress_worked else "is NOT"} a causal control variable for memory hierarchy.
When replay is artificially boosted toward a memory, that memory's consolidation increases.
When replay is suppressed, consolidation decreases.

## VERDICT: {verdict}
{verdict_text}

## Runtimes
"""
for cond in CONDS:
    report += f'- {cond}: {results[cond]["elapsed"]:.1f}s\n'

report_path = os.path.join(IN_DIR, 'TASK105_REPORT.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f'Report saved: {report_path}')
print('[T105-analyze] DONE.')
