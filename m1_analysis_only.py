"""
M1 Analysis Only -- runs statistics and generates figure from saved CSV.
Run after m1_task105_20seeds.py data collection completes.
"""
import os, sys
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

SEEDS_20 = [42 + i*1000 for i in range(20)]
OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\m1_results'
RESULTS_FILE = os.path.join(OUT_DIR, 'm1_task105_20seeds.csv')

from scipy import stats
import statsmodels.formula.api as smf

def is_done(seed, condition):
    done = pd.read_csv(RESULTS_FILE)
    return ((done['seed'] == seed) & (done['condition'] == condition)).any()

df = pd.read_csv(RESULTS_FILE)
print(f'[M1] Loaded {len(df)} rows from {RESULTS_FILE}')
print(df.groupby(['condition','memory_id'])['retention'].agg(['mean','std','count']).to_string())

m0_ctrl  = df[(df.condition=='CONTROL')      & (df.memory_id==0)]['retention'].values
m0_supp  = df[(df.condition=='SUPPRESS_MEM0')& (df.memory_id==0)]['retention'].values
m3_ctrl  = df[(df.condition=='CONTROL')      & (df.memory_id==3)]['retention'].values
m3_boost = df[(df.condition=='BOOST_MEM3')   & (df.memory_id==3)]['retention'].values

t_supp,  p_supp  = stats.ttest_rel(m0_ctrl, m0_supp)
t_boost, p_boost = stats.ttest_rel(m3_ctrl, m3_boost)
d_supp  = (m0_ctrl.mean() - m0_supp.mean())  / np.std(np.concatenate([m0_ctrl, m0_supp]))
d_boost = (m3_ctrl.mean() - m3_boost.mean()) / np.std(np.concatenate([m3_ctrl, m3_boost]))

print(f'\n[M1] === KEY STATISTICS ===')
print(f'SUPPRESS M0: CTRL={m0_ctrl.mean():.4f}+/-{m0_ctrl.std():.4f}  SUPP={m0_supp.mean():.4f}+/-{m0_supp.std():.4f}')
print(f'  t({len(m0_ctrl)-1})={t_supp:.3f}, p={p_supp:.6f}, d={d_supp:.3f}')
print(f'BOOST M3:    CTRL={m3_ctrl.mean():.4f}+/-{m3_ctrl.std():.4f}  BOOST={m3_boost.mean():.4f}+/-{m3_boost.std():.4f}')
print(f'  t({len(m3_ctrl)-1})={t_boost:.3f}, p={p_boost:.6f}, d={d_boost:.3f}')

n_supp_degraded = sum(
    df[(df.seed==s)&(df.condition=='SUPPRESS_MEM0')&(df.memory_id==0)]['retention'].values[0] <
    df[(df.seed==s)&(df.condition=='CONTROL')&(df.memory_id==0)]['retention'].values[0]
    for s in SEEDS_20 if is_done(s,'SUPPRESS_MEM0') and is_done(s,'CONTROL')
)
n_boost_failed = sum(
    df[(df.seed==s)&(df.condition=='BOOST_MEM3')&(df.memory_id==3)]['retention'].values[0] <=
    df[(df.seed==s)&(df.condition=='CONTROL')&(df.memory_id==3)]['retention'].values[0]
    for s in SEEDS_20 if is_done(s,'BOOST_MEM3') and is_done(s,'CONTROL')
)
n_done = sum(is_done(s,'CONTROL') for s in SEEDS_20)
print(f'SUPPRESS degrades M0: {n_supp_degraded}/{n_done} seeds')
print(f'BOOST fails to help M3: {n_boost_failed}/{n_done} seeds')

# Mixed-effects model
try:
    model = smf.mixedlm("retention ~ C(condition) * C(memory_id)", data=df, groups=df["seed"])
    result_lm = model.fit(reml=False)
    print('\n[M1] Mixed-effects model summary:')
    print(result_lm.summary())
except Exception as e:
    print(f'[M1] Mixed-effects failed: {e}')

# Save paste-ready text
summary_path = os.path.join(OUT_DIR, 'm1_statistics_summary.txt')
with open(summary_path, 'w') as f:
    f.write('=== M1: Task 10.5 Across 20 Seeds -- Statistics Summary ===\n\n')
    f.write(f'SUPPRESS M0: CTRL={m0_ctrl.mean():.4f}+/-{m0_ctrl.std():.4f}, SUPP={m0_supp.mean():.4f}+/-{m0_supp.std():.4f}\n')
    f.write(f'  t({len(m0_ctrl)-1})={t_supp:.3f}, p={p_supp:.6f}, d={d_supp:.3f}\n')
    f.write(f'  {n_supp_degraded}/{n_done} seeds showed degradation\n\n')
    f.write(f'BOOST M3:    CTRL={m3_ctrl.mean():.4f}+/-{m3_ctrl.std():.4f}, BOOST={m3_boost.mean():.4f}+/-{m3_boost.std():.4f}\n')
    f.write(f'  t({len(m3_ctrl)-1})={t_boost:.3f}, p={p_boost:.6f}, d={d_boost:.3f}\n')
    f.write(f'  {n_boost_failed}/{n_done} seeds showed no gain\n\n')
    f.write('=== PASTE-READY PAPER TEXT ===\n\n')
    f.write(
        f'"Replicating the Task 10.5 intervention across 20 independent seeds confirmed the\n'
        f'suppression-boost asymmetry. Suppression of M0 replay significantly degraded M0\n'
        f'retention (t({len(m0_ctrl)-1}) = {t_supp:.2f}, p = {p_supp:.4f}, d = {d_supp:.2f};\n'
        f'{n_supp_degraded}/20 seeds showed degradation: mean CTRL={m0_ctrl.mean():.4f},\n'
        f'mean SUPP={m0_supp.mean():.4f}). Boosting M3 replay produced no significant change\n'
        f'in M3 retention (t({len(m3_ctrl)-1}) = {t_boost:.2f}, p = {p_boost:.4f}, d = {d_boost:.2f};\n'
        f'{n_boost_failed}/20 seeds showed no gain: mean CTRL={m3_ctrl.mean():.4f},\n'
        f'mean BOOST={m3_boost.mean():.4f}). This asymmetry -- suppression effective,\n'
        f'boosting ineffective -- is consistent with the encoding-seed threshold\n'
        f'interpretation: replay can amplify traces above a W_slow threshold, but\n'
        f'cannot inscribe new traces in memories whose W_slow seed is already\n'
        f'saturated by interference."\n'
    )
print(f'[M1] Summary saved: {summary_path}')

# Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

done_seeds = [s for s in SEEDS_20 if is_done(s,'CONTROL') and is_done(s,'SUPPRESS_MEM0')]
for s in done_seeds:
    c_val = df[(df.seed==s)&(df.condition=='CONTROL')&(df.memory_id==0)]['retention'].values[0]
    s_val = df[(df.seed==s)&(df.condition=='SUPPRESS_MEM0')&(df.memory_id==0)]['retention'].values[0]
    ax1.plot([1,2], [c_val, s_val], 'o-', color='grey', alpha=0.35, linewidth=0.8, markersize=4)
for pos, cond, col in [(1,'CONTROL','steelblue'), (2,'SUPPRESS_MEM0','firebrick')]:
    vals = df[(df.condition==cond)&(df.memory_id==0)]['retention']
    ax1.errorbar(pos, vals.mean(), vals.sem()*1.96, fmt='s', color=col,
                 markersize=10, capsize=5, zorder=5,
                 label=f'{cond[:7]}: {vals.mean():.4f}+/-{vals.sem():.4f}')
ax1.set_xticks([1,2]); ax1.set_xticklabels(['CONTROL','SUPPRESS\nMEM0'], fontsize=11)
ax1.set_ylabel('M0 Retention (isyn_score)', fontsize=11)
ax1.set_title(f'A. Suppression degrades M0\n(n={len(done_seeds)} seeds, t={t_supp:.2f}, p={p_supp:.4f})', fontsize=10)
ax1.legend(fontsize=8)

done_seeds_b = [s for s in SEEDS_20 if is_done(s,'CONTROL') and is_done(s,'BOOST_MEM3')]
for s in done_seeds_b:
    c_val = df[(df.seed==s)&(df.condition=='CONTROL')&(df.memory_id==3)]['retention'].values[0]
    b_val = df[(df.seed==s)&(df.condition=='BOOST_MEM3')&(df.memory_id==3)]['retention'].values[0]
    ax2.plot([1,2], [c_val, b_val], 'o-', color='grey', alpha=0.35, linewidth=0.8, markersize=4)
for pos, cond, col in [(1,'CONTROL','steelblue'), (2,'BOOST_MEM3','darkorange')]:
    vals = df[(df.condition==cond)&(df.memory_id==3)]['retention']
    ax2.errorbar(pos, vals.mean(), vals.sem()*1.96, fmt='s', color=col,
                 markersize=10, capsize=5, zorder=5,
                 label=f'{cond[:7]}: {vals.mean():.4f}+/-{vals.sem():.4f}')
ax2.set_xticks([1,2]); ax2.set_xticklabels(['CONTROL','BOOST\nMEM3'], fontsize=11)
ax2.set_ylabel('M3 Retention (isyn_score)', fontsize=11)
ax2.set_title(f'B. Boosting fails to rescue M3\n(n={len(done_seeds_b)} seeds, t={t_boost:.2f}, p={p_boost:.4f})', fontsize=10)
ax2.legend(fontsize=8)

plt.suptitle('Task 10.5: Causal replay manipulation across 20 seeds\nGrey lines = individual seeds; squares = mean +/- 95% CI', fontsize=11, y=1.01)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'm1_task105_20seeds.png')
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
fig.savefig(fig_path.replace('.png','.pdf'), bbox_inches='tight')
plt.close()
print(f'[M1] Figure saved: {fig_path}')
print('\n[M1] === ANALYSIS DONE ===')
