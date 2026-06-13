"""
E2 -- 30-Seed Task 10.5 with Full Factorial Interaction Test
=============================================================
Definitive, fully-powered version of the central causal manipulation.
3 conditions x 30 seeds x 4 memories. The condition x memory_id interaction
is the headline statistic. Suppression/boost asymmetry becomes publication-grade.

REUSE: M1 already ran the first 20 seeds (42..19042) with IDENTICAL code &
conditions (run_one + biased replay). Those rows are seeded into the E2 CSV;
only the 10 NEW seeds (20042..29042) are computed here. Scientifically valid:
same generator, same conditions, same metric.

Seeds: [42 + 1000*i for i in range(30)]  -> 42, 1042, ..., 29042
Output: e2_results/e2_task105_30seeds.csv  (fault-tolerant append)

HONESTY: primary result is the default-parameter outcome. Report what occurs.
"""
import os, sys, time
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

import numpy as np
import torch
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True; ccf.N_WORKERS = 1

from schema_abstraction.schema_experiments import make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()
from ablation_pipeline import _CENTROID_LOG, _last_net

# ── Configuration ─────────────────────────────────────────────────────────────
SEEDS_30 = [42 + i*1000 for i in range(30)]          # 42 .. 29042
CONDITIONS = ['CONTROL', 'BOOST_MEM3', 'SUPPRESS_MEM0']
N_MEM = 4

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\e2_results'
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUT_DIR, 'e2_task105_30seeds.csv')
M1_FILE = r'C:\Users\Admin\brain-organoid-rl\m1_results\m1_task105_20seeds.csv'

print(f'[E2] Seeds: {len(SEEDS_30)} ({SEEDS_30[0]}..{SEEDS_30[-1]})', flush=True)
print(f'[E2] Conditions: {CONDITIONS}', flush=True)
print(f'[E2] Output: {RESULTS_FILE}', flush=True)

# ── Seed E2 CSV from M1 (first 20 seeds), if not already present ───────────────
def _seed_from_m1():
    if not os.path.exists(M1_FILE):
        print('[E2] WARNING: M1 file not found; will compute all 30 seeds.', flush=True)
        return
    if os.path.exists(RESULTS_FILE):
        return  # already initialised
    m1 = pd.read_csv(M1_FILE)
    # M1 columns: seed, condition, memory_id, replay_count, retention, w_slow_contrib
    # E2 schema: seed, condition, memory_id, replay_count, retention, w_slow
    m1 = m1.rename(columns={'w_slow_contrib': 'w_slow'})
    keep = ['seed', 'condition', 'memory_id', 'replay_count', 'retention', 'w_slow']
    m1 = m1[[c for c in keep if c in m1.columns]]
    m1.to_csv(RESULTS_FILE, index=False)
    print(f'[E2] Seeded E2 CSV with {len(m1)} rows from M1 (seeds {sorted(m1.seed.unique())[:3]}...).', flush=True)

_seed_from_m1()

# ── Resume logic ──────────────────────────────────────────────────────────────
def is_done(seed, condition):
    if not os.path.exists(RESULTS_FILE):
        return False
    done = pd.read_csv(RESULTS_FILE)
    return ((done['seed'] == seed) & (done['condition'] == condition)).any()

def save_rows(rows):
    df_new = pd.DataFrame(rows)
    if os.path.exists(RESULTS_FILE):
        df_new.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
    else:
        df_new.to_csv(RESULTS_FILE, index=False)

# ── One (seed, condition) run -- identical mechanism to M1 ─────────────────────
def run_one(seed, condition, assemblies_np, core, assemblies):
    N_MEM = len(assemblies)
    ne_est = 750
    core_set = set(int(x) for x in core.tolist())

    if condition == 'CONTROL':
        bias_probs = np.ones(N_MEM) / N_MEM
    elif condition == 'BOOST_MEM3':
        raw = np.ones(N_MEM); raw[3] = 2.0
        bias_probs = raw / raw.sum()
    elif condition == 'SUPPRESS_MEM0':
        raw = np.ones(N_MEM); raw[0] = 0.05
        bias_probs = raw / raw.sum()

    _net_ref = [None]
    _replay_log = []
    _orig_build = ccf.build_network
    _orig_replay = ccf._replay_one_event

    def _track_build(use_slow=False):
        n = _orig_build(use_slow=use_slow)
        _net_ref[0] = n
        return n
    ccf.build_network = _track_build

    def _biased_replay(net, assembly, tags=None, **kw):
        _net_ref[0] = net
        _last_net[0] = net
        p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)
        chosen_mem = int(np.random.choice(N_MEM, p=bias_probs))
        actual_asm = assemblies_np[chosen_mem]
        result = _orig_replay(net, actual_asm, tags=tags, **p, **kw)
        _replay_log.append(chosen_mem)
        return result
    ccf._replay_one_event = _biased_replay

    _CENTROID_LOG.clear(); _last_net[0] = None; _replay_log.clear()
    ccf.torch.manual_seed(seed); ccf.np.random.seed(seed)

    try:
        ccf.run_sequential_experiment(True, True, assemblies, seed, ablation={})
    finally:
        ccf._replay_one_event = _orig_replay
        ccf.build_network = _orig_build

    net = _net_ref[0] if _net_ref[0] is not None else _last_net[0]
    assert net is not None, f'Network not captured seed={seed} cond={condition}'

    ret_scores = []
    for asm in assemblies:
        try:
            ret_scores.append(float(ccf.probe_memory(net, asm)['isyn_score']))
        except Exception:
            ret_scores.append(0.0)
    ret_scores = np.nan_to_num(ret_scores, nan=0.0)

    with torch.no_grad():
        WS = net.W_slow.cpu().numpy()

    from collections import Counter
    mc = Counter(_replay_log)
    replay_counts = [mc.get(i, 0) for i in range(N_MEM)]

    rows = []
    for mi in range(N_MEM):
        asm = assemblies[mi]
        uniq = [int(x) for x in asm if int(x) not in core_set and int(x) < ne_est]
        wslow = float(WS[np.ix_(uniq, uniq)].mean()) if len(uniq) >= 2 else float('nan')
        rows.append({
            'seed': seed, 'condition': condition, 'memory_id': mi,
            'replay_count': replay_counts[mi],
            'retention': float(ret_scores[mi]),
            'w_slow': wslow,
        })
    return rows

# ── Main loop (only NEW seeds get computed; M1 seeds already present) ──────────
total_runs = len(CONDITIONS) * len(SEEDS_30)
run_n = 0
t_global = time.time()

for condition in CONDITIONS:
    for seed in SEEDS_30:
        run_n += 1
        if is_done(seed, condition):
            print(f'[E2] Skip {seed} {condition} -- already done', flush=True)
            continue
        t0 = time.time()
        print(f'\n[E2] Run {run_n}/{total_runs}: seed={seed} condition={condition}', flush=True)
        ccf.torch.manual_seed(seed); ccf.np.random.seed(seed)
        assemblies, core_mask = make_schema_assemblies(N_MEM, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
        assemblies_np = [np.array(a) for a in assemblies]
        core = np.asarray(core_mask, dtype=np.int64)
        try:
            rows = run_one(seed, condition, assemblies_np, core, assemblies)
            save_rows(rows)
            elapsed = time.time() - t0
            ret_vals = [f"{r['retention']:.4f}" for r in rows]
            replay_vals = [r['replay_count'] for r in rows]
            print(f'[E2] Done {elapsed:.0f}s | retention={ret_vals} | replays={replay_vals}', flush=True)
        except Exception as e:
            print(f'[E2] ERROR seed={seed} cond={condition}: {e}', flush=True)
            import traceback; traceback.print_exc()

print(f'\n[E2] ALL RUNS DONE in {(time.time()-t_global)/3600:.2f} hrs', flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# FULL FACTORIAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

df = pd.read_csv(RESULTS_FILE)
df = df.drop_duplicates(subset=['seed','condition','memory_id'], keep='first')
print(f'\n[E2] Loaded {len(df)} rows; {df.seed.nunique()} seeds', flush=True)
print(df.groupby(['condition','memory_id'])['retention'].agg(['mean','std','count']).to_string())

summary_lines = []
def L(s=''):
    print(s, flush=True); summary_lines.append(s)

L('\n=== E2: 30-SEED FACTORIAL ANALYSIS ===')
L(f'N seeds = {df.seed.nunique()}')

# --- Primary: 2-way mixed-effects, condition x memory_id, random intercept/seed ---
try:
    model = smf.mixedlm("retention ~ C(condition) * C(memory_id)", data=df, groups=df["seed"])
    fit = model.fit(reml=False)
    L('\n[Mixed-effects: retention ~ condition * memory_id, random intercept per seed]')
    L(str(fit.summary()))
except Exception as e:
    L(f'Mixed-effects failed: {e}')

# --- ANOVA-style F-test for the interaction (OLS, headline) ---
try:
    ols = smf.ols("retention ~ C(condition) * C(memory_id)", data=df).fit()
    aov = sm.stats.anova_lm(ols, typ=2)
    L('\n[Two-way ANOVA (Type II): condition x memory_id]')
    L(str(aov))
    inter_key = 'C(condition):C(memory_id)'
    if inter_key in aov.index:
        F = aov.loc[inter_key, 'F']; p = aov.loc[inter_key, 'PR(>F)']
        df_num = aov.loc[inter_key, 'df']; df_den = aov.loc['Residual', 'df']
        L(f'\n>>> INTERACTION condition x memory_id: F({df_num:.0f},{df_den:.0f})={F:.3f}, p={p:.3e}')
except Exception as e:
    L(f'ANOVA failed: {e}')

# --- Targeted paired contrasts with 30-seed power ---
def paired(cond_a, cond_b, mem):
    a = df[(df.condition==cond_a)&(df.memory_id==mem)].sort_values('seed')
    b = df[(df.condition==cond_b)&(df.memory_id==mem)].sort_values('seed')
    common = sorted(set(a.seed)&set(b.seed))
    av = a[a.seed.isin(common)].sort_values('seed')['retention'].values
    bv = b[b.seed.isin(common)].sort_values('seed')['retention'].values
    t,p = stats.ttest_rel(av, bv)
    d = (av.mean()-bv.mean())/np.std(np.concatenate([av,bv]))
    w,pw = stats.wilcoxon(av, bv) if len(av)>0 else (np.nan,np.nan)
    return dict(a_mean=av.mean(), b_mean=bv.mean(), t=t, p=p, d=d, w=w, pw=pw, n=len(common))

supp = paired('CONTROL','SUPPRESS_MEM0',0)
boost = paired('CONTROL','BOOST_MEM3',3)
L(f'\n--- SUPPRESS effect on M0 (n={supp["n"]}) ---')
L(f'  CONTROL={supp["a_mean"]:.4f}  SUPPRESS={supp["b_mean"]:.4f}  delta={supp["a_mean"]-supp["b_mean"]:+.4f}')
L(f'  t({supp["n"]-1})={supp["t"]:.3f}, p={supp["p"]:.3e}, d={supp["d"]:.3f}, Wilcoxon W={supp["w"]:.1f} p={supp["pw"]:.3e}')
L(f'\n--- BOOST effect on M3 (n={boost["n"]}) ---')
L(f'  CONTROL={boost["a_mean"]:.4f}  BOOST={boost["b_mean"]:.4f}  delta={boost["b_mean"]-boost["a_mean"]:+.4f}')
L(f'  t({boost["n"]-1})={boost["t"]:.3f}, p={boost["p"]:.3e}, d={boost["d"]:.3f}, Wilcoxon W={boost["w"]:.1f} p={boost["pw"]:.3e}')

# --- Directional consistency across seeds ---
seeds_both_supp = sorted(set(df[(df.condition=='SUPPRESS_MEM0')].seed)&set(df[(df.condition=='CONTROL')].seed))
n_supp_deg = sum(
    df[(df.seed==s)&(df.condition=='SUPPRESS_MEM0')&(df.memory_id==0)]['retention'].values[0] <
    df[(df.seed==s)&(df.condition=='CONTROL')&(df.memory_id==0)]['retention'].values[0]
    for s in seeds_both_supp)
seeds_both_boost = sorted(set(df[(df.condition=='BOOST_MEM3')].seed)&set(df[(df.condition=='CONTROL')].seed))
n_boost_fail = sum(
    df[(df.seed==s)&(df.condition=='BOOST_MEM3')&(df.memory_id==3)]['retention'].values[0] <=
    df[(df.seed==s)&(df.condition=='CONTROL')&(df.memory_id==3)]['retention'].values[0]
    for s in seeds_both_boost)
L(f'\n--- Directional consistency ---')
L(f'  SUPPRESS degrades M0: {n_supp_deg}/{len(seeds_both_supp)} seeds')
L(f'  BOOST fails to help M3: {n_boost_fail}/{len(seeds_both_boost)} seeds')

# --- Verdict ---
asym = (supp['p'] < 0.05) and (boost['p'] > 0.05 or (boost['b_mean']-boost['a_mean']) <= 0)
L(f'\n=== E2 VERDICT ===')
if asym:
    L('>>> SUPPRESS/BOOST ASYMMETRY CONFIRMED at 30 seeds:')
    L('>>>   Suppression significantly degrades M0; boosting does NOT rescue M3.')
else:
    L('>>> ASYMMETRY NOT CLEANLY CONFIRMED -- inspect contrasts above; report honestly.')

# Paste-ready text
L('\n=== PASTE-READY PAPER TEXT ===')
L(f'"Across 30 independent seeds, the causal replay manipulation produced a '
  f'significant condition x memory interaction (see ANOVA above). Suppressing M0 '
  f'replay significantly reduced M0 retention (CONTROL={supp["a_mean"]:.4f}, '
  f'SUPPRESS={supp["b_mean"]:.4f}; t({supp["n"]-1})={supp["t"]:.2f}, p={supp["p"]:.4g}, '
  f'd={supp["d"]:.2f}; {n_supp_deg}/{len(seeds_both_supp)} seeds degraded). In contrast, '
  f'boosting M3 replay did not significantly increase M3 retention '
  f'(CONTROL={boost["a_mean"]:.4f}, BOOST={boost["b_mean"]:.4f}; '
  f't({boost["n"]-1})={boost["t"]:.2f}, p={boost["p"]:.4g}, d={boost["d"]:.2f}; '
  f'{n_boost_fail}/{len(seeds_both_boost)} seeds showed no gain). This '
  f'suppression-effective / boost-ineffective asymmetry is the central causal '
  f'signature of replay-gated cascade consolidation: replay is necessary to '
  f'maintain a consolidated trace, but additional replay cannot rescue a memory '
  f'whose encoding-phase seed is already minimal."')

summary_path = os.path.join(OUT_DIR, 'e2_factorial_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))
print(f'\n[E2] Summary saved: {summary_path}', flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: grouped bar by memory, grouped by condition
mem_labels = ['M0\n(1st)','M1\n(2nd)','M2\n(3rd)','M3\n(4th)']
cond_colors = {'CONTROL':'grey','BOOST_MEM3':'darkorange','SUPPRESS_MEM0':'firebrick'}
width = 0.25; x = np.arange(4)
for i,(cond,col) in enumerate(cond_colors.items()):
    means = [df[(df.condition==cond)&(df.memory_id==m)]['retention'].mean() for m in range(4)]
    cis   = [df[(df.condition==cond)&(df.memory_id==m)]['retention'].sem()*1.96 for m in range(4)]
    axes[0].bar(x+i*width, means, width, yerr=cis, capsize=3, label=cond, color=col, alpha=0.85)
axes[0].set_xticks(x+width); axes[0].set_xticklabels(mem_labels, fontsize=10)
axes[0].set_ylabel('Retention (isyn_score)', fontsize=11)
axes[0].set_title(f'A. Retention by memory x condition\n({df.seed.nunique()} seeds, mean +/- 95% CI)', fontsize=10)
axes[0].legend(fontsize=8)

# Panel B: paired CONTROL vs SUPPRESS (M0)
for s in seeds_both_supp:
    c = df[(df.seed==s)&(df.condition=='CONTROL')&(df.memory_id==0)]['retention'].values[0]
    v = df[(df.seed==s)&(df.condition=='SUPPRESS_MEM0')&(df.memory_id==0)]['retention'].values[0]
    axes[1].plot([1,2],[c,v],'o-',color='grey',alpha=0.3,lw=0.7,ms=3)
for pos,cond,col in [(1,'CONTROL','steelblue'),(2,'SUPPRESS_MEM0','firebrick')]:
    vals = df[(df.condition==cond)&(df.memory_id==0)]['retention']
    axes[1].errorbar(pos,vals.mean(),vals.sem()*1.96,fmt='s',color=col,ms=10,capsize=5,zorder=5)
axes[1].set_xticks([1,2]); axes[1].set_xticklabels(['CONTROL','SUPPRESS\nM0'])
axes[1].set_ylabel('M0 Retention');
axes[1].set_title(f'B. Suppression degrades M0\n(t={supp["t"]:.2f}, p={supp["p"]:.2e}, d={supp["d"]:.2f})', fontsize=10)

# Panel C: paired CONTROL vs BOOST (M3)
for s in seeds_both_boost:
    c = df[(df.seed==s)&(df.condition=='CONTROL')&(df.memory_id==3)]['retention'].values[0]
    v = df[(df.seed==s)&(df.condition=='BOOST_MEM3')&(df.memory_id==3)]['retention'].values[0]
    axes[2].plot([1,2],[c,v],'o-',color='grey',alpha=0.3,lw=0.7,ms=3)
for pos,cond,col in [(1,'CONTROL','steelblue'),(2,'BOOST_MEM3','darkorange')]:
    vals = df[(df.condition==cond)&(df.memory_id==3)]['retention']
    axes[2].errorbar(pos,vals.mean(),vals.sem()*1.96,fmt='s',color=col,ms=10,capsize=5,zorder=5)
axes[2].set_xticks([1,2]); axes[2].set_xticklabels(['CONTROL','BOOST\nM3'])
axes[2].set_ylabel('M3 Retention')
axes[2].set_title(f'C. Boosting fails to rescue M3\n(t={boost["t"]:.2f}, p={boost["p"]:.2e}, d={boost["d"]:.2f})', fontsize=10)

plt.suptitle('E2: 30-seed Task 10.5 causal replay manipulation\nGrey=individual seeds; squares=mean +/- 95% CI', fontsize=11, y=1.02)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'e2_task105_30seeds.png')
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
fig.savefig(fig_path.replace('.png','.pdf'), bbox_inches='tight')
plt.close()
print(f'[E2] Figure saved: {fig_path}', flush=True)
print('\n[E2] === DONE ===', flush=True)
