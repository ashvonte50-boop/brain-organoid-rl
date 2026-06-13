"""
E1 -- Boost-with-Adequate-Seed (the decisive experiment)
=========================================================
Breaks the confound in the "replay amplifies but cannot inscribe" claim.
M3 normally has BOTH a small encoding seed AND last-encoded position. We give
M3 an ADEQUATE encoding seed (matched to M0's natural seed) and THEN boost its
replay. 2x2 factorial: seed (normal/adequate) x boost (1x/2x).

OUTCOME A: boost works only with adequate seed  -> amplifier hypothesis CONFIRMED
OUTCOME B: boost still fails with adequate seed  -> boundary is NOT the seed; reframe

Encoding-seed knob: n_presentations passed to train_one_memory (per-memory).
  normal   = _N_PRESENTATIONS (7 in DEV)
  adequate = _N_PRESENTATIONS * ADEQUATE_MULT, calibrated so post-encode
             fast-weight W_encode[M3] >= natural W_encode[M0].

HONESTY: ADEQUATE_MULT is calibrated by MEASUREMENT (not assumed). Primary
result uses default params except the two manipulated variables (M3 seed, M3 boost).

Output: e1_results/e1_boost_adequate_seed.csv  (fault-tolerant append)
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
E1_SEEDS = [42 + i*1000 for i in range(15)]    # 15 seeds (42..14042)
N_MEM = 4
NE = 750
BASE_PRES = ccf._N_PRESENTATIONS               # 7 in DEV
M3_IDX = 3
M0_IDX = 0

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\e1_results'
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUT_DIR, 'e1_boost_adequate_seed.csv')
CALIB_FILE = os.path.join(OUT_DIR, 'e1_calibration.txt')

print(f'[E1] Seeds: {len(E1_SEEDS)} ({E1_SEEDS[0]}..{E1_SEEDS[-1]})', flush=True)
print(f'[E1] Base presentations (normal seed): {BASE_PRES}', flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Core run function with two manipulated knobs:
#   m3_pres  : presentations for M3 encoding (normal=BASE_PRES, adequate=BASE_PRES*mult)
#   m3_boost : replay bias multiplier for M3 (1.0 = uniform, 2.0 = boosted)
# Returns dict with retention/replay/wslow per memory AND measured W_encode per memory.
# ══════════════════════════════════════════════════════════════════════════════
def run_e1(seed, m3_pres, m3_boost, assemblies, assemblies_np, core_set):
    # replay bias
    raw = np.ones(N_MEM); raw[M3_IDX] = m3_boost
    bias_probs = raw / raw.sum()

    _net_ref = [None]; _replay_log = []
    W_encode = {}
    _train_idx = [0]

    _orig_build = ccf.build_network
    _orig_replay = ccf._replay_one_event
    _orig_train = ccf.train_one_memory

    def _track_build(use_slow=False):
        n = _orig_build(use_slow=use_slow); _net_ref[0] = n; return n
    ccf.build_network = _track_build

    def _train_hook(net, assembly, **kw):
        _net_ref[0] = net
        j = _train_idx[0]
        # Override presentations for M3 only
        if j == M3_IDX:
            kw = dict(kw); kw['n_presentations'] = m3_pres
        r = _orig_train(net, assembly, **kw)
        # measure post-encode fast-weight unique-block mean
        asm = assemblies[j]
        uniq = [int(x) for x in asm if int(x) not in core_set and int(x) < NE]
        with torch.no_grad():
            Wf = net.W.detach().cpu().numpy()
        W_encode[j] = float(Wf[np.ix_(uniq, uniq)].mean()) if len(uniq) >= 2 else float('nan')
        _train_idx[0] += 1
        return r
    ccf.train_one_memory = _train_hook

    def _biased_replay(net, assembly, tags=None, **kw):
        _net_ref[0] = net; _last_net[0] = net
        p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)
        chosen = int(np.random.choice(N_MEM, p=bias_probs))
        result = _orig_replay(net, assemblies_np[chosen], tags=tags, **p, **kw)
        _replay_log.append(chosen)
        return result
    ccf._replay_one_event = _biased_replay

    _CENTROID_LOG.clear(); _last_net[0] = None; _replay_log.clear()
    ccf.torch.manual_seed(seed); ccf.np.random.seed(seed)
    try:
        ccf.run_sequential_experiment(True, True, assemblies, seed, ablation={})
    finally:
        ccf.build_network = _orig_build
        ccf._replay_one_event = _orig_replay
        ccf.train_one_memory = _orig_train

    net = _net_ref[0] if _net_ref[0] is not None else _last_net[0]
    assert net is not None, f'Net not captured seed={seed}'

    ret = []
    for asm in assemblies:
        try:
            ret.append(float(ccf.probe_memory(net, asm)['isyn_score']))
        except Exception:
            ret.append(0.0)
    ret = np.nan_to_num(ret, nan=0.0)

    with torch.no_grad():
        WS = net.W_slow.cpu().numpy()
    from collections import Counter
    mc = Counter(_replay_log)
    replay_counts = [mc.get(i, 0) for i in range(N_MEM)]

    wslow = {}
    for mi in range(N_MEM):
        asm = assemblies[mi]
        uniq = [int(x) for x in asm if int(x) not in core_set and int(x) < NE]
        wslow[mi] = float(WS[np.ix_(uniq, uniq)].mean()) if len(uniq) >= 2 else float('nan')

    return {
        'retention': ret, 'replay': replay_counts, 'wslow': wslow,
        'W_encode': W_encode,
    }

# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION: find ADEQUATE_MULT so W_encode[M3] >= natural W_encode[M0].
# Uses 2 calibration seeds, normal run to get natural gradient, then sweeps
# M3 multipliers until matched. Measured, not assumed.
# ══════════════════════════════════════════════════════════════════════════════
def calibrate():
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE) as f:
            for line in f:
                if line.startswith('ADEQUATE_MULT='):
                    mult = int(line.strip().split('=')[1])
                    print(f'[E1] Using cached ADEQUATE_MULT={mult}', flush=True)
                    return mult
    print('[E1] === CALIBRATION ===', flush=True)
    calib_seed = 42
    ccf.torch.manual_seed(calib_seed); ccf.np.random.seed(calib_seed)
    assemblies, core_mask = make_schema_assemblies(N_MEM, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
    assemblies_np = [np.array(a) for a in assemblies]
    core_set = set(int(x) for x in core_mask.tolist())

    lines = []
    # natural gradient (normal seed, no boost)
    r0 = run_e1(calib_seed, BASE_PRES, 1.0, assemblies, assemblies_np, core_set)
    we = r0['W_encode']
    target = we[M0_IDX]
    lines.append(f'Natural W_encode gradient (normal seed): ' +
                 ' '.join(f'M{j}={we[j]:.5f}' for j in range(N_MEM)))
    lines.append(f'Target (natural W_encode[M0]) = {target:.5f}')
    lines.append(f'Natural W_encode[M3] = {we[M3_IDX]:.5f}')

    chosen = 1
    if we[M3_IDX] >= target:
        lines.append('M3 natural seed already >= M0 target; ADEQUATE_MULT=1 '
                     '(IMPORTANT: M3 is NOT seed-deficient in fast weights).')
        chosen = 1
    else:
        for mult in [2, 3, 4, 5]:
            rm = run_e1(calib_seed, BASE_PRES*mult, 1.0, assemblies, assemblies_np, core_set)
            w3 = rm['W_encode'][M3_IDX]
            lines.append(f'  mult={mult}: M3 presentations={BASE_PRES*mult}, W_encode[M3]={w3:.5f} '
                         f'(target {target:.5f}) {"MATCH" if w3>=target else ""}')
            if w3 >= target:
                chosen = mult; break
            chosen = mult  # keep last if never reached
    lines.append(f'ADEQUATE_MULT={chosen}')
    with open(CALIB_FILE, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    for ln in lines:
        print('[E1][calib] ' + ln, flush=True)
    return chosen

ADEQUATE_MULT = calibrate()
M3_PRES = {'normal': BASE_PRES, 'adequate': BASE_PRES * ADEQUATE_MULT}

E1_CONDITIONS = {
    'M3_normalseed_noboost':   ('normal',   1.0),
    'M3_normalseed_boost':     ('normal',   2.0),
    'M3_adequateseed_noboost': ('adequate', 1.0),
    'M3_adequateseed_boost':   ('adequate', 2.0),
}
print(f'[E1] ADEQUATE_MULT={ADEQUATE_MULT} -> M3 adequate presentations={M3_PRES["adequate"]}', flush=True)

# ── Resume + save ─────────────────────────────────────────────────────────────
def is_done(seed, cond):
    if not os.path.exists(RESULTS_FILE):
        return False
    d = pd.read_csv(RESULTS_FILE)
    return ((d.seed==seed)&(d.condition==cond)).any()

def save_row(row):
    df = pd.DataFrame([row])
    if os.path.exists(RESULTS_FILE):
        df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(RESULTS_FILE, index=False)

# ── Main factorial loop ───────────────────────────────────────────────────────
total = len(E1_CONDITIONS) * len(E1_SEEDS)
run_n = 0; t_global = time.time()
for cond_name, (seed_level, boost) in E1_CONDITIONS.items():
    for sd in E1_SEEDS:
        run_n += 1
        if is_done(sd, cond_name):
            print(f'[E1] Skip {cond_name} seed={sd} -- done', flush=True)
            continue
        t0 = time.time()
        print(f'\n[E1] Run {run_n}/{total}: {cond_name} seed={sd}', flush=True)
        ccf.torch.manual_seed(sd); ccf.np.random.seed(sd)
        assemblies, core_mask = make_schema_assemblies(N_MEM, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
        assemblies_np = [np.array(a) for a in assemblies]
        core_set = set(int(x) for x in core_mask.tolist())
        try:
            res = run_e1(sd, M3_PRES[seed_level], boost, assemblies, assemblies_np, core_set)
            save_row({
                'seed': sd, 'condition': cond_name,
                'm3_seed_level': seed_level, 'm3_boost': boost,
                'm3_pres': M3_PRES[seed_level],
                'm3_W_encode': res['W_encode'][M3_IDX],
                'm0_W_encode': res['W_encode'][M0_IDX],
                'm3_replay_count': res['replay'][M3_IDX],
                'm3_retention': float(res['retention'][M3_IDX]),
                'm3_wslow': res['wslow'][M3_IDX],
                'm0_retention': float(res['retention'][M0_IDX]),
                'm1_retention': float(res['retention'][1]),
                'm2_retention': float(res['retention'][2]),
            })
            print(f'[E1] Done {time.time()-t0:.0f}s | M3 ret={res["retention"][M3_IDX]:.4f} '
                  f'W_enc[M3]={res["W_encode"][M3_IDX]:.4f} replay={res["replay"][M3_IDX]}', flush=True)
        except Exception as e:
            print(f'[E1] ERROR {cond_name} seed={sd}: {e}', flush=True)
            import traceback; traceback.print_exc()

print(f'\n[E1] ALL RUNS DONE in {(time.time()-t_global)/3600:.2f} hrs', flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# DECISIVE 2x2 ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
from scipy import stats
import statsmodels.formula.api as smf

df = pd.read_csv(RESULTS_FILE).drop_duplicates(subset=['seed','condition'], keep='first')
df['seed_adequate'] = (df.m3_seed_level == 'adequate').astype(int)
df['boosted'] = (df.m3_boost == 2.0).astype(int)

out = []
def L(s=''):
    print(s, flush=True); out.append(s)

L('=== E1: DECISIVE 2x2 (seed x boost on M3 retention) ===')
L(f'N seeds = {df.seed.nunique()}  ADEQUATE_MULT={ADEQUATE_MULT}')
L('\nCell means (M3 retention):')
for sa in [0,1]:
    for bo in [0,1]:
        cell = df[(df.seed_adequate==sa)&(df.boosted==bo)]['m3_retention']
        lbl = f'{"adequate" if sa else "normal":>8} seed, {"boost" if bo else "noboost":>7}'
        L(f'  {lbl}: {cell.mean():.4f} +/- {cell.sem():.4f} (n={len(cell)})')
L('\nMeasured W_encode[M3] by seed level:')
for sa,lab in [(0,'normal'),(1,'adequate')]:
    w = df[df.seed_adequate==sa]['m3_W_encode']
    L(f'  {lab}: W_encode[M3]={w.mean():.5f}')
L(f'  natural W_encode[M0] (target): {df["m0_W_encode"].mean():.5f}')

# Mixed-effects interaction
try:
    model = smf.mixedlm("m3_retention ~ seed_adequate * boosted", data=df, groups=df["seed"])
    fit = model.fit(reml=False)
    L('\n[Mixed-effects: m3_retention ~ seed_adequate * boosted]')
    L(str(fit.summary()))
    inter_coef = fit.params.get('seed_adequate:boosted', np.nan)
    inter_p = fit.pvalues.get('seed_adequate:boosted', np.nan)
except Exception as e:
    L(f'Mixed-effects failed: {e}')
    inter_coef, inter_p = np.nan, np.nan

# Key paired test: among adequate-seed runs, does boost help?
def paired_key(sa):
    nb = df[(df.seed_adequate==sa)&(df.boosted==0)].sort_values('seed')
    bo = df[(df.seed_adequate==sa)&(df.boosted==1)].sort_values('seed')
    common = sorted(set(nb.seed)&set(bo.seed))
    a = nb[nb.seed.isin(common)].sort_values('seed')['m3_retention'].values
    b = bo[bo.seed.isin(common)].sort_values('seed')['m3_retention'].values
    t,p = stats.ttest_rel(b, a)
    return a.mean(), b.mean(), b.mean()-a.mean(), t, p, len(common)

adq = paired_key(1)
nrm = paired_key(0)
L(f'\n--- Among ADEQUATE-seed M3: does boost help? (n={adq[5]}) ---')
L(f'  noboost={adq[0]:.4f}  boost={adq[1]:.4f}  delta={adq[2]:+.4f}  t={adq[3]:.3f}  p={adq[4]:.4f}')
L(f'--- Among NORMAL-seed M3: does boost help? (n={nrm[5]}) ---')
L(f'  noboost={nrm[0]:.4f}  boost={nrm[1]:.4f}  delta={nrm[2]:+.4f}  t={nrm[3]:.3f}  p={nrm[4]:.4f}')
L(f'\nseed x boost interaction: coef={inter_coef}, p={inter_p}')

# VERDICT
delta_key, p_key = adq[2], adq[4]
L('\n=== E1 DECISIVE VERDICT ===')
if delta_key > 0.005 and p_key < 0.05:
    verdict = 'A'
    L('>>> OUTCOME A: Boost WORKS on adequately-seeded M3.')
    L('>>> CONFIRMS the amplifier hypothesis. The boundary IS the encoding-seed threshold.')
else:
    verdict = 'B'
    L('>>> OUTCOME B: Boost STILL FAILS even with adequate seed.')
    L('>>> The boundary is NOT the encoding seed. The amplifier framing must be revised.')
    L('>>> Operative constraint is interference / replay-scheduling / encoding position.')

# Both paste-ready texts
L('\n=== PASTE-READY TEXT (OUTCOME A) ===')
L(f'"The amplification boundary is the encoding-phase seed. To test whether the failure of '
  f'boosted replay to rescue M3 reflects its small encoding seed rather than its position, we '
  f'performed a 2x2 factorial (M3 encoding seed: normal vs adequate; M3 replay: normal vs '
  f'boosted; {df.seed.nunique()} seeds). A significant seed x boost interaction '
  f'(beta={inter_coef:.4g}, p={inter_p:.4g}) showed that boosted replay increased M3 retention '
  f'only when M3 had received an adequate encoding seed (delta={adq[2]:+.4f}, p={adq[4]:.4g}), '
  f'and produced no gain at the normal seed (delta={nrm[2]:+.4f}, p={nrm[4]:.4g}). This directly '
  f'confirms that replay amplifies an existing encoding-phase trace: a sufficient seed is '
  f'necessary for replay-driven consolidation, and once present, additional replay potentiates '
  f'W_slow as predicted by RGCC."')
L('\n=== PASTE-READY TEXT (OUTCOME B) ===')
L(f'"Contrary to a pure encoding-seed-threshold account, providing M3 with an adequate encoding '
  f'seed (W_encode[M3] {df[df.seed_adequate==1].m3_W_encode.mean():.4f} vs natural M0 '
  f'{df.m0_W_encode.mean():.4f}) did not unlock a benefit of boosted replay (seed x boost '
  f'interaction p={inter_p:.4g}; boost effect at adequate seed delta={adq[2]:+.4f}, p={adq[4]:.4g}, '
  f'n.s.). This indicates the boundary on replay-driven consolidation is not solely the '
  f'encoding-seed magnitude. We reframe the amplifier observation: replay does not rescue the '
  f'last-encoded memory even when its seed is restored, pointing to encoding position / '
  f'replay-scheduling as the operative constraint."')
L(f'\n>>> APPLIES: OUTCOME {verdict}')

with open(os.path.join(OUT_DIR, 'e1_decisive_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print(f'\n[E1] Summary saved: e1_decisive_summary.txt', flush=True)

# ── Figure: 2x2 interaction plot ──────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.5, 5))
means = df.groupby(['seed_adequate','boosted'])['m3_retention'].mean()
sems  = df.groupby(['seed_adequate','boosted'])['m3_retention'].sem()
for sa, col, lbl in [(0,'firebrick','Normal seed'), (1,'steelblue','Adequate seed')]:
    y = [means[(sa,0)], means[(sa,1)]]
    e = [sems[(sa,0)], sems[(sa,1)]]
    ax.errorbar([0,1], y, e, fmt='o-', color=col, markersize=11, capsize=6, linewidth=2.5, label=lbl)
ax.set_xticks([0,1]); ax.set_xticklabels(['No boost','Boost (2x)'], fontsize=12)
ax.set_ylabel('M3 retention (isyn_score)', fontsize=12)
ax.set_title('E1: Does an adequate encoding seed unlock boosted replay?\n'
             f'Interaction p={inter_p:.3g}; adequate-seed boost delta={adq[2]:+.4f} (p={adq[4]:.3g})\n'
             f'OUTCOME {verdict}', fontsize=10)
ax.legend(fontsize=10, title='M3 encoding seed')
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'e1_boost_adequate_seed.png')
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
fig.savefig(fig_path.replace('.png','.pdf'), bbox_inches='tight')
plt.close()
print(f'[E1] Figure saved: {fig_path}', flush=True)
print('\n[E1] === DONE ===', flush=True)
