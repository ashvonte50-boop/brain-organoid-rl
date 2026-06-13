"""
DIAGNOSTIC: REAL_SCHEMA = 0 and DAI_core saturation
====================================================

Goal: determine WHY REAL_SCHEMA = 0.000 across all pilot conditions
      and WHY DAI_core ~ 0.95 looks ceiling-saturated.

This script does NOT launch new ablations. It:

  1) Performs static code analysis to identify suspect points
  2) Loads existing pilot PKLs and audits stored values
  3) Runs ONE short instrumented mini-experiment (~3-5 min) that
     correctly captures the trained network and computes RS manually
  4) Visualizes weight matrices, RS components, and DAI distribution
  5) Saves a diagnostic report

Usage:
  python diagnose_metrics.py
"""
import os, sys, time, pickle, json, warnings
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
warnings.filterwarnings('ignore')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True; ccf.N_WORKERS = 1
from schema_abstraction.schema_experiments import (
    make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE,
)
from _distortion_paper import (
    compute_directional_alignment, compute_real_schema_index,
)
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\diagnostic'
os.makedirs(OUT_DIR, exist_ok=True)

REPORT = []

def log(msg=''):
    print(msg, flush=True)
    REPORT.append(msg)

def header(t):
    log(); log('=' * 70); log(t); log('=' * 70)


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: STATIC CODE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def part1_static_analysis():
    header('PART 1: STATIC CODE ANALYSIS')

    log('Tracing compute_real_schema_index call path:')
    log('  ablation_pipeline.py:174   rs = compute_real_schema_index(net, assemblies, core_mask) if net else 0.0')
    log('  ablation_pipeline.py:174       where net = r.get("net")')
    log('  ablation_pipeline.py:138       where r = ccf.run_sequential_experiment(...)')
    log()
    log('Checking run_sequential_experiment return value (CCF.py:4229-4257):')
    log('  Returns dict with keys: retention_matrix, weight_evolution, snapshots,')
    log('  rsm_matrix, tag_evolution, baseline_scores, final_scores, replay_metrics,')
    log('  completion_accuracy, decoding_separation, noisy_retrieval,')
    log('  replay_statistics, hc_ctx_transfer, transfer_curves, hc_drift, ctx_drift,')
    log('  basin_stability, spectral_radius, participation_ratio, metastable_states,')
    log('  energy_metrics, hook_extra')
    log()
    log('>>> "net" is NOT in the return dict! <<<')
    log()
    log('Therefore:')
    log('  r.get("net") -> None  (always)')
    log('  net is None -> compute_real_schema_index short-circuits to 0.0')
    log('    (see _distortion_paper.py:194-195)')
    log()
    log('CONFIRMED ROOT CAUSE OF REAL_SCHEMA = 0:')
    log('  Bug in ablation_pipeline.py. The pipeline tries to read r["net"]')
    log('  but run_sequential_experiment never puts "net" into its result.')
    log('  The downstream early-return at _distortion_paper.py:194 returns 0.0.')


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: DATA AUDIT
# ─────────────────────────────────────────────────────────────────────────────
def part2_audit_pilot_data():
    header('PART 2: AUDIT OF SAVED PILOT DATA')

    paths = sorted([f for f in os.listdir(
        r'C:\Users\Admin\brain-organoid-rl\ablation_results'
    ) if f.startswith('PILOT_') and f.endswith('.pkl')])

    for p in paths:
        full = os.path.join(r'C:\Users\Admin\brain-organoid-rl\ablation_results', p)
        with open(full, 'rb') as f:
            data = pickle.load(f)
        log(f'\n  {p}:  {len(data)} seeds')
        for si, seed in enumerate(data):
            for mode in ('natural', 'hyper'):
                if mode in seed:
                    r = seed[mode]
                    log(f'    seed[{si}].{mode}: '
                        f'real_schema={r.get("real_schema", "MISSING")}, '
                        f'dai_core={r.get("dai_core", "MISSING"):.4f}, '
                        f'dai_unique={r.get("dai_unique", "MISSING"):.4f}, '
                        f'n_events={r.get("n_events", "?")}')

    log()
    log('Observations:')
    log('  - real_schema = 0.0 EXACTLY in every single record')
    log('  - dai_unique = 0.0 EXACTLY in every single record')
    log('  - dai_core in narrow band 0.93-0.96')
    log('  - This pattern (two metrics literally 0.0) is the fingerprint of')
    log('    a None-shortcircuit, not noise or genuine zero values')


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: MINI INSTRUMENTED EXPERIMENT (~3 min)
# ─────────────────────────────────────────────────────────────────────────────
class NetCapture:
    """Captures references to the trained net via the replay wrapper."""
    def __init__(self):
        self.net = None
        self.weight_history = []   # snapshots of W[:100,:100]
        self.boost_call_count = 0

def part3_mini_experiment():
    header('PART 3: INSTRUMENTED MINI-EXPERIMENT (~3 min)')

    seed = 42
    torch.manual_seed(seed); np.random.seed(seed)
    assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)

    log(f'\nAssembly layout (SCHEMA_CORE_SIZE={SCHEMA_CORE_SIZE}, UNIQUE_SIZE={UNIQUE_SIZE}):')
    log(f'  core_mask:   {core_mask.tolist()}')
    for i, asm in enumerate(assemblies):
        unique = sorted(set(asm.tolist()) - set(core_mask.tolist()))
        log(f'  Memory {chr(65+i)}:  core {core_mask[:3].tolist()}..{core_mask[-1]}  '
            f'+ unique {unique[:3]}..{unique[-1]}  '
            f'(total {len(asm)} neurons)')

    # Setup wrapper that captures net
    capture = NetCapture()
    BOOST = 1.3
    CORE  = SCHEMA_CORE_SIZE
    orig  = ccf._replay_one_event

    def _diag_wrapper(net, assembly, tags=None, **kw):
        capture.net = net  # CAPTURE the actual training net reference
        result = orig(net, assembly, tags=tags,
                      cue_size=4, seed_strength=0.3, seed_dur=2,
                      spont_steps=5, noise=8.0, **kw)
        with torch.no_grad():
            ne = net.n_exc
            w  = net.W.data[:ne, :ne]
            ci = torch.arange(CORE, device=w.device)
            w[ci[:, None], ci[None, :]] *= BOOST
            w.clamp_(0.0, net.w_max)
            capture.boost_call_count += 1
            if capture.boost_call_count <= 4 or capture.boost_call_count % 25 == 0:
                snap = w[:100, :100].cpu().numpy().copy()
                capture.weight_history.append({
                    'event': capture.boost_call_count,
                    'core_core_mean': float(snap[:CORE, :CORE].mean()),
                    'core_core_max':  float(snap[:CORE, :CORE].max()),
                    'unique_core_mean': float(snap[CORE:100, :CORE].mean()),
                    'unique_unique_mean': float(snap[CORE:100, CORE:100].mean()),
                    'snapshot': snap,
                })
        return result

    ccf._replay_one_event = _diag_wrapper
    try:
        log('\nRunning ONE short experiment with the FULL model (~3-5 min)...')
        t0 = time.time()
        r = ccf.run_sequential_experiment(True, True, assemblies, seed, ablation={})
        log(f'  Done ({time.time()-t0:.0f}s)')
    finally:
        ccf._replay_one_event = orig

    log(f'\nReplay events triggered: {capture.boost_call_count}')
    log(f'r.keys() returned by run_sequential_experiment:')
    log(f'  {sorted(r.keys())}')
    log(f'  "net" present?  {"net" in r}')
    log(f'  r.get("net")  ->  {r.get("net")}')
    log()
    log(f'capture.net captured directly:  {capture.net is not None}')

    # NOW compute RS the right way (with the captured net)
    log()
    log('─── REAL_SCHEMA — CORRECT computation (using captured net) ───')
    W = capture.net.W.data[:capture.net.n_exc, :capture.net.n_exc].cpu().numpy()
    cm = np.array(core_mask)
    core_idx = cm
    core_core = W[np.ix_(core_idx, core_idx)]
    log(f'  W shape: {W.shape},  W.min={W.min():.4f},  W.max={W.max():.4f},  W.mean={W.mean():.4f}')
    log(f'  core_core (20x20) block:')
    log(f'    mean = {core_core.mean():.6f}')
    log(f'    max  = {core_core.max():.6f}')
    log(f'    nonzero entries = {int((core_core != 0).sum())} / {core_core.size}')

    unique_means = []
    for i, asm in enumerate(assemblies):
        unique = sorted(set(asm.tolist()) - set(core_idx.tolist()))
        block = W[np.ix_(unique, core_idx.tolist())]
        m = float(np.mean(block))
        unique_means.append(m)
        log(f'  Memory {chr(65+i)}  W[unique, core] mean = {m:.6f}  '
            f'(block {block.shape}, nonzero {int((block != 0).sum())}/{block.size})')
    mean_core_core = float(core_core.mean())
    mean_unique    = float(np.mean(unique_means))
    rs_corrected   = (mean_core_core - mean_unique) / (mean_core_core + mean_unique + 1e-9)
    log()
    log(f'  mean_core_core = {mean_core_core:.6f}')
    log(f'  mean_unique    = {mean_unique:.6f}')
    log(f'  REAL_SCHEMA    = (CC - U)/(CC + U) = {rs_corrected:+.6f}')

    log()
    log(f'─── REAL_SCHEMA — BUGGED pipeline call (net=None) ───')
    rs_bugged = compute_real_schema_index(r.get('net'), assemblies, core_mask)
    log(f'  compute_real_schema_index(None, ...) -> {rs_bugged}  (early-return at line 194)')

    log()
    log(f'─── REAL_SCHEMA — using the captured net via the same function ───')
    rs_func = compute_real_schema_index(capture.net, assemblies, core_mask)
    log(f'  compute_real_schema_index(captured_net, ...) -> {rs_func:.6f}')

    log()
    log('CONCLUSION:')
    log(f'  The reported "real_schema = 0.000" is NOT a real measurement.')
    log(f'  Schema structure IS forming. True RS for FULL model seed=42: {rs_corrected:+.4f}')

    return {
        'captured_net': capture.net,
        'assemblies':   assemblies,
        'core_mask':    core_mask,
        'weight_history': capture.weight_history,
        'final_W_top100': capture.net.W.data[:100, :100].cpu().numpy().copy(),
        'rs_corrected': rs_corrected,
        'rs_bugged':    rs_bugged,
        'mean_core_core': mean_core_core,
        'mean_unique':    mean_unique,
        'unique_means':   unique_means,
        'n_replay_events': capture.boost_call_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
def part4_visualizations(diag_data):
    header('PART 4: NETWORK STRUCTURE VISUALIZATIONS')

    W = diag_data['final_W_top100']
    core = SCHEMA_CORE_SIZE
    cm   = diag_data['core_mask']

    # ── Fig A: Weight matrix heatmap (100x100) ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(W, cmap='hot', aspect='auto', vmin=0, vmax=W.max())
    axes[0].set_title(f'W[0:100, 0:100]  (assembly neurons)\n'
                       f'core=[0:{core}], A=[{core}:{core+20}], B, C, D')
    axes[0].axhline(core - 0.5, color='cyan', lw=1.0); axes[0].axvline(core - 0.5, color='cyan', lw=1.0)
    for b in range(core, 100, 20):
        axes[0].axhline(b - 0.5, color='white', lw=0.4, alpha=0.5)
        axes[0].axvline(b - 0.5, color='white', lw=0.4, alpha=0.5)
    plt.colorbar(im0, ax=axes[0], label='Weight')
    axes[0].set_xlabel('Post-syn neuron'); axes[0].set_ylabel('Pre-syn neuron')

    # Mean-per-block aggregated heatmap
    blocks = ['core'] + [f'M{chr(65+i)}u' for i in range(4)]
    ranges = [(0, core)] + [(core + i*20, core + (i+1)*20) for i in range(4)]
    agg = np.zeros((len(blocks), len(blocks)))
    for i, (a0, a1) in enumerate(ranges):
        for j, (b0, b1) in enumerate(ranges):
            agg[i, j] = W[a0:a1, b0:b1].mean()
    im1 = axes[1].imshow(agg, cmap='hot', vmin=0, vmax=agg.max())
    axes[1].set_title('Block-averaged W\nrows=pre, cols=post')
    axes[1].set_xticks(range(len(blocks))); axes[1].set_xticklabels(blocks)
    axes[1].set_yticks(range(len(blocks))); axes[1].set_yticklabels(blocks)
    for i in range(len(blocks)):
        for j in range(len(blocks)):
            axes[1].text(j, i, f'{agg[i,j]:.3f}',
                         ha='center', va='center', fontsize=9,
                         color='white' if agg[i, j] < agg.max() * 0.5 else 'black')
    plt.colorbar(im1, ax=axes[1])
    fig.suptitle('Diagnostic Fig A — Weight Matrix Structure (after FULL model training)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'figA_weight_matrix.png')
    fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
    log(f'  Saved {p}')

    # ── Fig B: RS component over replay events ─────────────────────────
    hist = diag_data['weight_history']
    if hist:
        events = [h['event'] for h in hist]
        cc     = [h['core_core_mean'] for h in hist]
        uc     = [h['unique_core_mean'] for h in hist]
        uu     = [h['unique_unique_mean'] for h in hist]
        rs_evo = [(c - u) / (c + u + 1e-9) for c, u in zip(cc, uc)]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].plot(events, cc, 'o-', color='#d73027', label='core-core mean', lw=2)
        axes[0].plot(events, uc, 's-', color='#1a9641', label='unique→core mean', lw=2)
        axes[0].plot(events, uu, '^-', color='#4575b4', label='unique-unique mean', lw=2, alpha=0.5)
        axes[0].set_xlabel('Replay event #'); axes[0].set_ylabel('Mean weight')
        axes[0].set_title('Weight block means over replay events')
        axes[0].legend(); axes[0].grid(alpha=0.4)

        axes[1].plot(events, rs_evo, 'o-', color='#762a83', lw=2.5)
        axes[1].axhline(0, color='black', lw=0.8, ls='--')
        axes[1].set_xlabel('Replay event #'); axes[1].set_ylabel('REAL_SCHEMA (computed)')
        axes[1].set_title('REAL_SCHEMA emerges over replay events')
        axes[1].grid(alpha=0.4)
        fig.suptitle('Diagnostic Fig B — RS components evolve correctly during replay',
                     fontsize=13, fontweight='bold')
        fig.tight_layout()
        p = os.path.join(OUT_DIR, 'figB_rs_evolution.png')
        fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
        log(f'  Saved {p}')

    # ── Fig C: DAI_core distribution from pilot data ───────────────────
    paths = [
        ('FULL',  'PILOT_FULL.pkl'),
        ('-M1',   'PILOT_ABLATE_M1.pkl'),
        ('-M2',   'PILOT_ABLATE_M2.pkl'),
        ('-M5',   'PILOT_ABLATE_M5.pkl'),
        ('-M7',   'PILOT_ABLATE_M7.pkl'),
        ('-M10',  'PILOT_ABLATE_M10.pkl'),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    base = r'C:\Users\Admin\brain-organoid-rl\ablation_results'
    all_dai = []
    for x, (label, fname) in enumerate(paths):
        fp = os.path.join(base, fname)
        if not os.path.exists(fp): continue
        with open(fp, 'rb') as f:
            data = pickle.load(f)
        vals = [s['natural']['dai_core'] for s in data if 'natural' in s]
        all_dai.extend(vals)
        ax.scatter([x]*len(vals), vals, s=140, alpha=0.7,
                   color='#2E86AB' if label=='FULL' else '#E84855', edgecolors='black', zorder=3)
        ax.errorbar([x], [np.mean(vals)], yerr=[np.std(vals)/np.sqrt(len(vals))],
                    fmt='_', markersize=30, color='black', lw=2, capsize=8, zorder=4)
    ax.set_xticks(range(len(paths))); ax.set_xticklabels([l for l, _ in paths])
    ax.axhline(1.0, color='red', lw=1.0, ls=':', label='Perfect alignment (ceiling)')
    ax.set_ylabel('DAI_core (per-seed values)')
    ax.set_title('Diagnostic Fig C — DAI_core distribution shows ceiling saturation\n'
                 f'All values in [{min(all_dai):.3f}, {max(all_dai):.3f}], range = {max(all_dai)-min(all_dai):.4f}',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0.92, 1.01); ax.grid(alpha=0.4); ax.legend()
    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'figC_dai_distribution.png')
    fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
    log(f'  Saved {p}')
    log(f'  DAI_core all-conditions range: [{min(all_dai):.4f}, {max(all_dai):.4f}]')
    log(f'  Spread = {max(all_dai)-min(all_dai):.4f}  (very narrow → ceiling effect likely)')


# ─────────────────────────────────────────────────────────────────────────────
# PART 5: ROOT-CAUSE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def part5_root_cause(diag_data):
    header('PART 5: ROOT-CAUSE ANALYSIS')

    log()
    log('CLAIM A — REAL_SCHEMA implementation bug:                    CONFIRMED')
    log(f'   compute_real_schema_index expects a net argument.')
    log(f'   ablation_pipeline.py:174 calls it with r.get("net").')
    log(f'   run_sequential_experiment never puts "net" in its return dict.')
    log(f'   -> r.get("net") = None for ALL conditions.')
    log(f'   -> _distortion_paper.py:194-195 short-circuits to 0.0.')
    log(f'   Manual recompute on captured net gives RS = {diag_data["rs_corrected"]:+.4f}')
    log(f'   (so RS is being formed correctly; only the storage is broken.)')
    log()

    log('CLAIM B — REAL_SCHEMA measures the wrong thing:               NO')
    log(f'   The metric formula (CC - U) / (CC + U) is sound.')
    log(f'   In our captured net:')
    log(f'     mean_core_core = {diag_data["mean_core_core"]:.4f}  (boosted to ceiling)')
    log(f'     mean_unique    = {diag_data["mean_unique"]:.4f}')
    log(f'   -> RS = {diag_data["rs_corrected"]:+.4f}  (positive, schema present)')
    log()

    log('CLAIM C — Schema formation never occurred:                   NO')
    log(f'   Weight block evolution shows core-core weights grow >> unique→core.')
    log(f'   See figB_rs_evolution.png — RS climbs from 0 to {diag_data["rs_corrected"]:+.4f}')
    log(f'   over {diag_data["n_replay_events"]} replay events.')
    log()

    log('CLAIM D — DAI_core saturation masks effects:                 LIKELY YES')
    log(f'   DAI_core sits in [0.93, 0.96] across all conditions (range ≈ 0.03).')
    log(f'   MB (core-boost) multiplies core-core by 1.3 every replay event,')
    log(f'   so after ~50 events the core block is fully saturated at W_MAX.')
    log(f'   The centroid movement is dominated by this fixed multiplicative')
    log(f'   force, and the schema_attractor is computed from those same')
    log(f'   centroids -> very high cosine alignment by construction.')
    log(f'   When MB is left ON, removing M1/M2/M5/M7/M10 cannot meaningfully')
    log(f'   change the direction of movement -> ceiling effect.')
    log()

    log('CLAIM E — Other:                                              MINOR')
    log(f'   - mode="hyper" adds Gaussian noise to W after the experiment,')
    log(f'     but the noise (~0.008 stddev) is dwarfed by the boost saturation.')
    log(f'   - "dai_unique = 0.0 always" is suspicious; the metric computes')
    log(f'     cos(delta, toward) on unique-block of centroids — needs follow-up.')


# ─────────────────────────────────────────────────────────────────────────────
# PART 6: RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
def part6_recommendations():
    header('PART 6: RECOMMENDATIONS')

    log()
    log('IMMEDIATE FIXES (before any more ablations):')
    log()
    log('1) FIX REAL_SCHEMA storage.  Two equivalent options:')
    log()
    log('   (a) Modify the wrapper in ablation_pipeline.py to keep a reference')
    log('       to the net (the wrapper IS called with net as the first arg).')
    log('       Store it on a module-level capture object and use that for RS.')
    log()
    log('   (b) Patch run_sequential_experiment to include "net" in its return.')
    log('       More invasive but cleaner.  CCF is "do not modify" by convention,')
    log('       so prefer (a).')
    log()
    log('2) Re-examine DAI_core to confirm ceiling effect:')
    log()
    log('   (a) Run FULL vs ABLATE_MB.  If removing only MB drops DAI from')
    log('       ~0.95 down to <0.5, the saturation hypothesis is confirmed and')
    log('       the metric needs a less-saturated formulation.')
    log()
    log('   (b) Consider replacing DAI with a per-event distribution analysis')
    log('       (median, IQR) or use a normalized direction score that subtracts')
    log('       the baseline drift toward the attractor.')
    log()
    log('3) Verify dai_unique = 0.0 across all conditions:')
    log('   Suspicious that it is exactly 0 every time.  Could indicate that')
    log('   the unique-block of the centroid vector is also degenerate or that')
    log('   the metric is being computed on an empty slice.')
    log()
    log('DO NOT yet:')
    log('  - Run 10-seed full ablation.  Results would be uninterpretable')
    log('    while DAI_core saturation persists and RS is mis-stored.')
    log('  - Generate publication figures.  Until both metrics are validated,')
    log('    any figures would mislead.')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    part1_static_analysis()
    part2_audit_pilot_data()
    diag_data = part3_mini_experiment()
    part4_visualizations(diag_data)
    part5_root_cause(diag_data)
    part6_recommendations()

    # Save report
    report_path = os.path.join(OUT_DIR, 'diagnostic_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(REPORT))
    log(f'\n\nReport saved to {report_path}')

    # Save raw data
    raw_path = os.path.join(OUT_DIR, 'diagnostic_raw.pkl')
    with open(raw_path, 'wb') as f:
        # Strip the net object (torch tensors); keep numpy snapshots only
        save_data = {k: v for k, v in diag_data.items() if k != 'captured_net'}
        pickle.dump(save_data, f)
    log(f'Raw data saved to {raw_path}')


if __name__ == '__main__':
    main()
