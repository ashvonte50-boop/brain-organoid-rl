"""
DIAGNOSTIC: Why is dai_unique always 0?

Short script — no new long experiments.
Reuses the captured net from the already-finished diagnostic run.
If that data exists, loads it. Otherwise runs a short 1-seed capture.

Findings expected:
  - Print per-event norm(du) vs norm(dc)
  - Explain why the guard (dn > 1e-12) triggers for unique but not core
  - Categorise as: storage bug / indexing bug / genuine network behaviour
"""
import os, sys, pickle, warnings
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results\diagnostic'
RAW_PATH = os.path.join(OUT_DIR, 'diagnostic_raw.pkl')
PILOT_PATH = r'C:\Users\Admin\brain-organoid-rl\ablation_results\PILOT_FULL.pkl'

os.makedirs(OUT_DIR, exist_ok=True)


def load_centroid_log():
    """Re-run a mini experiment to get the centroid log with full vectors."""
    import torch
    import compare_catastrophic_forgetting as ccf
    ccf.DEV_MODE = True; ccf.N_WORKERS = 1
    from schema_abstraction.schema_experiments import make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE
    import schema_abstraction.schema_core as sc
    sc.register_schema_hooks()

    seed = 42
    torch.manual_seed(seed); np.random.seed(seed)
    assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
    CORE = SCHEMA_CORE_SIZE

    log = []
    _net_ref = [None]

    orig = ccf._replay_one_event
    def _wrapper(net, assembly, tags=None, **kw):
        _net_ref[0] = net
        cb_cents = {}
        with torch.no_grad():
            ne = net.n_exc
            w = net.W.data[:ne, :ne].cpu().numpy()
            for i, asm in enumerate(assemblies):
                valid = [int(x) for x in asm if 0 <= int(x) < ne]
                if valid:
                    cb_cents[i] = w[np.ix_(valid, valid)].mean(axis=1).copy()

        result = orig(net, assembly, tags=tags,
                      cue_size=4, seed_strength=0.3, seed_dur=2,
                      spont_steps=5, noise=8.0, **kw)

        with torch.no_grad():
            ne = net.n_exc
            w = net.W.data[:ne, :ne]
            ci = torch.arange(CORE, device=w.device)
            w[ci[:, None], ci[None, :]] *= 1.3
            w.clamp_(0.0, net.w_max)

            ca_cents = {}
            wn = w.cpu().numpy()
            for i, asm in enumerate(assemblies):
                valid = [int(x) for x in asm if 0 <= int(x) < ne]
                if valid:
                    ca_cents[i] = wn[np.ix_(valid, valid)].mean(axis=1).copy()

        log.append({
            'memory_idx': kw.get('assembly_idx', -1),
            'cb': cb_cents,
            'ca': ca_cents,
        })
        return result

    ccf._replay_one_event = _wrapper
    try:
        print('Running short experiment to collect centroid vectors...', flush=True)
        ccf.run_sequential_experiment(True, True, assemblies, seed, ablation={})
    finally:
        ccf._replay_one_event = orig

    return log, assemblies, core_mask, CORE


def analyse(log, assemblies, core_mask, CORE):
    print(f'\nTotal replay events captured: {len(log)}', flush=True)

    # Build schema attractor from latest centroids (same method as compute_directional_alignment)
    latest = {}
    for e in log:
        for k, v in e['ca'].items():
            latest[k] = v
    schema_attractor = np.mean(list(latest.values()), axis=0)

    print(f'Schema attractor shape: {schema_attractor.shape}', flush=True)
    print(f'  attractor[:CORE]  norm = {np.linalg.norm(schema_attractor[:CORE]):.6f}')
    print(f'  attractor[CORE:]  norm = {np.linalg.norm(schema_attractor[CORE:]):.6f}')

    # Per-event analysis
    norms_dc, norms_du, norms_tc, norms_tu = [], [], [], []
    cos_core_vals, cos_uniq_vals = [], []
    triggered_core, triggered_uniq = 0, 0
    skipped_core, skipped_uniq = 0, 0

    for e in log:
        mid = e['memory_idx']
        if mid < 0 or mid not in e['cb'] or mid not in e['ca']:
            continue
        before = e['cb'][mid]
        after  = e['ca'][mid]
        if before.shape[0] <= CORE:
            continue

        delta  = after - before
        toward = schema_attractor - before

        dc, tc = delta[:CORE],  toward[:CORE]
        du, tu = delta[CORE:],  toward[CORE:]

        nd_c, nt_c = np.linalg.norm(dc), np.linalg.norm(tc)
        nd_u, nt_u = np.linalg.norm(du), np.linalg.norm(tu)

        norms_dc.append(nd_c); norms_du.append(nd_u)
        norms_tc.append(nt_c); norms_tu.append(nt_u)

        if nd_c > 1e-12 and nt_c > 1e-12:
            cos_core_vals.append(np.dot(dc, tc) / (nd_c * nt_c))
            triggered_core += 1
        else:
            skipped_core += 1

        if nd_u > 1e-12 and nt_u > 1e-12:
            cos_uniq_vals.append(np.dot(du, tu) / (nd_u * nt_u))
            triggered_uniq += 1
        else:
            skipped_uniq += 1

    print(f'\n--- Per-event norm statistics ---')
    print(f'  norm(dc) [core delta]:   mean={np.mean(norms_dc):.6f}  min={np.min(norms_dc):.2e}  max={np.max(norms_dc):.6f}')
    print(f'  norm(du) [uniq delta]:   mean={np.mean(norms_du):.6f}  min={np.min(norms_du):.2e}  max={np.max(norms_du):.6f}')
    print(f'  norm(tc) [core toward]:  mean={np.mean(norms_tc):.6f}  min={np.min(norms_tc):.2e}  max={np.max(norms_tc):.6f}')
    print(f'  norm(tu) [uniq toward]:  mean={np.mean(norms_tu):.6f}  min={np.min(norms_tu):.2e}  max={np.max(norms_tu):.6f}')

    print(f'\n--- Guard clause (dn > 1e-12) ---')
    print(f'  Core: triggered {triggered_core}/{len(log)}, skipped {skipped_core}')
    print(f'  Uniq: triggered {triggered_uniq}/{len(log)}, skipped {skipped_uniq}')

    print(f'\n--- Mean cosine (events that passed guard) ---')
    print(f'  dai_core   = {np.mean(cos_core_vals):.6f}  (n={len(cos_core_vals)})')
    print(f'  dai_unique = {np.mean(cos_uniq_vals):.6f}  (n={len(cos_uniq_vals)})')

    # Root cause analysis
    print(f'\n--- ROOT CAUSE ---')
    ratio = np.mean(norms_du) / (np.mean(norms_dc) + 1e-12)
    print(f'  mean norm(unique delta) / mean norm(core delta) = {ratio:.4f}')

    if np.mean(norms_du) < 1e-12:
        print('  RESULT: norm(du) < 1e-12 for ALL events -> guard always skips unique')
        print('  CAUSE:  MB boost only multiplies core->core block (W[0:20,0:20]).')
        print('          Unique neuron outgoing weights (rows 20:99) are unaffected.')
        print('          So unique centroid = mean of W[unique, assembly] barely changes.')
        print('  CLASSIFICATION: GENUINE network behaviour, not a bug.')
        conclusion = 'genuine'
    elif triggered_uniq < triggered_core * 0.5:
        print(f'  RESULT: unique guard fires only {triggered_uniq}/{triggered_core} times vs core')
        print('  CAUSE:  Unique delta is very small (often < 1e-12) because the MB')
        print('          boost does not affect unique rows. When M1-M10 effects are weak,')
        print('          unique centroid barely moves.')
        print('  CLASSIFICATION: GENUINE network behaviour — unique neurons contribute')
        print('          very little directionality under the current parameter regime.')
        conclusion = 'genuine_weak'
    else:
        print('  RESULT: norm(du) passes guard but cos_uniq near 0')
        print('  CAUSE:  schema_attractor unique component is the mean of DIFFERENT')
        print('          memory unique centroids. These point in different directions,')
        print('          so the mean cancels out -> toward[CORE:] near zero.')
        print('  CLASSIFICATION: METRIC DESIGN ISSUE — the schema attractor is not')
        print('          a meaningful target for unique-neuron drift.')
        conclusion = 'attractor_cancellation'

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0,0].semilogy(norms_dc, '.', color='#2166ac', alpha=0.7, label='core delta')
    axes[0,0].semilogy(norms_du, '.', color='#d73027', alpha=0.7, label='unique delta')
    axes[0,0].axhline(1e-12, color='black', lw=1, ls='--', label='guard threshold 1e-12')
    axes[0,0].set_title('norm(delta) per replay event  [log scale]')
    axes[0,0].set_xlabel('Replay event'); axes[0,0].set_ylabel('L2 norm of delta')
    axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    axes[0,1].semilogy(norms_tc, '.', color='#2166ac', alpha=0.7, label='core toward')
    axes[0,1].semilogy(norms_tu, '.', color='#d73027', alpha=0.7, label='unique toward')
    axes[0,1].axhline(1e-12, color='black', lw=1, ls='--')
    axes[0,1].set_title('norm(toward) per replay event  [log scale]')
    axes[0,1].set_xlabel('Replay event'); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

    axes[1,0].hist(norms_dc, bins=20, alpha=0.7, color='#2166ac', label='core delta')
    axes[1,0].hist(norms_du, bins=20, alpha=0.7, color='#d73027', label='unique delta')
    axes[1,0].set_title('Distribution of delta norms')
    axes[1,0].set_xlabel('L2 norm'); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

    if cos_core_vals or cos_uniq_vals:
        axes[1,1].hist(cos_core_vals, bins=15, alpha=0.7, color='#2166ac', label=f'cos_core n={len(cos_core_vals)}')
        axes[1,1].hist(cos_uniq_vals, bins=15, alpha=0.7, color='#d73027', label=f'cos_uniq n={len(cos_uniq_vals)}')
        axes[1,1].axvline(0, color='black', lw=1, ls='--')
        axes[1,1].set_title('Cosine distribution (events passing guard)')
        axes[1,1].set_xlabel('Cosine similarity'); axes[1,1].legend(); axes[1,1].grid(alpha=0.3)
    else:
        axes[1,1].text(0.5, 0.5, 'No events passed guard for unique', ha='center', va='center')

    fig.suptitle(f'DAI_unique Diagnostic — classification: {conclusion}', fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'figD_dai_unique_diagnostic.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure saved: {out}')

    return conclusion


if __name__ == '__main__':
    log, assemblies, core_mask, CORE = load_centroid_log()
    conclusion = analyse(log, assemblies, core_mask, CORE)
    print(f'\nFinal classification: {conclusion}', flush=True)
