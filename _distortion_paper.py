"""
REPLAY DISTORTION & SCHEMA ABSTRACTION

Three core conditions:
  NO_REPLAY — passive decay, no replay events
  NATURAL   — biological replay (partial cues, default noise)
  HYPER     — distorted replay (minimal cue, high noise, short spont)

Hypothesis: Replay distortion drives directional schema abstraction.
"""
import os, sys, time, pickle, warnings
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np
import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True; ccf.N_WORKERS = 1
import torch
from schema_abstraction.schema_core import register_schema_hooks
from schema_abstraction.schema_experiments import (
    make_schema_assemblies, _attach_schema_data,
    SCHEMA_CORE_SIZE, UNIQUE_SIZE,
)
from schema_abstraction.schema_novel_metrics import compute_all_novel_metrics
import schema_abstraction.schema_core as sc
import schema_analysis as sa
warnings.filterwarnings('ignore')

# ===== BUG FIXES =====

# 1. Probe memory NaN/Inf sanitization + voltage clamp during probe
_orig_probe = ccf.probe_memory
def _safe_probe(net, assembly):
    try:
        if hasattr(net, 'v') and net.v is not None:
            net.v.data.clamp_(-200.0, 200.0)
        if hasattr(net, 'u') and net.u is not None:
            net.u.data.clamp_(-100.0, 100.0)
        result = _orig_probe(net, assembly)
        if isinstance(result, dict):
            for k in list(result.keys()):
                v = result[k]
                if isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)):
                    result[k] = 0.0
    except Exception:
        result = {"isyn_score": 0.0, "spike_score": 0.0}
    return result
ccf.probe_memory = _safe_probe

N_SEEDS = 5; BASE_SEED = 42
DEVICE, N_NEURONS, N_EXC = ccf.DEVICE, ccf.N_NEURONS, ccf.N_EXC

# -- Track last net for forward transfer ------------------------------

_last_net = None
_orig_build = ccf.build_network
def _track_build(use_slow=False):
    global _last_net; _last_net = _orig_build(use_slow=use_slow); return _last_net
ccf.build_network = _track_build

# -- Replay wrappers --------------------------------------------------

_ORIG_REPLAY = None
_HYPER_NOISE_STD = 0.008
_CORE_INDICES = None  # set per seed from core_mask
_CENTROID_LOG = []    # populated by wrapper during replay events

def _core_boost(net):
    """Boost core-to-core weights by 30% to drive schema abstraction."""
    global _CORE_INDICES
    if _CORE_INDICES is None or len(_CORE_INDICES) == 0:
        return
    with torch.no_grad():
        w = net.W.data[:net.n_exc, :net.n_exc]
        ci = _CORE_INDICES
        ci = ci[ci < w.shape[0]]
        if len(ci) == 0:
            return
        ci_t = torch.as_tensor(ci, device=w.device)
        w[ci_t[:, None], ci_t[None, :]] *= 1.3
        w.clamp_(0.0, 5.0)

def _extract_centroids(net, assemblies):
    """Extract centroid dict {mem_idx: np.array} from network weights."""
    with torch.no_grad():
        ne = net.n_exc
        w = net.W.data[:ne, :ne].cpu().numpy()
        cents = {}
        for i, asm in enumerate(assemblies):
            valid = [int(x) for x in asm if 0 <= int(x) < ne]
            if len(valid) > 0:
                cents[i] = w[np.ix_(valid, valid)].mean(axis=1)
        return cents

def _make_wrapper(mode, assemblies):
    global _ORIG_REPLAY
    _ORIG_REPLAY = ccf._replay_one_event
    p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)
    def _wrapper(net, assembly, tags=None, **kw):
        # Capture centroids BEFORE replay
        cb = _extract_centroids(net, assemblies)
        result = _ORIG_REPLAY(net, assembly, tags=tags, **p, **kw)
        with torch.no_grad():
            ne = net.n_exc
            w = net.W.data[:ne, :ne]
            if mode == 'natural':
                _core_boost(net)
            elif mode == 'hyper':
                _core_boost(net)
                noise = torch.randn_like(w) * _HYPER_NOISE_STD
                w.add_(noise)
                w.clamp_(0.0, net.w_max)
            # Capture centroids AFTER replay + modifications
            ca = _extract_centroids(net, assemblies)
            replay_id = kw.get('burst_id', 0) * 1000 + kw.get('event_id', 0)
            _CENTROID_LOG.append({
                'replay_id': replay_id,
                'memory_idx': kw.get('assembly_idx', -1),
                'centroid_before': {k: v.tolist() for k, v in cb.items()},
                'centroid_after': {k: v.tolist() for k, v in ca.items()},
            })
        return result
    return _wrapper

def install_mode(mode, assemblies):
    old = ccf._replay_one_event
    ccf._replay_one_event = _make_wrapper(mode, assemblies)
    return old

def restore_replay(old):
    global _ORIG_REPLAY
    ccf._replay_one_event = old

# -- Forward transfer -------------------------------------------------

def make_extra_assemblies(assemblies, core_mask):
    used = set(np.concatenate(assemblies).tolist())
    avail = list(set(range(N_EXC)) - used)
    core = core_mask.tolist() if hasattr(core_mask,'tolist') else list(core_mask)
    ue = np.random.choice(avail, size=min(UNIQUE_SIZE, len(avail)), replace=False)
    me = np.concatenate([core, ue]).astype(np.int64)
    used.update(me.tolist())
    avail = list(set(range(N_EXC)) - used)
    uf = np.random.choice(avail, size=min(UNIQUE_SIZE+len(core), len(avail)), replace=False)
    mf = uf.astype(np.int64)
    return me, mf

def run_forward(mode, assemblies, core_mask):
    global _last_net
    net = _last_net
    if net is None: return None
    me, mf = make_extra_assemblies(assemblies, core_mask)
    old = install_mode(mode, assemblies)
    Ec, Fc = [], []
    try:
        for _ in range(5):
            ccf.train_one_memory(net, me, tags=None, n_presentations=1, prev_assembly=None)
            Ec.append(float(ccf.probe_memory(net, me)["isyn_score"]))
        for _ in range(5):
            ccf.train_one_memory(net, mf, tags=None, n_presentations=1, prev_assembly=None)
            Fc.append(float(ccf.probe_memory(net, mf)["isyn_score"]))
    except Exception as e:
        print(f"    Forward error: {e}", flush=True)
        Ec, Fc = [], []
    finally:
        restore_replay(old)
    return {"E": Ec, "F": Fc}

# -- Metrics ----------------------------------------------------------

def compute_metrics(r, assemblies, core_mask):
    m = {}
    try:
        from schema_abstraction.schema_novel_metrics import compute_all_novel_metrics
        m["novel"] = compute_all_novel_metrics(r, assemblies, core_mask)
    except Exception as e:
        m["novel"] = {"error": str(e)}
    ds = r.get("downscale_summary") or {}
    m["nrem"] = ds.get("nrem_count",0); m["rem"] = ds.get("rem_count",0)
    m["events"] = ds.get("downscale_events",0); m["tags"] = ds.get("n_tagged",0)
    m["cumulative_downscale"] = ds.get("cumulative_downscale",0.0)
    gen = r.get("generalization") or {}
    m["generalization"] = 0.0 if isinstance(gen.get("error"),str) else gen.get("0",{}).get("generalization_score",0.0)
    antic = r.get("anti_prediction") or {}
    if not isinstance(antic.get("error"),str):
        m["anti_nat"] = antic.get("natural_generalization",0.0)
        m["anti_hf"] = antic.get("high_fidelity_generalization",0.0)
    else: m["anti_nat"] = m["anti_hf"] = 0.0
    return m

# -- Functional Schema -----------------------------------------------

def compute_real_schema_index(net, assemblies, core_mask):
    """Schema strength measured directly from weights: core-core vs core-unique."""
    if net is None or not hasattr(net, 'W'):
        return 0.0
    W = net.W.data[:net.n_exc, :net.n_exc].cpu().numpy()
    # core_mask may be indices or boolean; handle both
    cm = np.array(core_mask)
    if cm.dtype == bool or cm.dtype == np.bool_:
        core_idx = np.where(cm)[0]
    else:
        core_idx = cm  # already indices
    if len(core_idx) == 0:
        return 0.0
    core_core = W[np.ix_(core_idx, core_idx)]
    mean_core_core = np.mean(core_core)
    unique_means = []
    for asm in assemblies:
        unique = [i for i in asm if i not in core_idx and i < W.shape[0]]
        if len(unique) > 0:
            unique_means.append(np.mean(W[np.ix_(unique, core_idx)]))
    mean_unique = np.mean(unique_means) if unique_means else 1e-9
    schema = (mean_core_core - mean_unique) / (mean_core_core + mean_unique + 1e-9)
    return float(schema)

def measure_functional_schema(net, assemblies, core_mask, n_trials=3):
    if net is None or not hasattr(net, 'spikes'): return 0.0
    cm = np.array(core_mask)
    core_idx = cm if not (cm.dtype == bool or cm.dtype == np.bool_) else np.where(cm)[0]
    if len(core_idx) == 0: return 0.0
    n_neurons = net.v.shape[0] if hasattr(net, 'v') else net.spikes.shape[0]
    orig_noise = net.noise_std
    counts = []
    for _ in range(n_trials):
        net.reset_state()
        net.noise_std = 2.0  # moderate noise for probing
        stim = torch.zeros(n_neurons)
        stim[core_idx] = 3.0
        all_spikes = np.zeros(n_neurons)
        for step in range(100):
            inp = stim if step < 30 else torch.zeros(n_neurons)
            net.forward(inp)
            all_spikes += net.spikes.detach().cpu().numpy()
        net.noise_std = orig_noise
        active = sum(1 for asm in assemblies if len(asm) > 0 and
                     float(all_spikes[np.array(list(asm), dtype=int)].mean()) > 0.3)
        counts.append(active)
    return float(np.mean(counts))


# ── Directional Abstraction Metric ─────────────────────────────

def compute_directional_alignment(centroid_log, n_mem=4, core_size=20):
    """For each replay event, measure whether centroid moves toward schema.

    Schema centroid = mean of EACH memory's LATEST centroid_after across all events.
    Directionality = cos(movement, toward_schema) where:
      movement = centroid_after - centroid_before
      toward_schema = schema_centroid - centroid_before

    Also computes REAL_SCHEMA change per event (delta in core-vs-unique ratio).
    """
    if not centroid_log:
        return {'per_event': [], 'mean_core': 0.0, 'mean_unique': 0.0, 'p_core': 1.0, 'p_unique': 1.0, 'mean_rs_delta': 0.0, 'p_rs': 1.0, 'n_events': 0}

    # Compute schema attractor = mean of each memory's LATEST centroid_after
    # (use latest rather than last event, which may only cover one memory)
    latest_centroids = {}
    for e in centroid_log:
        for mem_k, v in e.get('centroid_after', {}).items():
            latest_centroids[mem_k] = np.array(v)
    if not latest_centroids:
        return {'per_event': [], 'mean_core': 0.0, 'mean_unique': 0.0, 'p_core': 1.0, 'p_unique': 1.0, 'mean_rs_delta': 0.0, 'p_rs': 1.0, 'n_events': 0}
    schema_attractor = np.mean(list(latest_centroids.values()), axis=0)

    core_vals = []
    unique_vals = []
    rs_deltas = []

    for e in centroid_log:
        cb = e.get('centroid_before', {})
        ca = e.get('centroid_after', {})
        mem_idx = e.get('memory_idx', -1)
        if mem_idx < 0 or mem_idx not in cb or mem_idx not in ca:
            continue
        before = np.array(cb[mem_idx])
        after = np.array(ca[mem_idx])
        if before.shape[0] <= core_size:
            continue

        delta = after - before
        toward = schema_attractor - before

        # Core component
        dc = delta[:core_size]
        tc = toward[:core_size]
        dn = np.linalg.norm(dc)
        tn = np.linalg.norm(tc)
        cos_core = np.dot(dc, tc) / (dn * tn) if dn > 1e-12 and tn > 1e-12 else 0.0

        # Unique component
        du = delta[core_size:]
        tu = toward[core_size:]
        dn = np.linalg.norm(du)
        tn = np.linalg.norm(tu)
        cos_uniq = np.dot(du, tu) / (dn * tn) if dn > 1e-12 and tn > 1e-12 else 0.0

        core_vals.append(float(cos_core))
        unique_vals.append(float(cos_uniq))

        # REAL_SCHEMA change per event
        rs_before = _compute_rs_from_centroids(cb, core_size)
        rs_after = _compute_rs_from_centroids(ca, core_size)
        rs_deltas.append(rs_after - rs_before)

    if not core_vals:
        return {'per_event': [], 'mean_core': 0.0, 'mean_unique': 0.0, 'p_core': 1.0, 'p_unique': 1.0, 'n_events': 0}

    from scipy import stats as _st
    t_core, p_core = _st.ttest_1samp(core_vals, 0.0)
    t_uniq, p_uniq = _st.ttest_1samp(unique_vals, 0.0)
    t_rs, p_rs = _st.ttest_1samp(rs_deltas, 0.0) if rs_deltas else (0.0, 1.0)

    events = [{'cos_core': c, 'cos_unique': u} for c, u in zip(core_vals, unique_vals)]

    return {
        'per_event': events,
        'mean_core': float(np.mean(core_vals)),
        'mean_unique': float(np.mean(unique_vals)),
        'mean_rs_delta': float(np.mean(rs_deltas)),
        'p_core': float(p_core),
        'p_unique': float(p_uniq),
        'p_rs': float(p_rs),
        'n_events': len(events),
    }


def _compute_rs_from_centroids(centroids, core_size=20):
    """Compute schema ratio (core vs unique strength) from centroid dict.
    Higher = core dominates unique = more schema-like.
    """
    if not centroids:
        return 0.0
    core_means = []
    unique_means = []
    for k, v in centroids.items():
        arr = np.array(v)
        if arr.shape[0] > core_size:
            core_means.append(float(np.mean(arr[:core_size])))
            unique_means.append(float(np.mean(arr[core_size:])))
    if not core_means:
        return 0.0
    mc = np.mean(core_means)
    mu = np.mean(unique_means)
    if mu <= 0:
        return float(mc)
    return float((mc - mu) / (mc + mu + 1e-9))

def _extract_checkpoint_centroids(r, assemblies):
    """Build stage list from experiment snapshots."""
    snapshots = r.get('snapshots', [])
    asm_size = len(assemblies[0])
    stages = []
    # Stage 0: after training first memory
    for j in range(len(snapshots)):
        cents = {}
        for i in range(len(snapshots)):
            sv = snapshots[i][j]
            if sv is not None:
                try:
                    W_sub = sv.reshape(asm_size, asm_size)
                    cents[i] = W_sub.mean(axis=1)
                except Exception:
                    cents[i] = None
            else:
                cents[i] = None
        stage_name = ['initial', 'post_B', 'post_C', 'post_D'][j] if j < 4 else f'post_{j}'
        stages.append({'stage_name': stage_name, 'centroids': cents})
    return stages

def _save_trajectory(r, net, mode, seed, assemblies, core_mask):
    """Save trajectory data in phase-script format."""
    try:
        # Build stage list from checkpoints + final
        stages = _extract_checkpoint_centroids(r, assemblies)
        # Add final stage
        if net is not None and hasattr(net, 'W'):
            W = net.W.data[:net.n_exc, :net.n_exc].cpu().numpy()
            fc = {}
            for i, asm in enumerate(assemblies):
                valid = [int(x) for x in asm if 0 <= int(x) < W.shape[0]]
                if len(valid) > 0:
                    try:
                        fc[i] = W[np.ix_(valid, valid)].mean(axis=1)
                    except Exception:
                        fc[i] = None
                else:
                    fc[i] = None
            stages.append({'stage_name': 'final', 'centroids': fc})

        core_idx = np.where(np.array(core_mask))[0].tolist() if hasattr(core_mask, '__len__') else []
        asm_list = [a.tolist() if hasattr(a, 'tolist') else list(a) for a in assemblies]

        data = {
            'mode': mode,
            'seed': seed,
            'assemblies': asm_list,
            'core_idx': core_idx,
            'core_mask': core_mask.tolist() if hasattr(core_mask, 'tolist') else list(core_mask),
            'trajectory': stages,
            'replay_events': list(_CENTROID_LOG),
            'baseline_scores': r.get('baseline_scores', []).tolist(),
            'final_scores': r.get('final_scores', []).tolist(),
            'retention_matrix': r.get('retention_matrix', None).tolist() if r.get('retention_matrix') is not None else None,
        }

        fname = f"trajectory_{mode}_seed{seed}.pkl"
        import pickle as _pk
        with open(fname, 'wb') as f:
            _pk.dump(data, f, protocol=_pk.HIGHEST_PROTOCOL)
        print(f"  [TRAJ] Saved -> {fname}", flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [TRAJ] Error saving trajectory: {e}", flush=True)


# ─────────────────────────────────────────────────────────────────────

def main(hyper_noise=None, n_seeds=None):
    global N_SEEDS, _HYPER_NOISE_STD
    if n_seeds is not None:
        N_SEEDS = n_seeds
    if hyper_noise is not None:
        _HYPER_NOISE_STD = hyper_noise

    t0 = time.time()
    print("="*70, flush=True)
    print(f"REPLAY DISTORTION & SCHEMA ABSTRACTION: {N_SEEDS} seeds", flush=True)
    print(f"Core={SCHEMA_CORE_SIZE}, DEV_MODE={ccf.DEV_MODE}", flush=True)
    print(f"Hyper: cue=4, strength=0.3, dur=2, spont=5, noise=8.0", flush=True)
    print(f"Hyper noise: {_HYPER_NOISE_STD} (post-replay)", flush=True)
    print(f"Schema boost: 1.3x core-to-core after replay", flush=True)
    print("="*70, flush=True)

    register_schema_hooks()
    MODES = ["no_replay", "natural", "hyper"]
    all_data = {m: {"results": [], "metrics": [], "forward": [],
                     "schema": []} for m in MODES}

    for si in range(N_SEEDS):
        seed = BASE_SEED + si * 1000
        print(f"\n-- Seed {si+1}/{N_SEEDS} (seed={seed})", flush=True)
        ccf.torch.manual_seed(seed); ccf.np.random.seed(seed)
        assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
        global _CORE_INDICES
        _CORE_INDICES = core_mask  # np.arange(SCHEMA_CORE_SIZE)

        for mode in MODES:
            t1 = time.time()
            use_replay = mode != "no_replay"
            if mode != "no_replay":
                old = install_mode(mode, assemblies)
            _CENTROID_LOG.clear()  # fresh log for this mode run

            try:
                # use_slow=True for all conditions; only use_replay varies
                r = ccf.run_sequential_experiment(True, use_replay, assemblies, seed)
            except Exception as e:
                import traceback
                print(f"  {mode:12s} CRASH: {e}", flush=True)
                traceback.print_exc()
                if mode != "no_replay":
                    restore_replay(old)
                continue

            if mode != "no_replay":
                restore_replay(old)

            _attach_schema_data([{"cond": {}, "trials": [r]}])

            # Sanitize
            fs = r['final_scores']
            bs = r['baseline_scores']
            if np.any(np.isnan(fs)) or np.any(np.isinf(fs)):
                fs = np.nan_to_num(fs, nan=0.0, posinf=0.0, neginf=0.0)
                r['final_scores'] = fs
            if np.any(np.isnan(bs)) or np.any(np.isinf(bs)):
                bs = np.nan_to_num(bs, nan=0.0, posinf=0.0, neginf=0.0)
                r['baseline_scores'] = bs

            metrics = compute_metrics(r, assemblies, core_mask)
            schema_m = sa.compute_all(r, n_mem=4, centroid_log=_CENTROID_LOG)

            all_data[mode]["results"].append(r)
            all_data[mode]["metrics"].append(metrics)
            all_data[mode]["schema"].append(schema_m)

            net = r.get('net', None)
            if net is None:
                try: net = _last_net
                except: net = None
            func_schema = measure_functional_schema(net, assemblies, core_mask)
            real_schema = compute_real_schema_index(net, assemblies, core_mask)
            all_data[mode].setdefault("real_schemas", []).append(real_schema)
            all_data[mode].setdefault("func_schemas", []).append(func_schema)

            sci = metrics.get("novel", {}).get("schema_crystallization_index", {}).get("final_SCI", "?")
            ss = schema_m.get("schema_score", "?")
            di = schema_m.get("distortion_index", "?")
            cnv = schema_m.get("convergence", "?")
            pv = schema_m.get("permutation_p", "?")
            ret_a = fs[0]

            # Snapshot centroid log BEFORE run_forward contaminates it
            centroid_log_main = list(_CENTROID_LOG)

            # Directional abstraction metric (main experiment events only)
            dall = compute_directional_alignment(centroid_log_main, n_mem=4, core_size=SCHEMA_CORE_SIZE)
            all_data[mode].setdefault("directional_alignment", []).append(dall)

            print(f"  {mode:12s} base={np.round(bs,4).tolist()}  final={np.round(fs,4).tolist()}", flush=True)
            print(f"             A={ret_a:.4f}  FS={func_schema:.1f}  SCI={sci}  SchemaScore={ss}", flush=True)
            print(f"             REAL_SCHEMA={real_schema:.4f}  Conv={cnv}  DistIdx={di}  p_drift={pv}", flush=True)
            print(f"             DAI_core={dall['mean_core']:+.4f}  DAI_uniq={dall['mean_unique']:+.4f}  p_core={dall['p_core']:.4f}  n_events={dall['n_events']}  ({time.time()-t1:.0f}s)", flush=True)

            fwd = run_forward(mode, assemblies, core_mask)
            if fwd and fwd["E"]:
                all_data[mode]["forward"].append(fwd)

            # Save trajectory data for centroid analysis
            _save_trajectory(r, net, mode, seed, assemblies, core_mask)

    # == Aggregate ====================================================
    print(f"\n{'='*70}", flush=True)
    print("AGGREGATED RESULTS", flush=True)
    print('='*70, flush=True)

    agg = {}
    for mode in MODES:
        d = all_data[mode]
        n = len(d["results"])
        if n == 0:
            print(f"\n  {mode.upper()}  (no valid seeds)", flush=True)
            continue

        fs_arr = np.array([r['final_scores'] for r in d["results"]])
        bs_arr = np.array([r['baseline_scores'] for r in d["results"]])
        mf = fs_arr.mean(0)
        sf = fs_arr.std(0) / np.sqrt(n)
        mb = bs_arr.mean(0) if len(bs_arr) > 0 else np.zeros(4)

        # Schema metrics
        ss_vals = [sm.get("schema_score", np.nan) for sm in d["schema"]]
        cnv_vals = [sm.get("convergence", np.nan) for sm in d["schema"]]
        di_vals  = [sm.get("distortion_index", np.nan) for sm in d["schema"]]
        p_vals   = [sm.get("permutation_p", np.nan) for sm in d["schema"]]

        # Real schema and functional schema
        rs_vals = d.get("real_schemas", [np.nan] * n)
        fs_vals_func = d.get("func_schemas", [np.nan] * n)

        # Directional alignment
        dall = d.get("directional_alignment", [])
        dai_core_vals  = [x.get('mean_core', np.nan)   for x in dall]
        dai_uniq_vals  = [x.get('mean_unique', np.nan) for x in dall]
        p_core_vals    = [x.get('p_core', 1.0)         for x in dall]
        p_uniq_vals    = [x.get('p_unique', 1.0)       for x in dall]
        n_events_vals  = [x.get('n_events', 0)         for x in dall]
        rs_delta_vals  = [x.get('mean_rs_delta', np.nan) for x in dall]

        agg[mode] = {
            "n": n,
            "retention_mean": mf.tolist(),
            "retention_sem": sf.tolist(),
            "baseline_mean": mb.tolist(),
            "schema_score_mean": float(np.nanmean(ss_vals)),
            "schema_score_sem": float(np.nanstd(ss_vals) / np.sqrt(n)),
            "convergence_mean": float(np.nanmean(cnv_vals)),
            "distortion_mean": float(np.nanmean(di_vals)),
            "p_drift_mean": float(np.nanmean(p_vals)),
            "real_schema_mean": float(np.nanmean(rs_vals)),
            "real_schema_sem": float(np.nanstd(rs_vals) / np.sqrt(n)) if n > 1 else 0.0,
            "func_schema_mean": float(np.nanmean(fs_vals_func)),
            "func_schema_sem": float(np.nanstd(fs_vals_func) / np.sqrt(n)) if n > 1 else 0.0,
            "dai_core_mean": float(np.nanmean(dai_core_vals)),
            "dai_core_sem": float(np.nanstd(dai_core_vals) / np.sqrt(n)) if n > 1 else 0.0,
            "dai_unique_mean": float(np.nanmean(dai_uniq_vals)),
            "dai_unique_sem": float(np.nanstd(dai_uniq_vals) / np.sqrt(n)) if n > 1 else 0.0,
            "p_core_mean": float(np.nanmean(p_core_vals)),
            "p_unique_mean": float(np.nanmean(p_uniq_vals)),
            "n_events_mean": float(np.nanmean(n_events_vals)),
            "rs_delta_mean": float(np.nanmean(rs_delta_vals)),
        }

        print(f"\n  {mode.upper()}  (n={n})", flush=True)
        for i, name in enumerate(['A', 'B', 'C', 'D']):
            print(f"    {name}: {mb[i]:.4f} -> {mf[i]:.4f} +/-{sf[i]:.4f}", flush=True)
        print(f"    SchemaScore:  {agg[mode]['schema_score_mean']:.4f} +/-{agg[mode]['schema_score_sem']:.4f}", flush=True)
        print(f"    REAL_SCHEMA:  {agg[mode]['real_schema_mean']:.4f} +/-{agg[mode]['real_schema_sem']:.4f}", flush=True)
        print(f"    FuncSchema:   {agg[mode]['func_schema_mean']:.2f} +/-{agg[mode]['func_schema_sem']:.2f}", flush=True)
        print(f"    Convergence:  {agg[mode]['convergence_mean']:.4f}", flush=True)
        print(f"    Distortion:   {agg[mode]['distortion_mean']:.4f}", flush=True)
        print(f"    p_drift:      {agg[mode]['p_drift_mean']:.4f}", flush=True)
        if dall:
            print(f"    DAI_core:     {agg[mode]['dai_core_mean']:+.4f} +/-{agg[mode]['dai_core_sem']:.4f}  p={agg[mode]['p_core_mean']:.4f}", flush=True)
            print(f"    DAI_unique:   {agg[mode]['dai_unique_mean']:+.4f} +/-{agg[mode]['dai_unique_sem']:.4f}  p={agg[mode]['p_unique_mean']:.4f}", flush=True)
            print(f"    RS_delta:     {agg[mode]['rs_delta_mean']:+.4f}", flush=True)
            print(f"    n_events:     {agg[mode]['n_events_mean']:.0f}", flush=True)

        fwd = d.get("forward", [])
        if fwd:
            e_val = np.mean([f["E"][-1] for f in fwd if f.get("E")])
            f_val = np.mean([f["F"][-1] for f in fwd if f.get("F")])
            print(f"    Forward:      E={e_val:.4f}  F={f_val:.4f}", flush=True)
        print(flush=True)

    # ── Hypothesis tests ─────────────────────────────────────────────
    from scipy.stats import ttest_ind, ttest_1samp
    print("HYPOTHESIS TESTS", flush=True)
    print("-"*50, flush=True)

    def _vec(mode, key, sub=None):
        if sub:
            return np.array([sm.get(sub, {}).get(key, np.nan) if isinstance(sm.get(sub), dict) else sm.get(key, np.nan)
                             for sm in all_data[mode]["schema"]])
        return np.array([sm.get(key, np.nan) for sm in all_data[mode]["schema"]])

    for label, nat_key, src in [
        ("Convergence",  "convergence", "schema"),
        ("SchemaScore",  "schema_score", "schema"),
        ("Distortion",   "distortion_index", "schema"),
    ]:
        if "natural" in agg and "hyper" in agg:
            nat_v = _vec("natural", nat_key)
            hyp_v = _vec("hyper", nat_key)
            valid = np.isfinite(nat_v) & np.isfinite(hyp_v)
            if valid.sum() >= 2:
                t, p = ttest_ind(nat_v[valid], hyp_v[valid])
                print(f"  {label:16s} Natural vs Hyper:    t={t:+.3f}  p={p:.4f}", flush=True)
        if "natural" in agg and "no_replay" in agg:
            nat_v = _vec("natural", nat_key)
            nor_v = _vec("no_replay", nat_key)
            valid = np.isfinite(nat_v) & np.isfinite(nor_v)
            if valid.sum() >= 2:
                t, p = ttest_ind(nat_v[valid], nor_v[valid])
                print(f"  {label:16s} Natural vs NoReplay: t={t:+.3f}  p={p:.4f}", flush=True)

    # DAI hypothesis tests
    print("", flush=True)
    for label, key in [("DAI_core", "mean_core"), ("DAI_unique", "mean_unique")]:
        for cond_a, cond_b in [("natural", "hyper"), ("natural", "no_replay"), ("hyper", "no_replay")]:
            if cond_a not in all_data or cond_b not in all_data:
                continue
            va = np.array([x.get(key, np.nan) for x in all_data[cond_a].get("directional_alignment", [])])
            vb = np.array([x.get(key, np.nan) for x in all_data[cond_b].get("directional_alignment", [])])
            valid = np.isfinite(va) & np.isfinite(vb)
            if valid.sum() >= 2:
                t, p = ttest_ind(va[valid], vb[valid])
                print(f"  {label:12s} {cond_a:10s} vs {cond_b:10s}: t={t:+.3f}  p={p:.4f}", flush=True)

    # DAI one-sample test (vs 0): is there any directional movement?
    print("", flush=True)
    for mode in MODES:
        dall = all_data[mode].get("directional_alignment", [])
        if dall:
            for key, label in [("mean_core", "DAI_core"), ("mean_unique", "DAI_unique")]:
                vals = np.array([x.get(key, np.nan) for x in dall])
                vals = vals[np.isfinite(vals)]
                if len(vals) >= 2:
                    t, p = ttest_1samp(vals, 0.0)
                    print(f"  {mode:12s} {label:12s} vs 0: mean={np.mean(vals):+.4f}  t={t:+.3f}  p={p:.4f}", flush=True)
    print("", flush=True)

    # Save ============================================================
    save = {m: {
        "finals": np.array([r['final_scores'].tolist() for r in all_data[m]["results"]]).tolist() if all_data[m]["results"] else [],
        "baselines": np.array([r['baseline_scores'].tolist() for r in all_data[m]["results"]]).tolist() if all_data[m]["results"] else [],
        "metrics": all_data[m]["metrics"],
        "schema": all_data[m]["schema"],
        "forward": all_data[m]["forward"],
        "directional_alignment": all_data[m].get("directional_alignment", []),
        "func_schemas": all_data[m].get("func_schemas", []),
        "real_schemas": all_data[m].get("real_schemas", []),
        "agg": agg.get(m, {}),
    } for m in MODES}
    save["config"] = {"n_seeds": N_SEEDS, "core": SCHEMA_CORE_SIZE, "unique": UNIQUE_SIZE}

    out = r'C:\Users\Admin\brain-organoid-rl\figures\schema\distortion_data.pkl'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(save, f)
    print(f"\nSaved -> {out}", flush=True)
    print(f"Total: {time.time()-t0:.0f}s ({((time.time()-t0)/60):.1f} min)", flush=True)
    print("DONE.", flush=True)


if __name__ == '__main__':
    import sys as _sys
    if '--test' in _sys.argv:
        n_seeds = 1
        for a in _sys.argv:
            if a.startswith('--seeds='):
                n_seeds = int(a.split('=')[1])
        print(f"\n-- TEST MODE ({n_seeds} seeds) --", flush=True)
        main(hyper_noise=0.005, n_seeds=n_seeds)
    else:
        main()
