"""
Serial-position-effect experiment driver.

Phase 1: Pre-flight diagnostic (1 seed, 3 fractions)
Phase 2: Full 4-memory experiment (10 seeds, 6 fractions) + gamma=0 probes
Phase 3: 8-memory experiment (10 seeds, 2 fractions)

Checkpointed: appends to CSV after each run. Re-runs skip completed rows.
"""
import os, sys, time, gc, argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from worker import run_single_timepoint

RESULTS_DIR = os.path.abspath(os.path.join(_HERE, '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

PHASE2_CSV   = os.path.join(RESULTS_DIR, 'phase2_results.csv')
GAMMA0_CSV   = os.path.join(RESULTS_DIR, 'phase2_gamma0.csv')
PHASE3_CSV   = os.path.join(RESULTS_DIR, 'phase3_8memories.csv')
DIAG_CSV     = os.path.join(RESULTS_DIR, 'phase1_diagnostic.csv')

EXPERIMENT_SEEDS = [42, 1042, 2042, 3042, 4042, 5042, 6042, 7042, 8042, 9042]
PROBE_FRACTIONS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]


def _already_done(csv_path, **filters):
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False
    if df.empty:
        return False
    mask = np.ones(len(df), dtype=bool)
    for k, v in filters.items():
        if k not in df.columns:
            return False
        mask &= np.isclose(df[k].values, v) if isinstance(v, float) else (df[k].values == v)
    return mask.any()


def _append_rows(csv_path, rows):
    header = not os.path.exists(csv_path)
    pd.DataFrame(rows).to_csv(csv_path, mode='a', index=False, header=header)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — diagnostic
# ─────────────────────────────────────────────────────────────────────────────
def run_diagnostic(seed=42, fractions=(0.0, 0.5, 1.0)):
    print(f"\n=== PHASE 1: DIAGNOSTIC (seed {seed}) ===", flush=True)
    rows = []
    for frac in fractions:
        if _already_done(DIAG_CSV, seed=seed, probe_fraction=frac):
            print(f"  skip seed={seed} frac={frac} (already done)", flush=True)
            continue
        t0 = time.time()
        out = run_single_timepoint(seed=seed, consolidation_fraction=frac,
                                    n_memories=4, use_gamma0_probe=(frac == 0.0))
        elapsed = time.time() - t0
        ret = out['retention']
        print(f"  seed={seed} frac={frac:.2f}: "
              f"M0={ret[0]:.4f} M1={ret[1]:.4f} M2={ret[2]:.4f} M3={ret[3]:.4f} "
              f"({elapsed:.0f}s)", flush=True)
        for m in range(4):
            row = dict(seed=seed, probe_fraction=frac, memory_id=m,
                       encoding_position=m,
                       isyn_score=ret[m],
                       w_slow=out['w_slow'][m],
                       w_fast=out['w_fast'][m],
                       assembly_w_fast=out['assembly_w_fast'][m],
                       gamma0_retention=(out['retention_g0'][m]
                                          if out['retention_g0'] else None),
                       elapsed_seconds=elapsed)
            rows.append(row)
        _append_rows(DIAG_CSV, rows[-4:])

    # Verdict
    df = pd.read_csv(DIAG_CSV)
    df = df[df.seed == seed]
    summary = df.groupby('probe_fraction')['isyn_score'].apply(list).to_dict()
    print("\n=== DIAGNOSTIC SUMMARY ===", flush=True)
    print(f"{'frac':>6}  {'M0':>8} {'M1':>8} {'M2':>8} {'M3':>8}  {'M3>M0?':>8}", flush=True)
    for frac in sorted(summary):
        vals = summary[frac]
        m3gt = 'YES' if vals[3] > vals[0] else 'no'
        print(f"{frac:>6.2f}  {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} {vals[3]:>8.4f}  {m3gt:>8}",
              flush=True)

    immediate = summary.get(0.0)
    delayed = summary.get(1.0)
    if immediate and delayed:
        recency = immediate[3] > immediate[0]
        primacy = delayed[0] > delayed[3]
        print(f"\n  recency at immediate (M3>M0): {recency}", flush=True)
        print(f"  primacy at delayed   (M0>M3): {primacy}", flush=True)
        print(f"  CROSSOVER: {recency and primacy}", flush=True)

        # gamma=0 probe (Approach E)
        g0 = df[(df.probe_fraction == 0.0)].sort_values('memory_id')
        g0_vals = g0['gamma0_retention'].tolist()
        if g0_vals and not all(pd.isna(g0_vals)):
            print(f"\n  gamma=0 probe (M0..M3): " +
                  ' '.join(f"{v:.4f}" for v in g0_vals), flush=True)
            print(f"  gamma=0 recency (M3>M0): {g0_vals[3] > g0_vals[0]}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — full 4-memory experiment
# ─────────────────────────────────────────────────────────────────────────────
def run_phase2(seeds=EXPERIMENT_SEEDS, fractions=PROBE_FRACTIONS):
    print(f"\n=== PHASE 2: 4-MEMORY ({len(seeds)} seeds x {len(fractions)} fracs) ===",
          flush=True)
    total = len(seeds) * len(fractions)
    idx = 0
    for seed in seeds:
        for frac in fractions:
            idx += 1
            if _already_done(PHASE2_CSV, seed=seed, probe_fraction=frac):
                print(f"  [{idx}/{total}] skip seed={seed} frac={frac}", flush=True)
                continue
            t0 = time.time()
            # Add gamma=0 probe at frac=0.0 only (saves time)
            out = run_single_timepoint(
                seed=seed, consolidation_fraction=frac,
                n_memories=4, use_gamma0_probe=(frac == 0.0)
            )
            elapsed = time.time() - t0
            ret = out['retention']
            rows = []
            for m in range(4):
                rows.append(dict(
                    seed=seed, probe_fraction=frac, memory_id=m,
                    encoding_position=m,
                    isyn_score=ret[m],
                    w_slow=out['w_slow'][m],
                    w_fast=out['w_fast'][m],
                    assembly_w_fast=out['assembly_w_fast'][m],
                    gamma0_retention=(out['retention_g0'][m]
                                       if out['retention_g0'] else None),
                    elapsed_seconds=elapsed,
                ))
            _append_rows(PHASE2_CSV, rows)
            print(f"  [{idx}/{total}] seed={seed} frac={frac:.2f}: "
                  f"M0={ret[0]:.4f} M3={ret[3]:.4f} ({elapsed:.0f}s)", flush=True)
            gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — 8-memory experiment
# ─────────────────────────────────────────────────────────────────────────────
def run_phase3(seeds=EXPERIMENT_SEEDS, fractions=(0.0, 1.0)):
    print(f"\n=== PHASE 3: 8-MEMORY ({len(seeds)} seeds x {len(fractions)} fracs) ===",
          flush=True)
    total = len(seeds) * len(fractions)
    idx = 0
    for seed in seeds:
        for frac in fractions:
            idx += 1
            if _already_done(PHASE3_CSV, seed=seed, probe_fraction=frac):
                print(f"  [{idx}/{total}] skip seed={seed} frac={frac}", flush=True)
                continue
            t0 = time.time()
            out = run_single_timepoint(seed=seed, consolidation_fraction=frac,
                                        n_memories=8)
            elapsed = time.time() - t0
            ret = out['retention']
            rows = []
            for m in range(8):
                rows.append(dict(
                    seed=seed, probe_fraction=frac, memory_id=m,
                    encoding_position=m, n_memories=8,
                    isyn_score=ret[m],
                    w_slow=out['w_slow'][m],
                    w_fast=out['w_fast'][m],
                    assembly_w_fast=out['assembly_w_fast'][m],
                    elapsed_seconds=elapsed,
                ))
            _append_rows(PHASE3_CSV, rows)
            print(f"  [{idx}/{total}] 8mem seed={seed} frac={frac:.2f}: " +
                  ' '.join(f"M{i}={ret[i]:.3f}" for i in range(8)) +
                  f" ({elapsed:.0f}s)", flush=True)
            gc.collect()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['1', '2', '3', 'all'], default='all')
    ap.add_argument('--seeds', type=int, nargs='*', default=None)
    args = ap.parse_args()

    seeds = args.seeds if args.seeds else EXPERIMENT_SEEDS

    if args.phase in ('1', 'all'):
        run_diagnostic(seed=42, fractions=(0.0, 0.5, 1.0))
    if args.phase in ('2', 'all'):
        run_phase2(seeds=seeds)
    if args.phase in ('3', 'all'):
        run_phase3(seeds=seeds)

    print("\n=== ALL PHASES COMPLETE ===", flush=True)
