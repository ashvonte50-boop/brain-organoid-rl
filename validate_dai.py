"""
PHASE A: DAI Synthetic Validation
==================================
Build synthetic datasets with known ground truth to prove
compute_directional_alignment() is a calibrated instrument.

Three conditions:
  ALIGNED     — trajectories move directly toward schema  → DAI ≈ +1
  RANDOM      — trajectories are random walks             → DAI ≈  0
  ANTI_ALIGNED — trajectories move directly away          → DAI ≈ -1

100 synthetic logs per condition.
"""
import sys, os
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp
import json

from _distortion_paper import compute_directional_alignment
from schema_abstraction.schema_experiments import SCHEMA_CORE_SIZE, UNIQUE_SIZE

DIM       = SCHEMA_CORE_SIZE + UNIQUE_SIZE   # 40
CORE_SIZE = SCHEMA_CORE_SIZE                 # 20
N_MEM     = 4
N_EVENTS  = 45
N_TRIALS  = 100
ALPHA     = 0.10   # step fraction toward/away from schema
NOISE     = 0.01   # noise magnitude

OUT = r'C:\Users\Admin\brain-organoid-rl\figures\validation'
os.makedirs(OUT, exist_ok=True)


# ── Synthetic log generator ───────────────────────────────────────────────────

def make_log(condition, n_events=N_EVENTS, n_mem=N_MEM,
             dim=DIM, core_size=CORE_SIZE, seed=0):
    """
    Build a centroid_log in the format expected by compute_directional_alignment.

    Reference direction: the GROUP CENTROID (mean of all current centroids).
    This exactly matches what compute_directional_alignment uses as its schema
    attractor, so the expected DAI values are analytically predictable:

      ALIGNED      — move each memory toward the group centroid  → DAI ≈ +1
      RANDOM       — move in a random direction                  → DAI ≈  0
      ANTI_ALIGNED — move each memory away from the group centroid → DAI ≈ -1

    The group centroid stays approximately fixed across all three conditions
    (random walk is mean-preserving; aligned/anti-aligned movements cancel out
    across memories), so compute_directional_alignment's inferred schema_attractor
    ≈ group centroid throughout.
    """
    rng = np.random.RandomState(seed)

    # Start at random positions on the unit sphere
    centroids = {}
    for i in range(n_mem):
        c = rng.randn(dim)
        centroids[i] = c.copy()

    log = []
    for ev in range(n_events):
        mem_idx = ev % n_mem
        before  = centroids[mem_idx].copy()

        # Group centroid (schema attractor, known analytically)
        group_mean = np.mean([centroids[k] for k in range(n_mem)], axis=0)
        toward     = group_mean - before     # direction toward schema
        d_norm     = np.linalg.norm(toward)
        if d_norm < 1e-12:
            d_norm = 1.0

        if condition == 'aligned':
            # Move toward group centroid (schema convergence)
            delta = ALPHA * toward + NOISE * rng.randn(dim)

        elif condition == 'random':
            # Random direction, same expected step magnitude
            rand_dir = rng.randn(dim)
            rand_dir /= (np.linalg.norm(rand_dir) + 1e-12)
            delta = ALPHA * d_norm * rand_dir

        elif condition == 'anti_aligned':
            # Move AWAY from group centroid (memories diverge from schema)
            delta = -ALPHA * toward + NOISE * rng.randn(dim)

        else:
            raise ValueError(condition)

        after = before + delta

        cb = {k: v.tolist() for k, v in centroids.items()}
        centroids[mem_idx] = after.copy()
        ca = {k: v.tolist() for k, v in centroids.items()}

        log.append({
            'replay_id':       ev,
            'memory_idx':      mem_idx,
            'centroid_before': cb,
            'centroid_after':  ca,
        })

    # Return log and the initial group mean as the "true schema"
    true_schema = np.mean(list(centroids.values()), axis=0)  # ≈ initial mean
    return log, true_schema


# ── Run validation ────────────────────────────────────────────────────────────

def run_phase_a():
    print('='*65, flush=True)
    print('PHASE A: DAI SYNTHETIC VALIDATION', flush=True)
    print('='*65, flush=True)
    print(f'dim={DIM}  core={CORE_SIZE}  n_mem={N_MEM}  '
          f'n_events={N_EVENTS}  n_trials={N_TRIALS}', flush=True)
    print(flush=True)

    results = {}
    for condition in ('aligned', 'random', 'anti_aligned'):
        core_vals, uniq_vals, n_ev_vals = [], [], []
        attractor_errors = []   # |computed_attractor - true_schema|

        for trial in range(N_TRIALS):
            log, true_schema = make_log(condition, seed=trial * 7 + 13)
            out = compute_directional_alignment(
                log, n_mem=N_MEM, core_size=CORE_SIZE
            )
            core_vals.append(out['mean_core'])
            uniq_vals.append(out['mean_unique'])
            n_ev_vals.append(out['n_events'])

            # Check how well the inferred attractor matches true_schema
            # Reconstruct the attractor exactly as the function does
            latest = {}
            for e in log:
                for k, v in e['centroid_after'].items():
                    latest[int(k)] = np.array(v)
            if latest:
                inferred = np.mean(list(latest.values()), axis=0)
                err = 1.0 - abs(np.dot(inferred / np.linalg.norm(inferred),
                                       true_schema))
                attractor_errors.append(err)

        results[condition] = {
            'mean_core':  np.array(core_vals),
            'mean_unique': np.array(uniq_vals),
            'n_events':   np.array(n_ev_vals),
            'attractor_err': np.array(attractor_errors),
        }

        mc = np.mean(core_vals)
        sc = np.std(core_vals)
        t1, p1 = ttest_1samp(core_vals, 0.0)
        print(f'  {condition:14s}  DAI_core = {mc:+.4f} ± {sc:.4f}  '
              f'(t={t1:+.2f}, p={p1:.4e})  n_ev={np.mean(n_ev_vals):.1f}',
              flush=True)
        if attractor_errors:
            print(f'                  attractor_error = '
                  f'{np.mean(attractor_errors):.4f} ± {np.std(attractor_errors):.4f}',
                  flush=True)

    print(flush=True)

    # Between-condition t-tests
    print('BETWEEN-CONDITION TESTS (DAI_core):', flush=True)
    conds = list(results.keys())
    for i in range(len(conds)):
        for j in range(i+1, len(conds)):
            ca, cb = conds[i], conds[j]
            t, p = ttest_ind(results[ca]['mean_core'], results[cb]['mean_core'])
            stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
            print(f'  {ca:14s} vs {cb:14s}:  t={t:+.2f}  p={p:.4e}  {stars}',
                  flush=True)

    # Success criterion
    means = {k: float(np.mean(v['mean_core'])) for k, v in results.items()}
    sep_al_rn = means['aligned']  - means['random']
    sep_rn_aa = means['random']   - means['anti_aligned']
    print(flush=True)
    print('SUCCESS CRITERION (Aligned > Random > Anti-aligned):', flush=True)
    print(f'  aligned  = {means["aligned"]:+.4f}', flush=True)
    print(f'  random   = {means["random"]:+.4f}', flush=True)
    print(f'  anti     = {means["anti_aligned"]:+.4f}', flush=True)
    print(f'  sep(aligned-random)      = {sep_al_rn:.4f}  '
          f'{"PASS" if sep_al_rn > 0.1 else "FAIL"}', flush=True)
    print(f'  sep(random-anti_aligned) = {sep_rn_aa:.4f}  '
          f'{"PASS" if sep_rn_aa > 0.1 else "FAIL"}', flush=True)
    passed = sep_al_rn > 0.1 and sep_rn_aa > 0.1

    # Save raw data
    save = {c: {k: v.tolist() for k, v in d.items()} for c, d in results.items()}
    with open(os.path.join(OUT, 'dai_validation_raw.json'), 'w') as f:
        json.dump(save, f)
    print(f'\nRaw data -> {OUT}/dai_validation_raw.json', flush=True)

    return results, passed


if __name__ == '__main__':
    results, passed = run_phase_a()
    print(f'\nPhase A: {"PASS" if passed else "FAIL"}', flush=True)
