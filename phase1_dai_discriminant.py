"""
PHASE 1 — DAI Discriminant Validation
======================================
Demonstrate that DAI captures directional schema abstraction,
not merely convergence.

Four synthetic trajectory types (100 trials each):

  A: Convergent + schema-aligned    → DAI high, Convergence high
  B: Convergent + schema-misaligned → DAI low,  Convergence high
  C: Non-convergent + aligned       → DAI high, Convergence low
  D: Random walk                    → DAI ~0,   Convergence ~0

Key test: DAI and convergence must be dissociable (low correlation).

Convergence metric: reduction in mean pairwise cosine distance
    = (initial pairwise dist) - (final pairwise dist)
    Positive = memories converging toward each other.
"""
import sys, os, json
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np
from scipy.stats import ttest_ind, pearsonr
from scipy.spatial.distance import cosine as cosine_dist

from _distortion_paper import compute_directional_alignment
from schema_abstraction.schema_experiments import SCHEMA_CORE_SIZE, UNIQUE_SIZE

DIM       = SCHEMA_CORE_SIZE + UNIQUE_SIZE   # 40
CORE_SIZE = SCHEMA_CORE_SIZE                 # 20
N_MEM     = 4
N_EVENTS  = 45
N_TRIALS  = 100
ALPHA     = 0.10
NOISE     = 0.01

OUT = r'C:\Users\Admin\brain-organoid-rl\figures\validation'
os.makedirs(OUT, exist_ok=True)

COND_META = {
    'A_conv_aligned':     ('Convergent\n+ Schema-aligned',   '#27ae60'),
    'B_conv_misaligned':  ('Convergent\n+ Schema-misaligned','#e74c3c'),
    'C_nonconv_aligned':  ('Non-convergent\n+ Aligned',       '#f39c12'),
    'D_random':           ('Random Walk',                     '#95a5a6'),
}


# ── Convergence metric ────────────────────────────────────────────────────────

def pairwise_cosine_dist(centroids):
    """Mean pairwise cosine distance between memory centroids."""
    keys = sorted(centroids.keys())
    if len(keys) < 2:
        return 0.0
    dists = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            v1 = np.array(centroids[keys[i]])
            v2 = np.array(centroids[keys[j]])
            try:
                dists.append(float(cosine_dist(v1, v2)))
            except Exception:
                pass
    return float(np.mean(dists)) if dists else 0.0


def convergence_from_log(log, n_mem=N_MEM):
    """
    Convergence = reduction in mean pairwise cosine distance.
    Positive = memories got closer to each other.
    """
    if not log:
        return 0.0
    # Initial centroids: use centroid_before of earliest event for each memory
    init_cents = {}
    for e in log:
        cb = e.get('centroid_before', {})
        for k, v in cb.items():
            ik = int(k)
            if ik not in init_cents:
                init_cents[ik] = v

    # Final centroids: latest centroid_after for each memory
    final_cents = {}
    for e in log:
        for k, v in e.get('centroid_after', {}).items():
            final_cents[int(k)] = v

    if len(init_cents) < 2 or len(final_cents) < 2:
        return 0.0

    d_init  = pairwise_cosine_dist(init_cents)
    d_final = pairwise_cosine_dist(final_cents)
    return float(d_init - d_final)   # positive = converged


# ── Trajectory generators ─────────────────────────────────────────────────────

def _make_log(condition, rng, n_events=N_EVENTS, n_mem=N_MEM, dim=DIM):
    """
    Build a centroid log for the given condition.
    All conditions use the same initial random centroids.
    """
    # Shared initial state
    centroids = {i: rng.randn(dim) for i in range(n_mem)}
    true_schema = np.mean([centroids[k] for k in range(n_mem)], axis=0)

    log = []
    for ev in range(n_events):
        mem_idx = ev % n_mem
        before  = centroids[mem_idx].copy()

        group_mean = np.mean([centroids[k] for k in range(n_mem)], axis=0)
        toward_schema = group_mean - before        # toward schema = group centroid

        if condition == 'A_conv_aligned':
            # Move toward group mean (convergent AND schema-aligned)
            delta = ALPHA * toward_schema + NOISE * rng.randn(dim)

        elif condition == 'B_conv_misaligned':
            # Converge but toward a WRONG attractor (orthogonal to true schema)
            # Construct a "wrong schema" orthogonal to true_schema
            wrong = rng.randn(dim)
            wrong -= np.dot(wrong, true_schema / (np.linalg.norm(true_schema)+1e-12)) * (true_schema / (np.linalg.norm(true_schema)+1e-12))
            wrong /= (np.linalg.norm(wrong) + 1e-12)
            wrong_target = wrong * np.linalg.norm(group_mean)
            toward_wrong = wrong_target - before
            delta = ALPHA * toward_wrong + NOISE * rng.randn(dim)

        elif condition == 'C_nonconv_aligned':
            # Move in schema direction (group-mean direction) but also spread orthogonally
            unit_schema = toward_schema / (np.linalg.norm(toward_schema) + 1e-12)
            ortho = rng.randn(dim)
            ortho -= np.dot(ortho, unit_schema) * unit_schema
            ortho /= (np.linalg.norm(ortho) + 1e-12)
            # Strong orthogonal spread, moderate schema alignment
            delta = 0.5 * ALPHA * toward_schema + 1.5 * ALPHA * ortho + NOISE * rng.randn(dim)

        elif condition == 'D_random':
            rand_dir = rng.randn(dim)
            rand_dir /= (np.linalg.norm(rand_dir) + 1e-12)
            delta = ALPHA * np.linalg.norm(toward_schema) * rand_dir

        else:
            raise ValueError(condition)

        after = before + delta
        cb = {k: v.tolist() for k, v in centroids.items()}
        centroids[mem_idx] = after.copy()
        ca = {k: v.tolist() for k, v in centroids.items()}
        log.append({'replay_id': ev, 'memory_idx': mem_idx,
                    'centroid_before': cb, 'centroid_after': ca})

    return log


# ── Main ─────────────────────────────────────────────────────────────────────

def run_phase1():
    print('='*65, flush=True)
    print('PHASE 1: DAI DISCRIMINANT VALIDATION', flush=True)
    print('='*65, flush=True)
    print(f'N={N_TRIALS} trials per condition, {N_EVENTS} events each', flush=True)

    conditions = list(COND_META.keys())
    results = {c: {'dai': [], 'conv': []} for c in conditions}

    for trial in range(N_TRIALS):
        rng = np.random.RandomState(trial * 13 + 7)
        for cond in conditions:
            log = _make_log(cond, rng)
            out = compute_directional_alignment(log, n_mem=N_MEM, core_size=CORE_SIZE)
            conv = convergence_from_log(log, n_mem=N_MEM)
            results[cond]['dai'].append(float(out['mean_core']))
            results[cond]['conv'].append(float(conv))

    print(f'\n{"Condition":30s}  {"DAI_core":>10}  {"Convergence":>12}', flush=True)
    print('-'*55, flush=True)
    for cond in conditions:
        d = np.array(results[cond]['dai'])
        c = np.array(results[cond]['conv'])
        label = COND_META[cond][0].replace('\n', ' ')
        print(f'  {label:30s}  {np.mean(d):+.4f}±{np.std(d):.4f}  {np.mean(c):+.6f}±{np.std(c):.6f}',
              flush=True)

    # Between-condition tests — DAI
    print(f'\nDAI pairwise t-tests:', flush=True)
    for i, ca in enumerate(conditions):
        for cb in conditions[i+1:]:
            da = np.array(results[ca]['dai'])
            db = np.array(results[cb]['dai'])
            t, p = ttest_ind(da, db)
            stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
            la = COND_META[ca][0].replace('\n', ' ')
            lb = COND_META[cb][0].replace('\n', ' ')
            print(f'  {la:28s} vs {lb:28s}: t={t:+.2f}  p={p:.4e}  {stars}', flush=True)

    # Correlation: DAI vs Convergence (should be low)
    print(f'\nDAI vs Convergence correlation:', flush=True)
    all_dai  = np.concatenate([results[c]['dai']  for c in conditions])
    all_conv = np.concatenate([results[c]['conv'] for c in conditions])
    r, p = pearsonr(all_dai, all_conv)
    print(f'  r = {r:.4f}  p = {p:.4e}', flush=True)

    # Per-condition correlations
    for cond in conditions:
        d = np.array(results[cond]['dai'])
        c = np.array(results[cond]['conv'])
        if np.std(d) > 1e-10 and np.std(c) > 1e-10:
            r2, p2 = pearsonr(d, c)
            label = COND_META[cond][0].replace('\n', ' ')
            print(f'  {label:30s}: r={r2:.4f}  p={p2:.4f}', flush=True)

    # Discriminability: DAI separates A from B; Convergence does NOT
    dai_A  = np.array(results['A_conv_aligned']['dai'])
    dai_B  = np.array(results['B_conv_misaligned']['dai'])
    conv_A = np.array(results['A_conv_aligned']['conv'])
    conv_B = np.array(results['B_conv_misaligned']['conv'])
    t_dai,  p_dai  = ttest_ind(dai_A,  dai_B)
    t_conv, p_conv = ttest_ind(conv_A, conv_B)
    print(f'\nKEY TEST — Can separate Aligned vs Misaligned?', flush=True)
    print(f'  DAI discriminates  A vs B: t={t_dai:+.2f}  p={p_dai:.4e}  '
          f'{"PASS" if p_dai < 0.05 else "FAIL"}', flush=True)
    print(f'  Conv discriminates A vs B: t={t_conv:+.2f}  p={p_conv:.4e}  '
          f'(should NOT — shows DAI adds info beyond convergence)', flush=True)

    passed = p_dai < 0.05
    print(f'\nPhase 1: {"PASS" if passed else "FAIL"}', flush=True)

    # Save
    save = {c: {k: [float(x) for x in v] for k, v in d.items()}
            for c, d in results.items()}
    save['meta'] = {'correlation_all': float(r), 'p_correlation': float(p)}
    with open(os.path.join(OUT, 'phase1_discriminant_raw.json'), 'w') as f:
        json.dump(save, f)

    return results, passed


if __name__ == '__main__':
    results, passed = run_phase1()
