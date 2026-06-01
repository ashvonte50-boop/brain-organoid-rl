"""
PHASE B: REAL_SCHEMA Synthetic Validation
==========================================
Construct synthetic networks with known weight structure and verify
compute_real_schema_index() returns monotonically correct values.

Three cases:
  Case 1 — Strong core-core structure   → REAL_SCHEMA high   (≈ +1)
  Case 2 — Random / uniform weights     → REAL_SCHEMA ≈ 0
  Case 3 — Unique-to-core dominant      → REAL_SCHEMA low    (≈ -1)

Also runs:
  * Scaling curve: varying core-core strength 0 → 1
  * Noise robustness: adding increasing Gaussian noise
"""
import sys, os, json
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np
from scipy.stats import ttest_ind
import torch

from schema_abstraction.schema_experiments import SCHEMA_CORE_SIZE, UNIQUE_SIZE

N_EXC      = 400
CORE_SIZE  = SCHEMA_CORE_SIZE   # 20
UNIQUE_SZ  = UNIQUE_SIZE        # 20
N_MEM      = 4
N_TRIALS   = 100

OUT = r'C:\Users\Admin\brain-organoid-rl\figures\validation'
os.makedirs(OUT, exist_ok=True)


# ── Mock network ──────────────────────────────────────────────────────────────

class _MockW:
    def __init__(self, W_np):
        self.data = torch.tensor(W_np, dtype=torch.float32)

class MockNet:
    def __init__(self, W_np):
        self.W     = _MockW(W_np)
        self.n_exc = W_np.shape[0]


# ── Synthetic assemblies ──────────────────────────────────────────────────────

def make_assemblies(n_mem=N_MEM, core=CORE_SIZE, unique=UNIQUE_SZ):
    core_mask = np.arange(core, dtype=np.int64)
    assemblies = []
    for i in range(n_mem):
        unique_start = core + i * unique
        asm = np.concatenate([core_mask,
                               np.arange(unique_start, unique_start + unique,
                                         dtype=np.int64)])
        assemblies.append(asm)
    return assemblies, core_mask


ASSEMBLIES, CORE_MASK = make_assemblies()


# ── Weight matrix constructors ────────────────────────────────────────────────

def make_W(case, rng, n_exc=N_EXC,
           core=CORE_SIZE, base=0.2, hi=0.8, lo=0.05):
    """
    Build a symmetric excitatory weight matrix for the given case.
    All weights are non-negative (no inhibitory connections).
    """
    W = np.full((n_exc, n_exc), base, dtype=np.float32)
    np.fill_diagonal(W, 0.0)

    if case == 'strong_core':
        # Core-core much stronger than unique-to-core
        W[:core, :core] = hi
        np.fill_diagonal(W[:core, :core], 0.0)
        # unique-to-core / core-to-unique at base
        W[core:, :core] = base
        W[:core, core:] = base

    elif case == 'random':
        # Uniform random, no structure
        W = rng.uniform(base, base + 0.05, (n_exc, n_exc)).astype(np.float32)
        np.fill_diagonal(W, 0.0)

    elif case == 'core_unique_dominant':
        # Unique-to-core weights dominate core-core
        W[:core, :core] = lo
        np.fill_diagonal(W[:core, :core], 0.0)
        W[core:, :core] = hi   # unique rows, core columns
        W[:core, core:] = hi

    return W


# ── Wrapper that calls the real metric ────────────────────────────────────────

def real_schema(W_np):
    from _distortion_paper import compute_real_schema_index
    net = MockNet(W_np)
    return compute_real_schema_index(net, ASSEMBLIES, CORE_MASK)


# ── Phase B main ─────────────────────────────────────────────────────────────

def run_phase_b():
    print('='*65, flush=True)
    print('PHASE B: REAL_SCHEMA SYNTHETIC VALIDATION', flush=True)
    print('='*65, flush=True)
    print(f'n_exc={N_EXC}  core={CORE_SIZE}  n_trials={N_TRIALS}', flush=True)
    print(flush=True)

    rng = np.random.RandomState(42)
    cases = ['strong_core', 'random', 'core_unique_dominant']
    labels = {'strong_core': 'Strong core',
              'random':      'Random',
              'core_unique_dominant': 'Unique dominant'}
    expected = {'strong_core': 'HIGH (+)', 'random': 'NEAR 0', 'core_unique_dominant': 'LOW (-)'}

    results = {}
    for case in cases:
        vals = []
        for _ in range(N_TRIALS):
            W = make_W(case, rng)
            vals.append(real_schema(W))
        results[case] = np.array(vals)
        m, s = np.mean(vals), np.std(vals)
        print(f'  {labels[case]:22s}  RS = {m:+.4f} ± {s:.4f}'
              f'  expected: {expected[case]}', flush=True)

    print(flush=True)

    # Ordering checks
    print('MONOTONICITY CHECK:', flush=True)
    m_sc  = float(np.mean(results['strong_core']))
    m_rn  = float(np.mean(results['random']))
    m_ud  = float(np.mean(results['core_unique_dominant']))
    print(f'  strong_core ({m_sc:+.4f}) > random ({m_rn:+.4f}) > '
          f'unique_dominant ({m_ud:+.4f})', flush=True)
    mono = m_sc > m_rn > m_ud
    print(f'  Monotonic ordering: {"PASS" if mono else "FAIL"}', flush=True)

    # Between-case t-tests
    print(flush=True)
    print('BETWEEN-CASE TESTS:', flush=True)
    for (ca, cb) in [('strong_core','random'), ('random','core_unique_dominant'),
                     ('strong_core','core_unique_dominant')]:
        t, p = ttest_ind(results[ca], results[cb])
        stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
        print(f'  {labels[ca]:22s} vs {labels[cb]:22s}: '
              f't={t:+.2f}  p={p:.4e}  {stars}', flush=True)

    # Scaling curve: vary core-core weight from 0.05 to 0.95
    print(flush=True)
    print('SCALING CURVE (core-core strength 0.05 to 0.95):', flush=True)
    scaling = {}
    cc_strengths = np.linspace(0.05, 0.95, 19)
    for cc_strength in cc_strengths:
        vals = []
        for _ in range(20):
            W = make_W('random', rng)   # start from random
            W[:CORE_SIZE, :CORE_SIZE] = cc_strength
            np.fill_diagonal(W[:CORE_SIZE, :CORE_SIZE], 0.0)
            vals.append(real_schema(W))
        scaling[float(cc_strength)] = float(np.mean(vals))

    strengths = sorted(scaling.keys())
    rs_vals   = [scaling[s] for s in strengths]
    monotone  = all(rs_vals[i] <= rs_vals[i+1]
                    for i in range(len(rs_vals)-1))
    print(f'  {"Strength":>8}  {"RS":>8}')
    for s, r in zip(strengths[::3], rs_vals[::3]):
        print(f'  {s:8.3f}  {r:+8.4f}')
    print(f'  Monotonically increasing: {"PASS" if monotone else "FAIL"}',
          flush=True)

    # Noise robustness
    print(flush=True)
    print('NOISE ROBUSTNESS (adding Gaussian noise to strong-core matrix):', flush=True)
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5]
    noise_robustness = {}
    for sigma in noise_levels:
        vals = []
        for _ in range(50):
            W = make_W('strong_core', rng)
            W += rng.normal(0, sigma, W.shape).astype(np.float32)
            W = np.clip(W, 0, None)
            np.fill_diagonal(W, 0.0)
            vals.append(real_schema(W))
        noise_robustness[sigma] = {'mean': float(np.mean(vals)),
                                    'std': float(np.std(vals))}
        print(f'  sigma={sigma:.2f}  RS={np.mean(vals):+.4f} ± {np.std(vals):.4f}',
              flush=True)

    # Save
    save = {
        'per_case':     {c: v.tolist() for c, v in results.items()},
        'scaling':      scaling,
        'noise_robust': noise_robustness,
    }
    with open(os.path.join(OUT, 'real_schema_validation_raw.json'), 'w') as f:
        json.dump(save, f)
    print(f'\nRaw data -> {OUT}/real_schema_validation_raw.json', flush=True)

    return results, scaling, noise_robustness, mono


if __name__ == '__main__':
    results, scaling, noise, mono = run_phase_b()
    print(f'\nPhase B: {"PASS" if mono else "FAIL"}', flush=True)
