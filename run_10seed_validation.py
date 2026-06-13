"""
10-SEED MECHANISTIC VALIDATION RUNNER
FULL / -M5 / -M1 / -M10  x  10 seeds each

Each seed runs in its own subprocess (guaranteed memory isolation).
Per-seed checkpoints written immediately — crash-safe, fully resumable.

Usage:
  python run_10seed_validation.py            # fresh run
  python run_10seed_validation.py --resume   # resume from last checkpoint
"""
import os, sys, time, argparse, pickle, json, subprocess
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

OUT_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results'
WORK_DIR = r'C:\Users\Admin\brain-organoid-rl'
PREFIX   = 'VAL10'
N_SEEDS  = 10
BASE_SEED = 42

CONDITIONS = {
    'FULL':       {},
    'ABLATE_M5':  {'drift':           False},
    'ABLATE_M1':  {'overlap_penalty': False},
    'ABLATE_M10': {'reconsol':        False},
}
LABELS = {
    'FULL':       'Full Model',
    'ABLATE_M5':  '-M5: Directional Drift',
    'ABLATE_M1':  '-M1: Overlap Coherence',
    'ABLATE_M10': '-M10: Reconsolidation Window',
}

def chk_path(cname, si):
    return os.path.join(OUT_DIR, f'{PREFIX}_{cname}_seed{si}.pkl')

def cond_path(cname):
    return os.path.join(OUT_DIR, f'{PREFIX}_{cname}.pkl')

def load_checkpoints(cname):
    """Return (list_of_results, next_seed_index)."""
    results, si = [], 0
    while si < N_SEEDS:
        p = chk_path(cname, si)
        if os.path.exists(p):
            with open(p, 'rb') as f:
                results.append(pickle.load(f))
            si += 1
        else:
            break
    return results, si

def run_seed(cname, si, seed, abl_dict, retry=2):
    """Spawn a fresh subprocess for one seed. Returns result dict or None."""
    chk = chk_path(cname, si)
    log = os.path.join(OUT_DIR, f'{PREFIX}_{cname}_seed{si}.log')
    cmd = [
        sys.executable, 'ablation_single_seed.py',
        cname, str(si), str(seed), json.dumps(abl_dict),
        '--prefix', PREFIX,
    ]
    env = {**os.environ, 'DEV_MODE': '1', 'PYTHONIOENCODING': 'utf-8'}

    for attempt in range(1, retry + 1):
        t0 = time.time()
        with open(log, 'w', encoding='utf-8') as logf:
            proc = subprocess.run(
                cmd, env=env, cwd=WORK_DIR,
                stdout=logf, stderr=subprocess.STDOUT
            )
        elapsed = time.time() - t0

        # Print key line from worker log
        try:
            lines = open(log, encoding='utf-8').readlines()
            result_line = next((l.strip() for l in lines if 'RS=' in l), '')
            if result_line:
                print(f'    {result_line}', flush=True)
        except Exception:
            pass

        if proc.returncode == 0 and os.path.exists(chk):
            with open(chk, 'rb') as f:
                return pickle.load(f), elapsed
        else:
            print(f'  ATTEMPT {attempt} FAILED (exit={proc.returncode}, elapsed={elapsed:.0f}s)',
                  flush=True)
            # Print last few log lines for debugging
            try:
                lines = open(log, encoding='utf-8').readlines()
                for l in lines[-5:]:
                    print(f'    LOG: {l.rstrip()}', flush=True)
            except Exception:
                pass
            if attempt < retry:
                print(f'  Retrying...', flush=True)
    return None, 0


def run_all(resume=False):
    print('=' * 65, flush=True)
    print('10-SEED MECHANISTIC VALIDATION', flush=True)
    print(f'Conditions: {list(CONDITIONS)}', flush=True)
    print(f'Seeds: {N_SEEDS}  |  BASE_SEED={BASE_SEED}  |  step=1000', flush=True)
    print(f'Subprocess per seed — guaranteed memory isolation.', flush=True)
    if resume:
        print('RESUME MODE active.', flush=True)
    print('=' * 65, flush=True)

    all_conditions = {}
    t_total = time.time()

    for cname, abl_dict in CONDITIONS.items():
        # Check if full condition already done
        cp = cond_path(cname)
        if resume and os.path.exists(cp):
            with open(cp, 'rb') as f:
                results = pickle.load(f)
            print(f'\n--- {LABELS[cname]} --- [COMPLETE: {len(results)} seeds loaded]', flush=True)
            all_conditions[cname] = results
            continue

        # Load partial checkpoints
        results, start_si = load_checkpoints(cname)
        if start_si > 0:
            print(f'\n--- {LABELS[cname]} --- [Resuming from seed {start_si+1}/{N_SEEDS}]', flush=True)
        else:
            print(f'\n--- {LABELS[cname]} ---', flush=True)

        skipped = 0
        for si in range(start_si, N_SEEDS):
            seed = BASE_SEED + si * 1000
            print(f'  Seed {si+1}/{N_SEEDS} (seed={seed}) ...', flush=True)
            res, elapsed = run_seed(cname, si, seed, abl_dict)
            if res is None:
                print(f'  Seed {si+1} skipped after {2} failed attempts.', flush=True)
                skipped += 1
                continue
            results.append(res)
            nat = res.get('natural', {})
            print(f'  Done ({elapsed:.0f}s)  '
                  f'RS={nat.get("real_schema",0):.4f}  '
                  f'DAI={nat.get("dai_core",0):.4f}  '
                  f'Ret_A={nat.get("retention_A",0):.4f}',
                  flush=True)

        # Save full condition
        with open(cp, 'wb') as f:
            pickle.dump(results, f)
        print(f'  Saved {cp}  ({len(results)} seeds, {skipped} skipped)', flush=True)
        all_conditions[cname] = results

    # Save combined results
    combined = os.path.join(OUT_DIR, 'validation_10seed.pkl')
    with open(combined, 'wb') as f:
        pickle.dump(all_conditions, f)

    elapsed_total = time.time() - t_total
    print(f'\nAll conditions complete in {elapsed_total:.0f}s ({elapsed_total/3600:.2f} hr)',
          flush=True)
    print(f'Results: {combined}', flush=True)
    return all_conditions


def print_summary(all_conditions):
    import numpy as np
    from scipy.stats import ttest_ind

    mode = 'natural'
    def get(cname, metric):
        return np.array([s[mode][metric] for s in all_conditions.get(cname, [])
                         if mode in s and metric in s[mode]])

    full_dai = get('FULL', 'dai_core')
    full_rs  = get('FULL', 'real_schema')

    print(f'\n{"="*80}')
    print(f'FINAL 10-SEED RESULTS  (mode={mode})')
    print(f'{"="*80}')
    hdr = f'  {"Condition":<28}  {"n":>3}  {"DAI":>8}±{"SEM":>5}  {"ΔDAI":>8}  {"RS":>8}±{"SEM":>5}  {"ΔRS":>8}  {"d":>6}  {"p":>8}  Sig'
    print(hdr); print('  '+'-'*90)

    dai_m = np.mean(full_dai) if len(full_dai) else 0
    rs_m  = np.mean(full_rs)  if len(full_rs) else 0
    n = len(full_dai)
    dai_se = np.std(full_dai,ddof=1)/np.sqrt(n) if n>1 else 0
    rs_se  = np.std(full_rs, ddof=1)/np.sqrt(n) if n>1 else 0
    print(f'  {"Full Model":<28}  {n:>3}  {dai_m:+8.4f}±{dai_se:.4f}  {"—":>8}  {rs_m:8.4f}±{rs_se:.4f}  {"—":>8}  {"—":>6}  {"—":>8}')

    for cname in ['ABLATE_M5','ABLATE_M1','ABLATE_M10']:
        dai_v = get(cname,'dai_core'); rs_v = get(cname,'real_schema')
        if len(dai_v)==0: continue
        n2 = len(dai_v)
        m2_dai=np.mean(dai_v); se2_dai=np.std(dai_v,ddof=1)/np.sqrt(n2) if n2>1 else 0
        m2_rs =np.mean(rs_v);  se2_rs =np.std(rs_v, ddof=1)/np.sqrt(n2) if n2>1 else 0
        sp = np.sqrt(((n-1)*np.std(full_dai,ddof=1)**2+(n2-1)*np.std(dai_v,ddof=1)**2)/(n+n2-2))
        cd = float((dai_m-m2_dai)/(sp+1e-12))
        _,p = ttest_ind(full_dai,dai_v,equal_var=False) if n>1 and n2>1 else (0,1)
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
        lbl = LABELS.get(cname,'')
        print(f'  {lbl:<28}  {n2:>3}  {m2_dai:+8.4f}±{se2_dai:.4f}  {m2_dai-dai_m:+8.4f}  '
              f'{m2_rs:8.4f}±{se2_rs:.4f}  {m2_rs-rs_m:+8.4f}  {cd:+6.3f}  {p:8.4f}  {sig}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    conds = run_all(resume=args.resume)
    print_summary(conds)
    print('\nNow generating figures from 10-seed data...', flush=True)
    # Update single_ablations.pkl so figure scripts pick it up
    import shutil
    shutil.copy2(os.path.join(OUT_DIR,'validation_10seed.pkl'),
                 os.path.join(OUT_DIR,'single_ablations.pkl'))
