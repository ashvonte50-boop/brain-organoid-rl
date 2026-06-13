"""
Single-seed worker — spawned as subprocess.
Runs ONE seed for ONE condition with a specific REPLAY_COHERENCE_THR.
Saves PKL and exits (full memory release).

Usage:
  python ablation_single_seed.py <cname> <si> <seed> <abl_json> [--prefix P] [--coh_thr F]
"""
import os, sys, json, pickle, argparse
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

parser = argparse.ArgumentParser()
parser.add_argument('cname')
parser.add_argument('si',   type=int)
parser.add_argument('seed', type=int)
parser.add_argument('abl_json')
parser.add_argument('--prefix',  default='VAL')
parser.add_argument('--coh_thr', type=float, default=None,
                    help='Override REPLAY_COHERENCE_THR before running')
parser.add_argument('--boost_scale', type=float, default=1.3,
                    help='MB core-boost multiplier (1.0 = MB disabled)')
args = parser.parse_args()

import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True; ccf.N_WORKERS = 1

# Apply coherence threshold override BEFORE importing pipeline
if args.coh_thr is not None:
    ccf.REPLAY_COHERENCE_THR = args.coh_thr
    print(f'[worker] REPLAY_COHERENCE_THR overridden -> {args.coh_thr}', flush=True)

from ablation_pipeline import run_one

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results'

print(f'[worker] cname={args.cname} si={args.si} seed={args.seed} '
      f'abl={args.abl_json} coh_thr={args.coh_thr} prefix={args.prefix}', flush=True)

abl_dict = json.loads(args.abl_json)
print(f'[worker] boost_scale={args.boost_scale}', flush=True)
res = run_one(args.seed, abl_dict, boost_scale=args.boost_scale, label=args.cname)

# Embed the coh_thr used into the result
for mode in res:
    res[mode]['coh_thr_used']     = args.coh_thr if args.coh_thr is not None else 0.5
    res[mode]['boost_scale_used'] = args.boost_scale

out = os.path.join(OUT_DIR, f'{args.prefix}_{args.cname}_seed{args.si}.pkl')
with open(out, 'wb') as f:
    pickle.dump(res, f)
print(f'[worker] saved -> {out}', flush=True)
