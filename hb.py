import time, os, datetime

BASE = r'C:\Users\Admin\brain-organoid-rl\mod_results'
CSVS = {
    'M1': 'mod1_scaling_results.csv',
    'M5': 'mod5_bio_param_sweep.csv',
    'M4': 'mod4_istdp_results.csv',
    'M3': 'mod3_learned_schema_15seeds.csv',
}
LOG3 = os.path.join(BASE, 'mod3.log')

def count_rows(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return max(0, sum(1 for _ in f) - 1)
    except:
        return '?'

def last_line(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines[-1] if lines else ''
    except:
        return ''

while True:
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    counts = {k: count_rows(os.path.join(BASE, v)) for k, v in CSVS.items()}
    l3 = last_line(LOG3)
    print(f"BEAT {ts} | M1={counts['M1']} M5={counts['M5']} M4={counts['M4']} M3={counts['M3']} | mod3: {l3}", flush=True)
    time.sleep(120)
