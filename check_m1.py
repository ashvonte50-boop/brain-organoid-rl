import pandas as pd, os
f = r'C:\Users\Admin\brain-organoid-rl\m1_results\m1_task105_20seeds.csv'
if os.path.exists(f):
    df = pd.read_csv(f)
    print(len(df))
else:
    print(0)
