import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


def print_usage():
    print(f'usage: plot-ablation.py [--merge] <cpu | gpu> <base.csv> <ab1.csv> <ab2.csv> <ab3.csv>')


if len(sys.argv) < 6:
    print_usage()
    sys.exit(1)

do_merge = False
if '--merge' in sys.argv:
    do_merge = True
    sys.argv.remove('--merge')
    if len(sys.argv) < 6:
        print_usage()
        sys.exit(1)

dev = sys.argv[1]
base_csv = sys.argv[2]
ab1_csv = sys.argv[3]
ab2_csv = sys.argv[4]
ab3_csv = sys.argv[5]

df_base = pd.read_csv(base_csv)
df_ab1 = pd.read_csv(ab1_csv)
df_ab2 = pd.read_csv(ab2_csv)
df_ab3 = pd.read_csv(ab3_csv)

if do_merge:
    df = pd.DataFrame()
    df['dataset'] = df_base['dataset']
    df['base'] = df_base['avg']
    df['ab1'] = df_ab1['avg']
    df['ab2'] = df_ab2['avg']
    df['ab3'] = df_ab3['avg']
    Path('logs').mkdir(parents=True, exist_ok=True)
    df.to_csv(f'logs/ab-{dev}.csv', index=False)
    sys.exit(0)

###

speedup1 = (df_base['avg'] / df_ab1['avg']).to_numpy()
speedup2 = (df_base['avg'] / df_ab2['avg']).to_numpy()
speedup3 = (df_base['avg'] / df_ab3['avg']).to_numpy()
xup_base = np.ones_like(speedup1)

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rc('font', size=22)

fig = plt.figure(figsize=(16, 8), dpi=300)
ax1 = fig.add_subplot(111)
width = 0.2

labels = ['base', 'cache', 'cache+dedup', 'cache+dedup+time']
x_txt = list(df_base['dataset'])
x_pos = np.arange(len(x_txt))

ax1.bar(x_pos, xup_base, width=width, label=labels[0], color='tab:orange')
ax1.bar(x_pos + width, speedup1, width=width, label=labels[1], color='tab:blue')
ax1.bar(x_pos + width * 2, speedup2, width=width, label=labels[2], color='tab:green')
ax1.bar(x_pos + width * 3, speedup3, width=width, label=labels[3], color='tab:red')
ax1.set_xticks(x_pos + width * 1.5, x_txt)
ax1.set_ylabel(f'{dev} speedup (x)')
ax1.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2))

plt.savefig(f'plot-ablation-{dev}.pdf', bbox_inches='tight', pad_inches=0.01)
print(f'saved: plot-ablation-{dev}.pdf')
