# This script generates plot of ablation results for both CPU and GPU.
#
# The ablation results will need to be tabulated into a single csv file, one
# for CPU and one for GPU. It should contain the average runtimes with these
# headers:
#
#   dataset,base,ab1,ab2,ab3
#
# The two csv files should have the same datasets in the same order. You may
# generate this csv using the other script:
#
#   python scripts/plot-ablation.py --merge ...
#

import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


if len(sys.argv) < 3:
    print(f'usage: plot-ablation-both.py <cpu.csv> <gpu.csv>')
    sys.exit(1)

cpu_csv = sys.argv[1]
gpu_csv = sys.argv[2]

df_cpu = pd.read_csv(cpu_csv)
df_gpu = pd.read_csv(gpu_csv)

def calc_speedup(df: pd.DataFrame):
    arr = df.iloc[:, 1:].to_numpy()
    xup = arr[:, 0].repeat(arr.shape[1])
    xup = xup.reshape(arr.shape[0], -1) / arr
    xup[:, 0] = 1
    return xup

xup_cpu = calc_speedup(df_cpu)
xup_gpu = calc_speedup(df_gpu)

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rc('font', size=22)

fig = plt.figure(figsize=(16, 10), dpi=300)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
width = 0.2

labels = ['base', 'cache', 'cache+dedup', 'cache+dedup+time']
x_txt = list(df_cpu.dataset)
x_pos = np.arange(len(x_txt))

ax1.bar(x_pos, xup_cpu[:,0], width=width, label=labels[0], color='tab:orange')
ax1.bar(x_pos + width, xup_cpu[:,1], width=width, label=labels[1], color='tab:blue')
ax1.bar(x_pos + width * 2, xup_cpu[:,2], width=width, label=labels[2], color='tab:green')
ax1.bar(x_pos + width * 3, xup_cpu[:,3], width=width, label=labels[3], color='tab:red')
ax1.set_xticks(x_pos + width * 1.5, x_txt)
ax1.set_ylabel(f'cpu speedup (x)')

ax2.bar(x_pos, xup_gpu[:,0], width=width, label=labels[0], color='tab:orange')
ax2.bar(x_pos + width, xup_gpu[:,1], width=width, label=labels[1], color='tab:blue')
ax2.bar(x_pos + width * 2, xup_gpu[:,2], width=width, label=labels[2], color='tab:green')
ax2.bar(x_pos + width * 3, xup_gpu[:,3], width=width, label=labels[3], color='tab:red')
ax2.set_xticks(x_pos + width * 1.5, x_txt)
ax2.set_ylabel(f'gpu speedup (x)')
ax2.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.4))

plt.savefig(f'plot-ablation-both.pdf', bbox_inches='tight', pad_inches=0.01)
print(f'saved: plot-ablation-both.pdf')
