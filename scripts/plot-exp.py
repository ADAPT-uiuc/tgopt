import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


if len(sys.argv) < 4:
    print(f'usage: plot-exp.py <cpu | gpu> <base.csv> <opt.csv>')
    sys.exit(1)

dev = sys.argv[1]
base_csv = sys.argv[2]
opt_csv = sys.argv[3]

df_base = pd.read_csv(base_csv)
df_opt = pd.read_csv(opt_csv)
speedup = (df_base['avg'] / df_opt['avg']).to_numpy()

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rc('font', size=22)

fig = plt.figure(figsize=(16, 8), dpi=300)
width = 0.45

x_txt = list(df_base['dataset'])
x_pos = np.arange(len(x_txt))
y_base_avg = df_base['avg'].to_numpy()
y_base_err = df_base['std'].to_numpy()
y_ours_avg = df_opt['avg'].to_numpy()
y_ours_err = df_opt['std'].to_numpy()

bar_labels = [f'{x:.1f}x' for x in speedup]
err_kw = {'elinewidth': 2, 'capsize': 4}

ax1 = fig.add_subplot(111)
ax1.bar(x_pos, y_base_avg, width=width, yerr=y_base_err, error_kw=err_kw, label='baseline *', color='tab:orange')
bar_ours = ax1.bar(x_pos + width, y_ours_avg, width=width, yerr=y_ours_err, error_kw=err_kw, label='TGOpt', color='tab:blue')
ax1.set_xticks(x_pos + width / 2, x_txt)
ax1.tick_params(axis='x', labelrotation=15)

ax1.bar_label(bar_ours, labels=bar_labels)
ax1.set_ylabel(f'avg {dev} runtime (secs)')
ax1.legend(loc='upper center', ncol=2)

plt.savefig(f'plot-exp-{dev}.pdf', bbox_inches='tight', pad_inches=0.01)
print(f'saved: plot-exp-{dev}.pdf')
