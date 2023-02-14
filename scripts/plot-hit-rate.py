import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


if len(sys.argv) < 3:
    print(f'usage: plot-hit-rate.py <top-dataset> <top.csv> <bot-dataset> <bot.csv>')
    sys.exit(1)

top_name = sys.argv[1]
top_csv = sys.argv[2]
bot_name = sys.argv[3]
bot_csv = sys.argv[4]

df_top = pd.read_csv(top_csv)
df_bot = pd.read_csv(bot_csv)

def calc_num_batches(df: pd.DataFrame):
    single_batch = df[df['batch'] == 1]
    per_batch = len(single_batch)
    assert (len(df) % per_batch == 0)
    return len(df) // per_batch

def running_avg(df: pd.DataFrame, n_batch: int, window: int):
    batch = np.zeros((n_batch, 3))
    for b, h, s in df.to_numpy():
        batch[b - 1, 0] += h
        batch[b - 1, 1] += s
    for b in range(n_batch):
        batch[b, 2] = batch[b, 0] / batch[b, 1]
    avgs = np.zeros(n_batch)
    for b in range(n_batch):
        s_idx = max(b - window + 1, 0)
        e_idx = b + 1
        avgs[b] = np.mean(batch[s_idx:e_idx, 2])
    return avgs

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rc('font', size=22)

fig = plt.figure(figsize=(16, 8), dpi=300)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

window = 10
n_batches = [calc_num_batches(df_top), calc_num_batches(df_bot)]

x1_pos = np.arange(n_batches[0])
x2_pos = np.arange(n_batches[1])
y1_avgs = running_avg(df_top, n_batches[0], window)
y2_avgs = running_avg(df_bot, n_batches[1], window)

ax1.plot(x1_pos, y1_avgs, label=top_name, linewidth=3, color='tab:orange')
ax2.plot(x2_pos, y2_avgs, label=bot_name, linewidth=3, color='tab:blue')
ax2.set_xlabel('batches')
ax2.set_ylabel('running avg hit rate')
ax2.yaxis.set_label_coords(-0.05, 1.0)
ax1.legend()
ax2.legend()

plt.savefig(f'plot-hit-rate.pdf', bbox_inches='tight', pad_inches=0.01)
print(f'saved: plot-hit-rate.pdf')
