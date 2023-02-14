import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:
        s = next(f).strip()
        print('headers:', s)

        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])
            ts = float(e[2])
            label = int(e[3])
            feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            feat_l.append(feat)

    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)


def reindex(df, bipartite=False):
    df = df.sort_values(by='ts')

    dst_values = df.i.values
    if bipartite:
        dst_values += df.u.max() + 1

    all_ids = np.sort(np.unique(np.concatenate([df.u.values, dst_values])))
    n_2_idx = {n_id: idx + 1 for idx, n_id in enumerate(all_ids)}

    src_idx_l = [n_2_idx[id] for id in df.u]
    dst_idx_l = [n_2_idx[id] for id in dst_values]
    e_idx_l = list(range(1, df.shape[0] + 1))

    df = pd.DataFrame({
        'u': src_idx_l,
        'i': dst_idx_l,
        'ts': df.ts.values,
        'label': df.label.values,
        'idx': e_idx_l})

    return df


def run(name, data_dir, bipartite=False):
    PATH = str(data_dir / '{}.csv'.format(name))
    OUT_DF = str(data_dir / 'ml_{}.csv'.format(name))
    OUT_FEAT = str(data_dir / 'ml_{}.npy'.format(name))
    OUT_NODE_FEAT = str(data_dir / 'ml_{}_node.npy'.format(name))

    df, feat = preprocess(PATH)
    df = reindex(df, bipartite=bipartite)

    max_idx = max(df.u.max(), df.i.max())
    feat_dim = feat.shape[1]
    df.to_csv(OUT_DF)
    del df

    print('raw edge feat shape:', feat.shape)
    empty = np.zeros(feat_dim)[np.newaxis, :]
    feat = np.vstack([empty, feat])

    print('pad edge feat shape:', feat.shape)
    np.save(OUT_FEAT, feat)
    del feat

    node_feat = np.zeros((max_idx + 1, feat_dim))
    print('pad node feat shape:', node_feat.shape)
    np.save(OUT_NODE_FEAT, node_feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(Path(__file__).name)
    parser.add_argument('-d', '--data', type=str, required=True, help='dataset to process (e.g. snap-msg or jodie-wiki)')
    parser.add_argument('--dir', type=str, default='data', help='directory to load data files (default: data)')
    parser.add_argument('--bipartite', action='store_true', help='shift node ids for bipartite graphs')
    args = parser.parse_args()

    run(args.data, Path(args.dir), bipartite=args.bipartite)
