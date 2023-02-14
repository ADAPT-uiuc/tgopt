"""Script for initial preprocessing of datasets to match format for TGAT."""

import argparse
import sys
from datetime import datetime as dt
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


HEADERS = 'user_id,item_id,timestamp,state_label,comma_separated_list_of_features'


def rand_edge_feats(n_edges: int, dim: int) -> np.ndarray:
    print('generated random edge features')
    return np.random.randn(n_edges, dim)


def read_snap_reddit_tsv(fpath: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []

    with fpath.open() as f:
        next(f)
        for line in f:
            e = line.strip().split('\t')
            u = str(e[0])
            i = str(e[1])
            ts = dt.fromisoformat(e[3]).timestamp()
            label = int(e[4])

            feat = e[5].split(',')
            feat = np.array([float(x) for x in feat])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            feat_l.append(feat)

    df = pd.DataFrame({
            'u': u_list,
            'i': i_list,
            'ts': ts_list,
            'label': label_list})
    return df, np.array(feat_l)


def reformat_snap_reddit(data_dir: Path):
    df_title, ft_title = read_snap_reddit_tsv(data_dir / 'reddit-hyperlinks-title.tsv')
    df_body, ft_body = read_snap_reddit_tsv(data_dir / 'reddit-hyperlinks-body.tsv')

    df_full = pd.concat([df_title, df_body])
    del df_title
    del df_body

    ft_full = pd.DataFrame(np.concatenate([ft_title, ft_body]))
    del ft_title
    del ft_body

    sub_names = set()
    for sub in df_full.u.values:
        sub_names.add(sub)
    for sub in df_full.i.values:
        sub_names.add(sub)
    sub_names = sorted(sub_names)
    subreddits = {n: i for i, n in enumerate(sub_names)}
    del sub_names

    edges = np.zeros((df_full.shape[0], 4))
    for idx, r in enumerate(df_full.itertuples()):
        src_id = subreddits[str(r.u)]
        dst_id = subreddits[str(r.i)]
        edges[idx, :] = [src_id, dst_id, r.ts, r.label]
    del subreddits

    df = pd.DataFrame(edges, columns=['u', 'i', 'ts', 'label'])
    df = df.astype({'u': int, 'i': int, 'ts': float, 'label': int})
    del edges

    df = pd.concat([df, ft_full], axis=1)
    save_csv(data_dir / 'snap-reddit.csv', df)


def reformat_snap_txt(in_fpath: Path, out_fpath: Path, rand_dim: int):
    df = pd.read_csv(in_fpath, sep=' ', header=None, names=['u', 'i', 'ts'])
    df['label'] = np.zeros(df.shape[0], dtype=int)
    feat = pd.DataFrame(rand_edge_feats(df.shape[0], rand_dim))
    df = pd.concat([df, feat], axis=1)
    del feat
    save_csv(out_fpath, df)


def reformat_jodie_lastfm(data_dir: Path, rand_dim: int):
    fpath = data_dir / 'lastfm.csv'
    df = pd.read_csv(fpath, index_col=False, header=0, usecols=[0, 1, 2, 3])
    feat = pd.DataFrame(rand_edge_feats(df.shape[0], rand_dim))
    df = pd.concat([df, feat], axis=1)
    del feat
    save_csv(data_dir / 'jodie-lastfm.csv', df)


def save_csv(fpath: Path, df: pd.DataFrame):
    with fpath.open('w') as f:
        f.write(f'{HEADERS}\n')
    df.to_csv(str(fpath), index=False, header=False, mode='a')
    print('saved:', fpath)


def run(name: str, data_dir: Path, rand_dim: int):
    if name == 'jodie-lastfm':
        reformat_jodie_lastfm(data_dir, rand_dim)
    elif name == 'jodie-mooc':
        print('nothing to do')
    elif name == 'jodie-wiki':
        print('nothing to do')
    elif name == 'jodie-reddit':
        print('nothing to do')
    elif name == 'snap-msg':
        inp_fpath = data_dir / 'college-msg.txt'
        out_fpath = data_dir / 'snap-msg.csv'
        reformat_snap_txt(inp_fpath, out_fpath, rand_dim)
    elif name == 'snap-email':
        inp_fpath = data_dir / 'email-eu-temporal.txt'
        out_fpath = data_dir / 'snap-email.csv'
        reformat_snap_txt(inp_fpath, out_fpath, rand_dim)
    elif name == 'snap-reddit':
        reformat_snap_reddit(data_dir)
    else:
        print('dataset not yet supported:', args.data)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(Path(__file__).name)
    parser.add_argument('-d', '--data', type=str, required=True, help='dataset to reformat (e.g. snap-msg or jodie-wiki)')
    parser.add_argument('--dir', type=str, default='data', help='directory to load/save data files (default: data)')
    parser.add_argument('--rand-dim', type=int, default=100, help='dimension to use if generating random edge features (default: 100)')
    parser.add_argument('--seed', type=int, default=-1, help='seed to use when doing random (default: -1 to not set)')
    args = parser.parse_args()

    if args.seed >= 0:
        np.random.seed(args.seed)

    run(args.data, Path(args.dir), args.rand_dim)
