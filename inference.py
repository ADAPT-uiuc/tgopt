import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from module import TGAN
from tgopt import TGOpt, NeighborFinder


# NOTE: for more accurate stats/timings when running with GPU, uncomment the
# `synchronize()` calls in `inference.py`, `module.py`, and `tgopt.py`.


### Argument and global variables
parser = argparse.ArgumentParser(Path(__file__).name)
parser.add_argument('-d', '--data', type=str, required=True, help='dataset to use (e.g. snap-msg or jodie-wiki)')
parser.add_argument('--model', type=str, required=True, help='prefix for loading saved model')
parser.add_argument('--prefix', type=str, required=True, help='prefix for this inference run')
parser.add_argument('--dir', type=str, default='data', help='directory to load data files (default: data)')
parser.add_argument('--bs', type=int, default=200, help='batch size (default: 200)')
parser.add_argument('--runs', type=int, default=1, help='number of runs (default: 1)')
parser.add_argument('--n-degree', type=int, default=20, help='number of neighbors to sample (default: 20)')
parser.add_argument('--n-layer', type=int, default=2, help='number of network layers (default: 2)')
parser.add_argument('--n-head', type=int, default=2, help='number of heads used in attention layer (default: 2)')
parser.add_argument('--gpu', type=int, default=-1, help='idx for the gpu to use (default: -1 for cpu)')
parser.add_argument('--save-embeds', action='store_true', help='save embeddings for each batch')
parser.add_argument('--opt-all', action='store_true', help='enable all optimizations')
parser.add_argument('--opt-dedup', action='store_true', help='enable deduplication optimization')
parser.add_argument('--opt-cache', action='store_true', help='enable caching optimization')
parser.add_argument('--opt-time', action='store_true', help='enable precomputing time encodings')
parser.add_argument('--cache-limit', type=int, default=int(2e6), help='max number of embeds to cache (default: 2e6)')
parser.add_argument('--time-window', type=int, default=int(1e4), help='time window to precompute (default: 1e4)')
parser.add_argument('--csv', type=str, default='', help="csv file to write avg/std into")
parser.add_argument('--stats', action='store_true', help="enable printing of more detailed stats")
args = parser.parse_args()

DATA = args.data
BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_LAYER = args.n_layer
NUM_HEADS = args.n_head
GPU = args.gpu

ENABLE_OPTS = (args.opt_all or args.opt_dedup or args.opt_cache or args.opt_time)

Path('./logs').mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.model}-{args.data}.pth'
if args.save_embeds:
    Path('./saved_embeds').mkdir(parents=True, exist_ok=True)
    get_embed_path = lambda batch: f'./saved_embeds/{args.prefix}-{args.data}-{batch}.pth'
data_dir = Path(args.dir)


### Set up logger
log_time = int(time.time())
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('logs/infer-{}-{}-{}.log'.format(args.prefix, args.data, str(log_time)))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


### Load data and build graph
g_df = pd.read_csv(data_dir / 'ml_{}.csv'.format(DATA))
e_feat = np.load(data_dir / 'ml_{}.npy'.format(DATA))
n_feat = np.load(data_dir / 'ml_{}_node.npy'.format(DATA))

src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
ts_l = g_df.ts.values

max_idx = max(src_l.max(), dst_l.max())
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list)

del full_adj_list
del e_idx_l
del g_df


### Run inference

def init_opt(args, model: TGAN) -> TGOpt:
    opt = TGOpt(False)
    if ENABLE_OPTS:
        do_dedup = args.opt_dedup or args.opt_all
        do_cache = args.opt_cache or args.opt_all
        do_time = args.opt_time or args.opt_all
        opt = TGOpt(True, device=device,
                        dedup_targets=do_dedup,
                        cache_embeds=do_cache,
                        precompute_time=do_time,
                        collect_hits=args.stats)
        if do_cache:
            opt.init_cache(
                    n_layers=NUM_LAYER,
                    feat_dim=n_feat.shape[1],
                    limit=args.cache_limit)
        if do_time:
            opt.init_time(
                    time_dim=n_feat.shape[1],
                    time_window=args.time_window,
                    encoder=model.time_encoder)
    model._opt = opt
    return opt


device = torch.device('cuda:{}'.format(GPU) if GPU >= 0 else 'cpu')

num_instance = len(src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)
logger.info('num of instances: {}'.format(num_instance))
logger.info('num of batches: {}'.format(num_batch))

runtimes = []
for r in range(args.runs):
    logger.info('start run {}'.format(r + 1))

    model = TGAN(full_ngh_finder, n_feat, e_feat,
            num_layers=NUM_LAYER, num_heads=NUM_HEADS)

    state = torch.load(MODEL_SAVE_PATH, map_location=device)
    state['n_feat_th'] = model.n_feat_th
    state['e_feat_th'] = model.e_feat_th
    state['node_raw_embed.weight'] = model.n_feat_th
    state['edge_raw_embed.weight'] = model.e_feat_th
    model.load_state_dict(state)
    del state

    model = model.to(device)
    model = model.eval()

    opt = init_opt(args, model)

    t_total = 0
    for k in range(num_batch):
        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance, s_idx + BATCH_SIZE)

        node_l_cut = np.concatenate([src_l[s_idx:e_idx], dst_l[s_idx:e_idx]])
        time_l_cut = np.concatenate([ts_l[s_idx:e_idx], ts_l[s_idx:e_idx]])

        node_l_cut = node_l_cut.astype(np.int32)
        time_l_cut = time_l_cut.astype(np.float32)

        if args.stats and opt.enabled_cache:
            opt.prep_next_batch()

        with torch.no_grad():
            #torch.cuda.synchronize()
            t_start = time.perf_counter()
            embed = model.tem_conv(node_l_cut, time_l_cut, NUM_LAYER, n_ngh=NUM_NEIGHBORS)
            #torch.cuda.synchronize()
            t_total += (time.perf_counter() - t_start)

        if args.save_embeds:
            torch.save(embed.cpu(), get_embed_path(k))
        del embed

    if args.stats:
        logger.info(f't_ngh_lookup: {full_ngh_finder._t_ngh_lookup} secs')
        logger.info(f't_dedup_filter: {opt._t_dedup_filter} secs')
        logger.info(f't_dedup_invert: {opt._t_dedup_invert} secs')
        logger.info(f't_time_encode_zero: {opt._t_time_encode_zero} secs')
        logger.info(f't_time_encode_nghs: {opt._t_time_encode_nghs} secs')
        logger.info(f't_cache_keys:   {opt._t_cache_keys} secs')
        logger.info(f't_cache_lookup: {opt._t_cache_lookup} secs')
        logger.info(f't_cache_store:  {opt._t_cache_store} secs')
        logger.info(f't_attn: {opt._t_attn} secs')

        if opt.enabled_cache:
            for i, bytes in enumerate(opt.cache_sizes()):
                mibs = bytes / 1024.0 / 1024.0
                logger.info(f'layer {i + 1} cache table size: {mibs} MiBs')

            batch_df = opt.get_batch_hits_df()
            batch_df.to_csv(f'logs/{args.prefix}-{args.data}-hits.csv', index=False)
            batch_avg = batch_df.hits.sum() / batch_df.sizes.sum()
            logger.info(f'avg hit rate: {batch_avg}')
            del batch_df

    logger.info(f'inference total elapsed: {t_total} secs')
    runtimes.append(t_total)


runtimes = np.array(runtimes)
avg = runtimes.mean()
std = runtimes.std()
logger.info(f'average runtime: {avg} +/- {std} secs')

if args.csv:
    mode = 'a' if Path(args.csv).is_file() else 'w'
    with Path(args.csv).open(mode) as f:
        f.write(f'{DATA},{avg},{std}\n')
