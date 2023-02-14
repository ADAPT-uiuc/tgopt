import time

import numpy as np
import pandas as pd
import torch
from torch import Tensor

import tgopt_ext


class TGOpt(object):
    def __init__(self, enabled: bool, device='cpu',
                 dedup_targets=False, cache_embeds=False, precompute_time=False,
                 collect_hits=False):
        self.enabled = enabled
        self.enabled_dedup = dedup_targets
        self.enabled_cache = cache_embeds
        self.enabled_time = precompute_time
        self.collect_hits = collect_hits
        self.device = torch.device(device)

        self._t_dedup_filter = 0.0
        self._t_dedup_invert = 0.0
        self._t_time_encode_zero = 0.0
        self._t_time_encode_nghs = 0.0
        self._t_cache_keys = 0.0
        self._t_cache_lookup = 0.0
        self._t_cache_store = 0.0
        self._t_attn = 0.0
        self._c_hits = []

    ### Initialization and setup

    def init_cache(self, n_layers: int, feat_dim: int, limit: int):
        if self.enabled and self.enabled_cache:
            self._n_layers = n_layers
            self._feat_dim = feat_dim
            # key: (layer) node, ts -> val: embedding
            self._cache = [tgopt_ext.EmbedTable(limit) for _ in range(n_layers - 1)]

    def init_time(self, time_dim: int, time_window: int, encoder):
        if self.enabled and self.enabled_time:
            self._time_dim = time_dim
            self._time_window = time_window
            window = torch.arange(time_window + 1).float().to(self.device)
            self._time_embeds = encoder(window.view(-1, 1)).squeeze(dim=1)

    ### Deduplication

    def dedup_filter(self, src_l: np.ndarray, ts_l: np.ndarray):
        t_start = time.perf_counter()
        src_l, ts_l, inv_idx = tgopt_ext.dedup_src_ts(src_l, ts_l)
        self._t_dedup_filter += (time.perf_counter() - t_start)
        return src_l, ts_l, inv_idx

    def dedup_invert(self, embed: Tensor, inv_idx: np.ndarray):
        t_start = time.perf_counter()
        embed = embed[inv_idx]
        self._t_dedup_invert += (time.perf_counter() - t_start)
        return embed

    ### Time-encoding precomputation

    def get_time_zero_embed(self, num_delta: int) -> Tensor:
        t_start = time.perf_counter()
        output = self._time_embeds[0].repeat(num_delta, 1)
        output = output.view(-1, 1, self._time_dim)
        self._t_time_encode_zero += (time.perf_counter() - t_start)
        return output

    def compute_time_embed(self, ts_delta: Tensor, encoder):
        t_start = time.perf_counter()
        batch_size = ts_delta.shape[0]

        hit_count, hit_idx, out_embeds, ts_delta, inv_idx = \
            tgopt_ext.find_dedup_time_hits(ts_delta, self._time_embeds, self._time_window)

        uniq_size = ts_delta.shape[0]

        if hit_count != uniq_size:
            miss_idx = (~ hit_idx)
            ts_delta = ts_delta[miss_idx]
            miss_embeds = encoder(ts_delta.view(-1, 1)).squeeze(dim=1)
            out_embeds[miss_idx] = miss_embeds

        out_embeds = out_embeds[inv_idx]
        out_embeds = out_embeds.view(batch_size, -1, self._time_dim)

        self._t_time_encode_nghs += (time.perf_counter() - t_start)
        return out_embeds

    ### Caching/Memoization

    def cache_enabled_at(self, layer: int):
        # Only caching the intermediate layers
        return self.enabled and self.enabled_cache and layer < self._n_layers

    def compute_keys(self, src_l: np.ndarray, ts_l: np.ndarray):
        t_start = time.perf_counter()
        keys = tgopt_ext.compute_keys(src_l, ts_l)
        self._t_cache_keys += (time.perf_counter() - t_start)
        return keys

    def cache_lookup(self, layer: int, keys: np.ndarray):
        t_start = time.perf_counter()
        table = self._cache[layer - 1]
        hit_idx, embeds = table.lookup(keys, self._feat_dim, self.device)
        self._t_cache_lookup += (time.perf_counter() - t_start)
        return hit_idx, embeds

    def cache_store(self, layer: int, keys: np.ndarray, embeds: Tensor):
        t_start = time.perf_counter()
        table = self._cache[layer - 1]
        table.store(keys, embeds)
        self._t_cache_store += (time.perf_counter() - t_start)

    def cache_sizes(self):
        sizes = []
        for table in self._cache:
            sizes.append(table.size_in_bytes())
        return sizes

    ### Collecting hit statistics

    def prep_next_batch(self):
        self._c_hits.append([])

    def record_batch_hits(self, hits, size):
        self._c_hits[-1].append((hits, size))

    def get_batch_hits_df(self):
        c_hits = []
        c_sizes = []
        c_batch = []
        for i, b in enumerate(self._c_hits):
            for h, s in b:
                c_hits.append(h)
                c_sizes.append(s)
                c_batch.append(i + 1)
        df = pd.DataFrame({
            'batch': np.array(c_batch),
            'hits': np.array(c_hits),
            'sizes': np.array(c_sizes)})
        return df


class NeighborFinder:
    """Simple temporal graph representation and neighborhood sampler."""

    def __init__(self, adj_list):
        self.node_to_nghs = []
        self.node_to_eidx = []
        self.node_to_time = []
        self._t_ngh_lookup = 0.0

        by_timestamp = lambda x: x[2]
        for neighbors in adj_list:
            # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
            sorted_neighhbors = sorted(neighbors, key=by_timestamp)
            self.node_to_nghs.append(np.array([x[0] for x in sorted_neighhbors], dtype=np.int32))
            self.node_to_eidx.append(np.array([x[1] for x in sorted_neighhbors], dtype=np.int32))
            self.node_to_time.append(np.array([x[2] for x in sorted_neighhbors], dtype=np.float32))

    def ngh_lookup(self, src_l: np.ndarray, ts_l: np.ndarray, n_ngh=20):
        assert (len(src_l) == len(ts_l))

        t_start = time.perf_counter()

        nghs_l, eidx_l, time_l = [], [], []
        for src_idx in src_l:
            nghs_l.append(self.node_to_nghs[src_idx])
            eidx_l.append(self.node_to_eidx[src_idx])
            time_l.append(self.node_to_time[src_idx])

        out_nghs, out_eidx, out_time = tgopt_ext.sample_recent_ngh(
                n_ngh, ts_l, nghs_l, eidx_l, time_l)

        self._t_ngh_lookup += (time.perf_counter() - t_start)

        return out_nghs, out_eidx, out_time
