import logging
import time

import numpy as np
import torch
import torch.nn as nn

from tgopt import TGOpt


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class TimeEncode(torch.nn.Module):
    def __init__(self, time_dim, factor=5):
        super().__init__()
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1) # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]

        output = torch.bmm(attn, v)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class AttnModel(torch.nn.Module):
    """Attention based temporal layers"""

    def __init__(self, feat_dim, edge_dim, time_dim, n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super().__init__()

        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.model_dim = (feat_dim + edge_dim + time_dim)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        assert(self.model_dim % n_head == 0)
        self.multi_head_target = MultiHeadAttention(n_head,
                                    d_model=self.model_dim,
                                    d_k=self.model_dim // n_head,
                                    d_v=self.model_dim // n_head,
                                    dropout=drop_out)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.
        returns:
          output: float Tensor of shape [B, D]
        """

        src_ext = torch.unsqueeze(src, dim=1) # src [B, 1, D]
        src_e_ph = torch.zeros_like(src_ext)
        q = torch.cat([src_ext, src_e_ph, src_t], dim=2) # [B, 1, D + De + Dt] -> [B, 1, D]
        k = torch.cat([seq, seq_e, seq_t], dim=2) # [B, 1, D + De + Dt] -> [B, 1, D]

        mask = torch.unsqueeze(mask, dim=2) # mask [B, N, 1]
        mask = mask.permute([0, 2, 1]) #mask [B, 1, N]

        output = self.multi_head_target(q=q, k=k, v=k, mask=mask) # output: [B, 1, D + Dt], attn: [B, 1, N]
        output = output.squeeze()

        if len(output.shape) == 1:
            output = output.view(1, -1)

        output = self.merger(output, src)
        return output


class TGAN(torch.nn.Module):
    def __init__(self, ngh_finder, n_feat, e_feat,
                 num_layers=2, num_heads=2, null_idx=0, drop_out=0.1):
        super().__init__()

        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)

        self.feat_dim = self.n_feat_th.shape[1]
        self.n_feat_dim = self.feat_dim
        self.e_feat_dim = self.feat_dim
        self.model_dim = self.feat_dim

        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)

        self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim,
                                                    self.feat_dim,
                                                    self.feat_dim,
                                                    n_head=num_heads,
                                                    drop_out=drop_out) for _ in range(num_layers)])

        self.time_encoder = TimeEncode(time_dim=self.n_feat_th.shape[1])

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1)

        # Note: for now this will be configured outside of class.
        self._opt = TGOpt(False)

    def forward(self, src_idx_l, target_idx_l, cut_time_l, n_ngh=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, n_ngh)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, n_ngh)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        return score

    def contrast(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l, n_ngh=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, n_ngh)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, n_ngh)
        background_embed = self.tem_conv(background_idx_l, cut_time_l, self.num_layers, n_ngh)
        pos_score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed, background_embed).squeeze(dim=-1)
        return pos_score.sigmoid(), neg_score.sigmoid()

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, n_ngh=20):
        assert(curr_layers >= 0)
        self.device = self.n_feat_th.device
        if not self.training and self._opt.enabled:
            return self._opt_tem_conv(src_idx_l, cut_time_l, curr_layers, n_ngh)
        else:
            return self._base_tem_conv(src_idx_l, cut_time_l, curr_layers, n_ngh)

    def _base_tem_conv(self, src_idx_l, cut_time_l, curr_layers, n_ngh=20):
        if curr_layers == 0:
            src_node_batch_th = torch.from_numpy(src_idx_l).long().to(self.device)
            src_node_feat = self.node_raw_embed(src_node_batch_th)
            return src_node_feat
        else:
            batch_size = len(src_idx_l)

            src_node_conv_feat = self._base_tem_conv(src_idx_l,
                                           cut_time_l,
                                           curr_layers=curr_layers - 1,
                                           n_ngh=n_ngh)

            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.ngh_lookup(
                                                                    src_idx_l,
                                                                    cut_time_l,
                                                                    n_ngh=n_ngh)

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(self.device)
            src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(self.device)

            #torch.cuda.synchronize()
            t_start = time.perf_counter()
            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(self.device)
            #torch.cuda.synchronize()
            self._opt._t_time_encode_nghs += (time.perf_counter() - t_start)

            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten() #reshape(batch_size, -1)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten() #reshape(batch_size, -1)
            src_ngh_node_conv_feat = self._base_tem_conv(src_ngh_node_batch_flat,
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1,
                                                   n_ngh=n_ngh)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, n_ngh, -1)

            # get edge time features and node features
            #torch.cuda.synchronize()
            t_start = time.perf_counter()
            cut_time_l_th = torch.from_numpy(cut_time_l).float().to(self.device)
            cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
            src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
            #torch.cuda.synchronize()
            self._opt._t_time_encode_zero += (time.perf_counter() - t_start)

            t_start = time.perf_counter()
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            #torch.cuda.synchronize()
            self._opt._t_time_encode_nghs += (time.perf_counter() - t_start)

            src_ngh_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]

            #torch.cuda.synchronize()
            t_start = time.perf_counter()
            local = attn_m(src_node_conv_feat,
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed,
                                   src_ngh_edge_feat,
                                   mask)
            #torch.cuda.synchronize()
            self._opt._t_attn += (time.perf_counter() - t_start)

            return local

    ### Optimized implementation

    def _opt_tem_conv(self, src_l: np.ndarray, ts_l: np.ndarray, layer: int, n_ngh=20):
        if self._opt.enabled_dedup and layer > 0:
            src_l, ts_l, inv_idx = self._opt.dedup_filter(src_l, ts_l)
            embed = self._compute_embed(src_l, ts_l, layer, n_ngh)
            return self._opt.dedup_invert(embed, inv_idx)
        else:
            return self._compute_embed(src_l, ts_l, layer, n_ngh)

    def _compute_embed(self, src_l: np.ndarray, ts_l: np.ndarray, layer: int, n_ngh: int):
        if layer == 0:
            src_l = torch.from_numpy(src_l).long().to(self.device)
            src_node_feat = self.node_raw_embed(src_l)
            return src_node_feat
        else:
            batch_size = len(src_l)

            if self._opt.cache_enabled_at(layer):
                keys = self._opt.compute_keys(src_l, ts_l)
                hit_idx, embeds = self._opt.cache_lookup(layer, keys)

                hit_count = torch.sum(hit_idx).item()
                if self._opt.collect_hits:
                    self._opt.record_batch_hits(hit_count, batch_size)
                if hit_count == batch_size:
                    # All hits, return the cached embeds
                    return embeds

                miss_idx = (~ hit_idx)
                miss_idx_np = miss_idx.cpu().numpy()
                del hit_idx

                if hit_count != 0:
                    # If not all misses (aka some hits), then filter down the lists
                    src_l = src_l[miss_idx_np]
                    ts_l = ts_l[miss_idx_np]
                    keys = keys[miss_idx_np]
                    batch_size = len(keys)

            ### If not using cache or has misses, then do computations

            # Directly call embed fn since we don't need to dedup src_l again
            src_embed = self._compute_embed(src_l, ts_l, layer=layer - 1, n_ngh=n_ngh)

            ngh_batch, ngh_eidx_batch, ngh_ts_batch = self.ngh_finder.ngh_lookup(
                                                            src_l, ts_l, n_ngh=n_ngh)

            # Call higher-level embed fn to dedup neighbors
            ngh_embed = self._opt_tem_conv(ngh_batch.flatten(), ngh_ts_batch.flatten(),
                                           layer=layer - 1, n_ngh=n_ngh)
            ngh_embed = ngh_embed.view(batch_size, n_ngh, -1)

            ### Compute attention aggregation for filtered inputs

            #torch.cuda.synchronize()
            t_start = time.perf_counter()
            ts_l = ts_l.reshape(-1, 1)
            ngh_ts_delta = ts_l - ngh_ts_batch
            ngh_ts_delta = torch.from_numpy(ngh_ts_delta).float().to(self.device)
            #torch.cuda.synchronize()
            self._opt._t_time_encode_nghs += (time.perf_counter() - t_start)

            if self._opt.enabled_time:
                src_t_embed = self._opt.get_time_zero_embed(batch_size)
                ngh_t_embed = self._opt.compute_time_embed(ngh_ts_delta, self.time_encoder)
            else:
                ### Otherwise, do the original code path
                t_start = time.perf_counter()
                ts_l = torch.from_numpy(ts_l).float().to(self.device)
                src_t_embed = self.time_encoder(torch.zeros_like(ts_l))
                #torch.cuda.synchronize()
                self._opt._t_time_encode_zero += (time.perf_counter() - t_start)

                t_start = time.perf_counter()
                ngh_t_embed = self.time_encoder(ngh_ts_delta)
                #torch.cuda.synchronize()
                self._opt._t_time_encode_nghs += (time.perf_counter() - t_start)

            ngh_eidx_batch = torch.from_numpy(ngh_eidx_batch).long().to(self.device)
            ngh_edge_feat = self.edge_raw_embed(ngh_eidx_batch)

            ngh_batch = torch.from_numpy(ngh_batch).long().to(self.device)
            mask = ngh_batch == 0
            attn_m = self.attn_model_list[layer - 1]

            del ts_l
            del src_l
            del ngh_batch
            del ngh_eidx_batch
            del ngh_ts_batch
            del ngh_ts_delta

            #torch.cuda.synchronize()
            t_start = time.perf_counter()
            local_embed = attn_m(src_embed, src_t_embed,
                                    ngh_embed, ngh_t_embed,
                                    ngh_edge_feat, mask)
            #torch.cuda.synchronize()
            self._opt._t_attn += (time.perf_counter() - t_start)

            if self._opt.cache_enabled_at(layer):
                self._opt.cache_store(layer, keys, local_embed)
                embeds[miss_idx] = local_embed
                return embeds
            else:
                ### If not using cache, then nothing else to do
                return local_embed
