#include <tbb/concurrent_unordered_map.h>
#include <torch/extension.h>
#include <pybind11/numpy.h>
#include <omp.h>

#include <deque>

py::tuple sample_recent_ngh(ssize_t n_ngh,
                            py::array_t<float> &ts_l,
                            std::vector<py::array_t<int32_t>> &nghs_l,
                            std::vector<py::array_t<int32_t>> &eidx_l,
                            std::vector<py::array_t<float>> &time_l) {
  ssize_t size = ts_l.size();

  auto out_nghs = py::array_t<int32_t>({size, n_ngh});
  auto out_eidx = py::array_t<int32_t>({size, n_ngh});
  auto out_time = py::array_t<float>({size, n_ngh});

  auto ts_buf = ts_l.request();
  auto out_nghs_buf = out_nghs.request();
  auto out_eidx_buf = out_eidx.request();
  auto out_time_buf = out_time.request();

  auto *ts_ptr = static_cast<float *>(ts_buf.ptr);
  auto *out_nghs_ptr = static_cast<int32_t *>(out_nghs_buf.ptr);
  auto *out_eidx_ptr = static_cast<int32_t *>(out_eidx_buf.ptr);
  auto *out_time_ptr = static_cast<float *>(out_time_buf.ptr);

  // Zero out all the arrays to comply with masking.
  for (ssize_t i = 0; i < size * n_ngh; i++) {
    out_nghs_ptr[i] = 0;
    out_eidx_ptr[i] = 0;
    out_time_ptr[i] = 0.0;
  }

  std::vector<int32_t *> nghs_ptr_l(size);
  std::vector<int32_t *> eidx_ptr_l(size);
  std::vector<float *> time_ptr_l(size);

  // Need to request pointers sequentially.
  for (ssize_t i = 0; i < size; i++) {
    auto nghs_buf = nghs_l[i].request();
    auto eidx_buf = eidx_l[i].request();
    auto time_buf = time_l[i].request();

    auto *nghs_ptr = static_cast<int32_t *>(nghs_buf.ptr);
    auto *eidx_ptr = static_cast<int32_t *>(eidx_buf.ptr);
    auto *time_ptr = static_cast<float *>(time_buf.ptr);

    nghs_ptr_l[i] = nghs_ptr;
    eidx_ptr_l[i] = eidx_ptr;
    time_ptr_l[i] = time_ptr;
  }

  #ifndef tgopt_1t_sampling
  #pragma omp parallel for
  #endif
  for (ssize_t i = 0; i < size; i++) {
    size_t time_size = time_l[i].size();

    if (time_size > 0) {
      float cut_time = ts_ptr[i];
      auto *time_ptr = time_ptr_l[i];

      std::vector<float> ts_vec(time_ptr, time_ptr + time_size);
      auto low = std::lower_bound(ts_vec.begin(), ts_vec.end(), cut_time);
      ssize_t last = std::distance(ts_vec.begin(), low);

      if (last > 0) {
        auto *nghs_ptr = nghs_ptr_l[i];
        auto *eidx_ptr = eidx_ptr_l[i];

        ssize_t first = last > n_ngh ? last - n_ngh : 0;
        ssize_t start = n_ngh - (last - first);
        ssize_t idx = i * n_ngh + start;

        while (first < last) {
          out_nghs_ptr[idx] = nghs_ptr[first];
          out_eidx_ptr[idx] = eidx_ptr[first];
          out_time_ptr[idx] = time_ptr[first];
          first++;
          idx++;
        }
      }
    }
  }

  return py::make_tuple(out_nghs, out_eidx, out_time);
}

/// Custom hash function for collision-free keys.
int64_t opt_hash(int32_t s, float t) {
  return (static_cast<int64_t>(s) << 32) | static_cast<int32_t>(t);
}

py::tuple dedup_src_ts(py::array_t<int32_t> &src_l, py::array_t<float> &ts_l) {
  ssize_t size = src_l.size();
  auto inv_idx = py::array_t<int32_t>(size);

  auto s_buf = src_l.request();
  auto t_buf = ts_l.request();
  auto i_buf = inv_idx.request();

  auto *s_ptr = static_cast<int32_t *>(s_buf.ptr);
  auto *t_ptr = static_cast<float *>(t_buf.ptr);
  auto *i_ptr = static_cast<int32_t *>(i_buf.ptr);

  std::unordered_map<int64_t, ssize_t> key2idx;
  std::vector<int32_t> uniq_src;
  std::vector<float> uniq_ts;
  uniq_src.reserve(size);
  uniq_ts.reserve(size);
  key2idx.reserve(size);

  for (ssize_t i = 0; i < size; i++) {
    int32_t src = s_ptr[i];
    float ts = t_ptr[i];

    int64_t key = opt_hash(src, ts);

    auto iter = key2idx.find(key);
    if (iter != key2idx.end()) {
      auto uniq_idx = iter->second;
      i_ptr[i] = uniq_idx;
    } else {
      auto idx = uniq_src.size();
      uniq_src.push_back(src);
      uniq_ts.push_back(ts);
      key2idx.emplace(key, idx);
      i_ptr[i] = idx;
    }
  }

  py::array_t<int32_t> src_res = py::cast(uniq_src);
  py::array_t<float> ts_res = py::cast(uniq_ts);

  return py::make_tuple(src_res, ts_res, inv_idx);
}

py::tuple find_dedup_time_hits(torch::Tensor &ts_delta,
                               torch::Tensor &time_embeds,
                               int time_window) {
  auto tup = torch::_unique(ts_delta.flatten(), /*sorted=*/true, /*return_inverse=*/true);

  ts_delta = std::get<0>(tup);
  auto inv_idx = std::get<1>(tup);

  auto delta_int = ts_delta.to(torch::kInt64);
  auto hit_idx = (delta_int == ts_delta) & (0 <= delta_int) & (delta_int <= time_window);
  auto hit_delta = delta_int.index({hit_idx});

  int64_t hit_count = torch::sum(hit_idx).item().toLong();
  int64_t uniq_size = ts_delta.size(0);

  torch::Tensor out_embeds;
  if (hit_count == uniq_size) {
    out_embeds = time_embeds.index({hit_delta});
  } else {
    int64_t time_dim = time_embeds.size(1);
    auto options = torch::TensorOptions().device(time_embeds.device());
    out_embeds = torch::zeros({uniq_size, time_dim}, options);
    out_embeds.index_put_({hit_idx}, time_embeds.index({hit_delta}));
  }

  return py::make_tuple(hit_count, hit_idx, out_embeds, ts_delta, inv_idx);
}

py::array_t<int64_t> compute_keys(py::array_t<int32_t> &src_l, py::array_t<float> &ts_l) {
  ssize_t size = src_l.size();
  auto keys = py::array_t<int64_t>(size);

  auto k_buf = keys.request();
  auto s_buf = src_l.request();
  auto t_buf = ts_l.request();

  auto *k_ptr = static_cast<int64_t *>(k_buf.ptr);
  auto *s_ptr = static_cast<int32_t *>(s_buf.ptr);
  auto *t_ptr = static_cast<float *>(t_buf.ptr);

  #ifndef tgopt_1t_cache_keys
  #pragma omp parallel for
  #endif
  for (ssize_t i = 0; i < size; i++) {
    k_ptr[i] = opt_hash(s_ptr[i], t_ptr[i]);
  }

  return keys;
}

/// Table for caching computed embeddings.
class EmbedTable {
 public:
  using KeyType = int64_t;
  using ValType = torch::Tensor;

  EmbedTable(size_t limit) : _limit(limit) {}

  void store(py::array_t<int64_t> &keys, torch::Tensor &embeds) {
    ssize_t size = keys.size();
    auto k_buf = keys.request();
    auto *k_ptr = static_cast<int64_t *>(k_buf.ptr);

    #ifdef tgopt_embed_store_dev
    auto tensor = embeds;
    #else
    auto tensor = embeds.cpu();
    #endif

    size_t new_size = _keys.size() + size;
    while (new_size > _limit) {
      auto key = _keys.front();
      _table.unsafe_erase(key);
      _keys.pop_front();
      new_size--;
    }

    for (ssize_t i = 0; i < size; i++) {
      _keys.push_back(k_ptr[i]);
    }

    #ifndef tgopt_1t_cache_store
    #pragma omp parallel for
    #endif
    for (ssize_t i = 0; i < size; i++) {
      int64_t key = k_ptr[i];
      _table.emplace(key, tensor[i]);
    }
  }

  py::tuple lookup(py::array_t<int64_t> &keys, ssize_t feat_dim, torch::Device &device) {
    ssize_t size = keys.size();

    #ifdef tgopt_embed_store_dev
    auto embeds = torch::zeros({size, feat_dim}, torch::TensorOptions().device(device));
    #else
    auto embeds = torch::zeros({size, feat_dim});
    #endif

    auto k_buf = keys.request();
    auto *k_ptr = static_cast<int64_t *>(k_buf.ptr);

    auto hit_idx = torch::zeros(size, torch::TensorOptions().dtype(torch::kBool));
    auto hit_idx_view = hit_idx.accessor<bool, 1>();

    #ifndef tgopt_1t_cache_lookup
    #pragma omp parallel for
    #endif
    for (ssize_t i = 0; i < size; i++) {
      int64_t key = k_ptr[i];
      auto iter = _table.find(key);
      if (iter != _table.end()) {
        embeds[i] = iter->second;
        hit_idx_view[i] = true;
      }
    }

    hit_idx = hit_idx.to(device);

    #ifndef tgopt_embed_store_dev
    embeds = embeds.to(device);
    #endif

    return py::make_tuple(hit_idx, embeds);
  }

  size_t size_in_bytes() {
    size_t bytes = 0;
    size_t dim = 0;
    if (!_keys.empty()) {
      auto embed = _table[_keys.front()];
      bytes = embed.element_size();
      dim = embed.size(0);
    }

    size_t pad = 64;
    size_t key_size = sizeof(KeyType);
    size_t val_size = sizeof(ValType);
    size_t each_size = dim * bytes + 2 * key_size + val_size + pad;
    size_t total_size = _table.size() * each_size;

    return total_size;
  }

 private:
  tbb::concurrent_unordered_map<KeyType, ValType> _table;
  std::deque<KeyType> _keys;
  size_t _limit = 1000000;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // API: sampling
  m.def("sample_recent_ngh", &sample_recent_ngh, "Sample most recent temporal neighbors");

  // API: dedup
  m.def("dedup_src_ts", &dedup_src_ts, "Deduplicate node and timestamp lists");

  // API: time-encode
  m.def("find_dedup_time_hits", &find_dedup_time_hits, "Deduplicate time deltas and find hits");

  // API: cache
  m.def("compute_keys", &compute_keys, "Compute cache keys using node and timestamp inputs");
  py::class_<EmbedTable>(m, "EmbedTable")
    .def(py::init<size_t>())
    .def("store", &EmbedTable::store, "Store embeddings")
    .def("lookup", &EmbedTable::lookup, "Lookup stored embeddings")
    .def("size_in_bytes", &EmbedTable::size_in_bytes, "Get approximate size in bytes");
}
