#pragma once

#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>

#include "ivf_pthread_dynamic.h"   // 复用已有的 IVFIndex 结构体 & g_ivf_index

/* ------------------------------------------------------------------ */
/*                 一些小工具函数 (保持 inline 以免重复符号)           */
/* ------------------------------------------------------------------ */
inline float l2_distance(const float* a, const float* b, int dim) {
    float res = 0.f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        res += diff * diff;
    }
    return res;
}

/* ------------------------------------------------------------------ */
/*                     核心：OpenMP 并行搜索                           */
/* ------------------------------------------------------------------ */
//
// 返回 pair(-dist, idx) 的大顶堆，接口与原先完全一致
//
inline std::priority_queue<std::pair<float,int>>
ivf_search(const float* base,           // 原始数据矩阵 (row-major)
           const float* query,          // 查询向量
           size_t       base_number,    // 总行数（此处不用，但为了兼容保留）
           size_t       dim,            // 维度
           size_t       k,              // 需要的最近邻个数
           int          nprobe)         // 要探测的 centroid 数
{
    using DistIdx = std::pair<float,int>;   // (distance, id)

    const int nlist = g_ivf_index.nlist;
    if (nlist == 0) {
        std::cerr << "[ivf_omp] ERROR: IVF index is empty!\n";
        return {};
    }
    /* --------------------------------------------------------------
     * 1. 计算 query -> 所有 centroid 的 L2 距离 (并行 for)
     * --------------------------------------------------------------*/
    std::vector<DistIdx> centroid_dists(nlist);

    #pragma omp parallel for schedule(static)
    for (int cid = 0; cid < nlist; ++cid) {
        float dist = l2_distance(query, g_ivf_index.centroids[cid].data(),
                                 static_cast<int>(dim));
        centroid_dists[cid] = {dist, cid};
    }

    // 取最小的 nprobe 个 centroid
    const int real_probe = std::min(nprobe, nlist);
    std::partial_sort(centroid_dists.begin(),
                      centroid_dists.begin() + real_probe,
                      centroid_dists.end());

    /* --------------------------------------------------------------
     * 2. 并行遍历 nprobe 个倒排表
     *    每个线程维护本地 top-k，大顶堆
     * --------------------------------------------------------------*/
    const int max_threads = omp_get_max_threads();
    std::vector<std::priority_queue<DistIdx>> local_heaps(max_threads);

    #pragma omp parallel for schedule(dynamic)
    for (int p = 0; p < real_probe; ++p) {
        const int  tid      = omp_get_thread_num();
        auto&      heap     = local_heaps[tid];

        int list_id         = centroid_dists[p].second;
        const auto& invlist = g_ivf_index.invlists[list_id];

        for (int idx : invlist) {
            const float* vec = base + static_cast<size_t>(idx) * dim;
            float dist       = l2_distance(query, vec, static_cast<int>(dim));

            if (heap.size() < k) {
                heap.emplace(dist, idx);
            } else if (dist < heap.top().first) {
                heap.pop();
                heap.emplace(dist, idx);
            }
        }
    }

    /* --------------------------------------------------------------
     * 3. 归并各线程的局部 top-k，得到全局 top-k
     * --------------------------------------------------------------*/
    std::priority_queue<DistIdx> global_heap;
    for (auto& heap : local_heaps) {
        while (!heap.empty()) {
            const auto cand = heap.top(); heap.pop();

            if (global_heap.size() < k) {
                global_heap.push(cand);
            } else if (cand.first < global_heap.top().first) {
                global_heap.pop();
                global_heap.push(cand);
            }
        }
    }

    /* --------------------------------------------------------------
     * 4. 转成 (-dist, id) 的形式返回，保持与 main.cc 兼容
     * --------------------------------------------------------------*/
    std::priority_queue<DistIdx> results;
    while (!global_heap.empty()) {
        results.emplace(-global_heap.top().first, global_heap.top().second);
        global_heap.pop();
    }
    return results;
}