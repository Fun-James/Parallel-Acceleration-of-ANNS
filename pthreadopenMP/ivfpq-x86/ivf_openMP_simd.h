#pragma once

#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <immintrin.h>                 // 替换为x86 SIMD指令集

#include "ivf_pthread_dynamic.h"      // 你的 IVFIndex / g_ivf_index 声明

/* ------------------------------------------------------------------ */
/*                       SSE L2 距离 (float)                          */
/* ------------------------------------------------------------------ */
inline float l2_distance_simd(const float* a, const float* b, int dim)
{
    int i = 0;
    float sum = 0.0f;
    
    #ifdef __AVX2__
    // 使用AVX2指令集
    __m256 sum256 = _mm256_setzero_ps();
    
    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(diff, diff));
    }
    
    // 水平求和
    __m128 sum128 = _mm_add_ps(
        _mm256_extractf128_ps(sum256, 0),
        _mm256_extractf128_ps(sum256, 1)
    );
    
    // 使用hadd进行水平求和
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum += _mm_cvtss_f32(sum128);
    #else
    // 使用SSE指令集
    __m128 sum128 = _mm_setzero_ps();
    
    for (; i + 3 < dim; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        sum128 = _mm_add_ps(sum128, _mm_mul_ps(diff, diff));
    }
    
    // 水平求和
    __m128 shuf = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    sum += _mm_cvtss_f32(sums);
    #endif
    
    // 处理剩余元素
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    return sum;
}

/* ------------------------------------------------------------------ */
/*             IVF + OpenMP (+SIMD) 搜索主函数                        */
/* ------------------------------------------------------------------ */
inline std::priority_queue<std::pair<float,int>>
ivf_search(const float* base,
           const float* query,
           size_t       base_number,   // 兼容参数，不使用
           size_t       dim,
           size_t       k,
           int          nprobe)
{
    using DistIdx = std::pair<float,int>;

    const int nlist = g_ivf_index.nlist;
    if (nlist == 0) { std::cerr<<"[ivf_simd] Empty index\n"; return {}; }

    /* 1️⃣  query→centroid 距离并行计算 */
    std::vector<DistIdx> cdists(nlist);

    #pragma omp parallel for schedule(static)
    for (int cid = 0; cid < nlist; ++cid) {
        float dist = l2_distance_simd(query,  // 替换l2_distance_neon
                                    g_ivf_index.centroids[cid].data(),
                                    static_cast<int>(dim));
        cdists[cid] = {dist, cid};
    }

    const int real_probe = std::min(nprobe, nlist);
    std::partial_sort(cdists.begin(),
                      cdists.begin()+real_probe,
                      cdists.end());

    /* 2️⃣  并行扫描倒排表，每线程局部 top-k */
    const int T = omp_get_max_threads();
    std::vector<std::priority_queue<DistIdx>> local_heaps(T);

    #pragma omp parallel for schedule(dynamic)
    for (int p = 0; p < real_probe; ++p) {
        const int tid = omp_get_thread_num();
        auto& heap    = local_heaps[tid];

        int list_id         = cdists[p].second;
        const auto& invlist = g_ivf_index.invlists[list_id];

        for (int idx : invlist) {
            const float* vec = base + static_cast<size_t>(idx) * dim;
            float dist = l2_distance_simd(query, vec, static_cast<int>(dim));  // 替换l2_distance_neon

            if (heap.size() < k) {
                heap.emplace(dist, idx);
            } else if (dist < heap.top().first) {
                heap.pop(); heap.emplace(dist, idx);
            }
        }
    }

    /* 3️⃣  归并局部堆 */
    std::priority_queue<DistIdx> global_heap;
    for (auto& h : local_heaps) {
        while (!h.empty()) {
            DistIdx cand = h.top(); h.pop();
            if (global_heap.size() < k) {
                global_heap.push(cand);
            } else if (cand.first < global_heap.top().first) {
                global_heap.pop();
                global_heap.push(cand);
            }
        }
    }

    /* 4️⃣  转成 (-dist, id) 以兼容主程序输出 */
    std::priority_queue<DistIdx> ret;
    while (!global_heap.empty()) {
        ret.emplace(-global_heap.top().first,
                     global_heap.top().second);
        global_heap.pop();
    }
    return ret;
}