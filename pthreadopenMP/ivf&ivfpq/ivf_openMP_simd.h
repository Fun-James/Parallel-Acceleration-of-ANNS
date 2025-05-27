#pragma once

#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <arm_neon.h>                 // NEON intrinsics

#include "ivf_pthread_dynamic.h"      // 你的 IVFIndex / g_ivf_index 声明

/* ------------------------------------------------------------------ */
/*                       NEON L2 距离 (float)                         */
/* ------------------------------------------------------------------ */
inline float l2_distance_neon(const float* a, const float* b, int dim)
{
    int i = 0;
    float32x4_t vsum = vdupq_n_f32(0.f);       // 4 lanes 累加

    // 每次处理 16 个 float，展开 4×4 可降低 loop overhead
    for (; i + 15 < dim; i += 16)
    {
        float32x4_t va0 = vld1q_f32(a + i     );
        float32x4_t vb0 = vld1q_f32(b + i     );
        float32x4_t d0  = vsubq_f32(va0, vb0);
        vsum = vmlaq_f32(vsum, d0, d0);        // diff^2 累加

        float32x4_t va1 = vld1q_f32(a + i + 4 );
        float32x4_t vb1 = vld1q_f32(b + i + 4 );
        float32x4_t d1  = vsubq_f32(va1, vb1);
        vsum = vmlaq_f32(vsum, d1, d1);

        float32x4_t va2 = vld1q_f32(a + i + 8 );
        float32x4_t vb2 = vld1q_f32(b + i + 8 );
        float32x4_t d2  = vsubq_f32(va2, vb2);
        vsum = vmlaq_f32(vsum, d2, d2);

        float32x4_t va3 = vld1q_f32(a + i + 12);
        float32x4_t vb3 = vld1q_f32(b + i + 12);
        float32x4_t d3  = vsubq_f32(va3, vb3);
        vsum = vmlaq_f32(vsum, d3, d3);
    }

    // 每次处理 4 个 float
    for (; i + 3 < dim; i += 4)
    {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t d  = vsubq_f32(va, vb);
        vsum = vmlaq_f32(vsum, d, d);
    }

    // 水平求和 vsum
    float32x2_t vlow  = vget_low_f32(vsum);
    float32x2_t vhigh = vget_high_f32(vsum);
    float32x2_t vp    = vpadd_f32(vlow, vhigh);      // [0]+[1] , [2]+[3]
    float sum = vget_lane_f32(vp,0) + vget_lane_f32(vp,1);

    // 处理剩余元素
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/* 若非 NEON 平台亦希望编译通过，可提供普通版本并用宏选择 */
#ifndef __ARM_NEON
#  warning "Building without NEON! l2_distance will fall back to scalar."
inline float l2_distance_neon(const float* a,const float* b,int dim){
    float res=0.f;for(int i=0;i<dim;++i){float d=a[i]-b[i];res+=d*d;}return res;}
#endif

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
        float dist = l2_distance_neon(query,
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
            float dist = l2_distance_neon(query, vec, static_cast<int>(dim));

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