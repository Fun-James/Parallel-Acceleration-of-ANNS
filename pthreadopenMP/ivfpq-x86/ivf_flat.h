#pragma once

#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

//
// IVF 索引结构体
//
struct IVFIndex {
    int nlist = 0;                           // 聚类中心个数
    int dim   = 0;                           // 向量维度

    std::vector<std::vector<float>> centroids;  // [nlist][dim]
    std::vector<std::vector<int>>   invlists;   // 倒排表：每表存 base 中的行号

    // -------- 文件持久化 ----------
    bool load(const std::string& filename) {
        std::ifstream fin(filename, std::ios::binary);
        if (!fin.is_open()) return false;

        fin.read(reinterpret_cast<char*>(&nlist), sizeof(int));
        fin.read(reinterpret_cast<char*>(&dim),   sizeof(int));

        centroids.resize(nlist, std::vector<float>(dim));
        for (int i = 0; i < nlist; ++i)
            fin.read(reinterpret_cast<char*>(centroids[i].data()),
                     dim * sizeof(float));

        invlists.resize(nlist);
        for (int i = 0; i < nlist; ++i) {
            int list_size = 0;
            fin.read(reinterpret_cast<char*>(&list_size), sizeof(int));
            invlists[i].resize(list_size);
            fin.read(reinterpret_cast<char*>(invlists[i].data()),
                     list_size * sizeof(int));
        }
        fin.close();
        return true;
    }

    void save(const std::string& filename) const {
        std::ofstream fout(filename, std::ios::binary);

        fout.write(reinterpret_cast<const char*>(&nlist), sizeof(int));
        fout.write(reinterpret_cast<const char*>(&dim),   sizeof(int));

        for (int i = 0; i < nlist; ++i)
            fout.write(reinterpret_cast<const char*>(centroids[i].data()),
                       dim * sizeof(float));

        for (int i = 0; i < nlist; ++i) {
            int list_size = static_cast<int>(invlists[i].size());
            fout.write(reinterpret_cast<const char*>(&list_size), sizeof(int));
            fout.write(reinterpret_cast<const char*>(invlists[i].data()),
                       list_size * sizeof(int));
        }
        fout.close();
    }
};

// 给 main / build 函数用的全局索引
extern IVFIndex g_ivf_index;   // 仅声明，不再使用 inline 变量

/* ----------------------------------------------------------- */
/*                   一些小工具函数                            */
/* ----------------------------------------------------------- */
inline float l2_distance(const float* a, const float* b, int dim) {
    float res = 0.f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        res += diff * diff;
    }
    return res;
}

inline float inner_product(const float* a, const float* b, int dim) {
    float res = 0.f;
    for (int i = 0; i < dim; ++i) res += a[i] * b[i];
    return res;
}

/* ----------------------------------------------------------- */
/*                        核心搜索函数                         */
/* ----------------------------------------------------------- */
//
// 返回一个 pair(-dist, idx) 的大顶堆；main.cc 维持不变.
//
inline std::priority_queue<std::pair<float,int>>
ivf_search(const float* base,           // 原始数据矩阵
           const float* query,          // 查询向量
           size_t       base_number,    // 数据行数（没用到，但接口保留）
           size_t       dim,            // 维度
           size_t       k,              // 需要的最近邻个数
           int          nprobe)         // 要探测的 centroid 数
{
    using DistIdx = std::pair<float,int>;

    //------------------------------------------------------------------
    // 1. 找到 query 最近的 nprobe 个聚类中心
    //------------------------------------------------------------------
    std::vector<DistIdx> cent_dists;  cent_dists.reserve(g_ivf_index.nlist);
    for (int cid = 0; cid < g_ivf_index.nlist; ++cid) {
        float dist = l2_distance(query, g_ivf_index.centroids[cid].data(), dim);
        cent_dists.emplace_back(dist, cid);
    }
    std::partial_sort(cent_dists.begin(),
                      cent_dists.begin() + std::min(nprobe, g_ivf_index.nlist),
                      cent_dists.end());

    //------------------------------------------------------------------
    // 2. 在选中的倒排表里枚举所有元素，维护一个 size<=k 的大顶堆
    //------------------------------------------------------------------
    std::priority_queue<DistIdx> topk;  // 大顶堆, top() 是"当前最远"
    for (int p = 0; p < nprobe && p < g_ivf_index.nlist; ++p) {
        int list_id = cent_dists[p].second;
        const auto& inv = g_ivf_index.invlists[list_id];

        for (int idx : inv) {
            const float* vec = base + static_cast<size_t>(idx) * dim;
            float dist = l2_distance(query, vec, static_cast<int>(dim));

            if (topk.size() < k) {
                topk.emplace(dist, idx);
            } else if (dist < topk.top().first) {  // 更近 → 替换
                topk.pop();
                topk.emplace(dist, idx);
            }
        }
    }

    //------------------------------------------------------------------
    // 3. 把结果倒进另一个大顶堆，距离取负以保持 main.cc 兼容
    //------------------------------------------------------------------
    std::priority_queue<DistIdx> results;
    while (!topk.empty()) {
        results.emplace(-topk.top().first, topk.top().second);
        topk.pop();
    }
    return results;   // 注意：距离是负值
}
