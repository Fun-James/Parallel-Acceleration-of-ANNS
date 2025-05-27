#pragma once

#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <omp.h>

// PQ参数配置
constexpr int PQIVF_M = 16;  // 子空间数量
constexpr int PQIVF_K = 256;  // 每个子空间的聚类数

// PQIVF索引结构
struct PQIVFIndex {
    // PQ部分 (先执行PQ)
    int M = PQIVF_M;                        // 子空间数量
    int K = PQIVF_K;                        // 每个子空间的聚类数
    int dim = 0;                            // 向量维度
    int sub_dim = 0;                        // 每个子空间的维度 (dim / M)
    
    std::vector<std::vector<std::vector<float>>> codebooks;  // [M][K][sub_dim] PQ码本
    std::vector<std::vector<uint8_t>> codes;  // [n][M] 所有数据的PQ编码
    
    // IVF部分 (基于PQ编码空间的聚类)
    int nlist = 0;                           // 聚类中心个数
    std::vector<std::vector<uint8_t>> centroids;  // [nlist][M] 聚类中心 (基于PQ距离的中心)
    std::vector<std::vector<int>> invlists;  // [nlist][n_i] 倒排表
    
    // 查询时用的临时距离表
    float* dist_tables = nullptr;  // 大小为 M * K，用于PQ距离计算
    
    // 析构函数
    ~PQIVFIndex() {
        if (dist_tables) {
            delete[] dist_tables;
        }
    }
    
    // 计算查询向量的PQ距离表
    void compute_distance_table(const float* query) {
        if (!dist_tables) {
            dist_tables = new float[M * K];
        }
        
        for (int m = 0; m < M; m++) {
            const float* query_sub = query + m * sub_dim;
            for (int k = 0; k < K; k++) {
                const float* centroid = codebooks[m][k].data();
                float dist = 0.0f;
                
                // 计算子空间内的欧几里得距离平方
                for (int d = 0; d < sub_dim; d++) {
                    float diff = query_sub[d] - centroid[d];
                    dist += diff * diff;
                }
                dist_tables[m * K + k] = dist;
            }
        }
    }
    
    // 对查询向量进行PQ编码
    std::vector<uint8_t> encode_query(const float* query) {
        std::vector<uint8_t> query_code(M);
        
        for (int m = 0; m < M; m++) {
            const float* query_sub = query + m * sub_dim;
            float min_dist = INFINITY;
            int best_centroid = 0;
            
            for (int k = 0; k < K; k++) {
                const float* centroid = codebooks[m][k].data();
                float dist = 0.0f;
                
                for (int d = 0; d < sub_dim; d++) {
                    float diff = query_sub[d] - centroid[d];
                    dist += diff * diff;
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_centroid = k;
                }
            }
            
            query_code[m] = best_centroid;
        }
        
        return query_code;
    }
    
    // 计算两个PQ编码向量之间的距离（使用预计算的距离表）
    float pq_distance(const std::vector<uint8_t>& code) {
        float dist = 0.0f;
        for (int m = 0; m < M; m++) {
            dist += dist_tables[m * K + code[m]];
        }
        return dist;
    }
    
    // 计算两个PQ编码向量之间的欧式距离
    float pq_distance_symmetric(const std::vector<uint8_t>& code1, const std::vector<uint8_t>& code2) {
        float dist = 0.0f;
        for (int m = 0; m < M; m++) {
            for (int d = 0; d < sub_dim; d++) {
                float diff = codebooks[m][code1[m]][d] - codebooks[m][code2[m]][d];
                dist += diff * diff;
            }
        }
        return dist;
    }
    
    // 加载索引文件
    bool load(const std::string& filename) {
        std::ifstream fin(filename, std::ios::binary);
        if (!fin.is_open()) return false;
        
        // 读取PQ基础参数
        fin.read(reinterpret_cast<char*>(&dim), sizeof(int));
        fin.read(reinterpret_cast<char*>(&M), sizeof(int));
        fin.read(reinterpret_cast<char*>(&K), sizeof(int));
        
        // 计算子空间维度
        sub_dim = dim / M;
        
        // 读取PQ码本
        codebooks.resize(M);
        for (int m = 0; m < M; m++) {
            codebooks[m].resize(K);
            for (int k = 0; k < K; k++) {
                codebooks[m][k].resize(sub_dim);
                fin.read(reinterpret_cast<char*>(codebooks[m][k].data()), 
                        sub_dim * sizeof(float));
            }
        }
        
        // 读取IVF基础参数
        fin.read(reinterpret_cast<char*>(&nlist), sizeof(int));
        
        // 读取IVF聚类中心
        centroids.resize(nlist, std::vector<uint8_t>(M)); // <--- 修改类型
        for (int i = 0; i < nlist; ++i) {
            fin.read(reinterpret_cast<char*>(centroids[i].data()), M * sizeof(uint8_t)); // <--- 修改类型和大小
        }
        
        // 读取数据总数
        int ntotal = 0;
        fin.read(reinterpret_cast<char*>(&ntotal), sizeof(int));
        
        // 读取所有数据的PQ编码
        codes.resize(ntotal);
        for (int i = 0; i < ntotal; i++) {
            codes[i].resize(M);
            fin.read(reinterpret_cast<char*>(codes[i].data()), M * sizeof(uint8_t));
        }
        
        // 读取倒排表
        invlists.resize(nlist);
        for (int list_id = 0; list_id < nlist; list_id++) {
            // 读取倒排表大小
            int list_size = 0;
            fin.read(reinterpret_cast<char*>(&list_size), sizeof(int));
            
            // 读取倒排表ID
            invlists[list_id].resize(list_size);
            fin.read(reinterpret_cast<char*>(invlists[list_id].data()), 
                    list_size * sizeof(int));
        }
        
        fin.close();
        return true;
    }
    
    // 保存索引到文件
    void save(const std::string& filename) const {
        std::ofstream fout(filename, std::ios::binary);
        
        // 写入PQ基础参数
        fout.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        fout.write(reinterpret_cast<const char*>(&M), sizeof(int));
        fout.write(reinterpret_cast<const char*>(&K), sizeof(int));
        
        // 写入PQ码本
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                fout.write(reinterpret_cast<const char*>(codebooks[m][k].data()), 
                        sub_dim * sizeof(float));
            }
        }
        
        // 写入IVF基础参数
        fout.write(reinterpret_cast<const char*>(&nlist), sizeof(int));
        
        // 写入IVF聚类中心
        for (int i = 0; i < nlist; ++i) {
            fout.write(reinterpret_cast<const char*>(centroids[i].data()), 
                    M * sizeof(uint8_t)); // <--- 修改类型和大小
        }
        
        // 写入数据总数
        int ntotal = static_cast<int>(codes.size());
        fout.write(reinterpret_cast<const char*>(&ntotal), sizeof(int));
        
        // 写入所有数据的PQ编码
        for (int i = 0; i < ntotal; i++) {
            fout.write(reinterpret_cast<const char*>(codes[i].data()), 
                    M * sizeof(uint8_t));
        }
        
        // 写入倒排表
        for (int list_id = 0; list_id < nlist; list_id++) {
            // 写入倒排表大小
            int list_size = static_cast<int>(invlists[list_id].size());
            fout.write(reinterpret_cast<const char*>(&list_size), sizeof(int));
            
            // 写入倒排表ID
            fout.write(reinterpret_cast<const char*>(invlists[list_id].data()), 
                    list_size * sizeof(int));
        }
        
        fout.close();
    }
};

// 全局PQIVF索引
extern PQIVFIndex g_pqivf_index;

/* ----------------------------------------------------------- */
/*                   一些工具函数                              */
/* ----------------------------------------------------------- */
inline float l2_distance(const float* a, const float* b, int dim) {
    float res = 0.f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        res += diff * diff;
    }
    return res;
}

/* ----------------------------------------------------------- */
/*                     PQIVF搜索函数                          */
/* ----------------------------------------------------------- */
inline std::priority_queue<std::pair<float, int>>
pqivf_search(const float* base,           // 原始数据矩阵
           const float* query,          // 查询向量
           size_t       base_number,    // 数据行数
           size_t       dim,            // 维度
           size_t       k,              // 需要的最近邻个数
           int          nprobe,         // 要探测的IVF聚类中心数
           int          rerank_k = 0)   // 使用原始向量重排的候选数量，0表示不重排
{
    using DistIdx = std::pair<float, int>;
    
    //------------------------------------------------------------------
    // 1. 计算查询向量的PQ距离表
    //------------------------------------------------------------------
    g_pqivf_index.compute_distance_table(query);
    
    // 对查询向量进行PQ编码
    std::vector<uint8_t> query_code = g_pqivf_index.encode_query(query);
    
    //------------------------------------------------------------------
    // 2. 找到查询向量最近的nprobe个聚类中心
    //------------------------------------------------------------------
    std::vector<DistIdx> cent_dists;
    cent_dists.reserve(g_pqivf_index.nlist);
    
    for (int cid = 0; cid < g_pqivf_index.nlist; ++cid) {
        // 计算查询向量与聚类中心的距离
        float dist = 0.0f;
        for (int m = 0; m < g_pqivf_index.M; ++m) {
            int query_cluster = query_code[m];
            // centroids[cid][m] 现在是 uint8_t, 会自动提升为 int
            int centroid_cluster = g_pqivf_index.centroids[cid][m]; 
            for (int d = 0; d < g_pqivf_index.sub_dim; ++d) {
                float diff = g_pqivf_index.codebooks[m][query_cluster][d] - 
                            g_pqivf_index.codebooks[m][centroid_cluster][d];
                dist += diff * diff;
            }
        }
        cent_dists.emplace_back(dist, cid);
    }
    
    std::partial_sort(cent_dists.begin(),
                      cent_dists.begin() + std::min(nprobe, g_pqivf_index.nlist),
                      cent_dists.end());
    
    //------------------------------------------------------------------
    // 3. 在选中的倒排表中搜索
    //------------------------------------------------------------------
    // 确定PQ搜索阶段需要获取的候选者数量
    size_t num_candidates_for_pq_search = (rerank_k > 0) ? static_cast<size_t>(rerank_k) : k;
    
    std::priority_queue<DistIdx> topk;  // 大顶堆，维护当前最近的候选向量
    
    for (int p = 0; p < nprobe && p < g_pqivf_index.nlist; ++p) {
        int list_id = cent_dists[p].second;
        const auto& invlist = g_pqivf_index.invlists[list_id];
        
        if (invlist.empty()) continue;
        
        // 遍历倒排表中的所有向量
        for (size_t i = 0; i < invlist.size(); ++i) {
            int idx = invlist[i];
            float approx_dist = g_pqivf_index.pq_distance(g_pqivf_index.codes[idx]);
            
            // 维护topk
            if (topk.size() < num_candidates_for_pq_search) {
                topk.emplace(approx_dist, idx);
            } else if (approx_dist < topk.top().first) {
                topk.pop();
                topk.emplace(approx_dist, idx);
            }
        }
    }
    
    //------------------------------------------------------------------
    // 4. 如果需要重排序，使用原始向量计算精确距离
    //------------------------------------------------------------------
    if (rerank_k > 0) {
        // 收集需要重排的向量ID
        std::vector<DistIdx> candidates;
        candidates.reserve(topk.size());
        
        while (!topk.empty()) {
            candidates.push_back(topk.top());
            topk.pop();
        }
        
        // 限制重排数量
        int actual_rerank = std::min(static_cast<int>(candidates.size()), rerank_k);
        
        // 使用多线程并行重新计算距离
        std::vector<DistIdx> exact_distances(actual_rerank);
        
        #pragma omp parallel for
        for (int i = 0; i < actual_rerank; ++i) {
            int idx = candidates[i].second;
            const float* vec = base + static_cast<size_t>(idx) * dim;
            float exact_dist = l2_distance(query, vec, dim);
            exact_distances[i] = std::make_pair(exact_dist, idx);
        }
        
        // 填充结果堆
        for (const auto& dist_idx : exact_distances) {
            if (topk.size() < k) {
                topk.push(dist_idx);
            } else if (dist_idx.first < topk.top().first) {
                topk.pop();
                topk.push(dist_idx);
            }
        }
    }
    
    //------------------------------------------------------------------
    // 5. 返回结果，将距离取负
    //------------------------------------------------------------------
    std::priority_queue<DistIdx> results;
    while (!topk.empty()) {
        results.emplace(-topk.top().first, topk.top().second);
        topk.pop();
    }
    
    return results;  // 注意：距离是负值
}
