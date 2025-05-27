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
constexpr int IVFPQ_M = 16;  // 子空间数量
constexpr int IVFPQ_K = 16;  // 每个子空间的聚类数

// IVFPQ索引结构
struct IVFPQIndex {
    // IVF部分
    int nlist = 0;                           // 聚类中心个数
    int dim = 0;                            // 向量维度
    std::vector<std::vector<float>> centroids;  // [nlist][dim] 聚类中心
    
    // PQ部分
    int M = IVFPQ_M;                        // 子空间数量
    int K = IVFPQ_K;                        // 每个子空间的聚类数
    int sub_dim = 0;                        // 每个子空间的维度 (dim / M)
    
    // 每个倒排表的PQ码本和编码
    std::vector<std::vector<std::vector<std::vector<float>>>> codebooks;  // [nlist][M][K][sub_dim] 每个倒排表的码本
    std::vector<std::vector<std::vector<uint8_t>>> codes;  // [nlist][n_i][M] 每个倒排表内的编码数据
    std::vector<std::vector<int>> invlists;  // 倒排表：记录原始行号，便于重排序
    
    // 查询时用的临时距离表
    float* dist_tables = nullptr;  // 大小为 M * K，用于PQ距离计算
    
    // 析构函数
    ~IVFPQIndex() {
        if (dist_tables) {
            delete[] dist_tables;
        }
    }
    
    // 计算查询向量与某个倒排表的PQ距离表
    void compute_distance_table(const float* query, int list_id) {
        if (!dist_tables) {
            dist_tables = new float[M * K];
        }
        
        for (int m = 0; m < M; m++) {
            const float* query_sub = query + m * sub_dim;
            for (int k = 0; k < K; k++) {
                const float* centroid = codebooks[list_id][m][k].data();
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
    
    // 加载索引文件
    bool load(const std::string& filename) {
        std::ifstream fin(filename, std::ios::binary);
        if (!fin.is_open()) return false;
        
        // 读取IVF基础参数
        fin.read(reinterpret_cast<char*>(&nlist), sizeof(int));
        fin.read(reinterpret_cast<char*>(&dim), sizeof(int));
        fin.read(reinterpret_cast<char*>(&M), sizeof(int));
        fin.read(reinterpret_cast<char*>(&K), sizeof(int));
        
        // 计算子空间维度
        sub_dim = dim / M;
        
        // 读取聚类中心
        centroids.resize(nlist, std::vector<float>(dim));
        for (int i = 0; i < nlist; ++i) {
            fin.read(reinterpret_cast<char*>(centroids[i].data()), dim * sizeof(float));
        }
        
        // 读取PQ码本
        codebooks.resize(nlist);
        for (int list_id = 0; list_id < nlist; list_id++) {
            codebooks[list_id].resize(M);
            for (int m = 0; m < M; m++) {
                codebooks[list_id][m].resize(K);
                for (int k = 0; k < K; k++) {
                    codebooks[list_id][m][k].resize(sub_dim);
                    fin.read(reinterpret_cast<char*>(codebooks[list_id][m][k].data()), 
                            sub_dim * sizeof(float));
                }
            }
        }
        
        // 读取倒排表和PQ编码
        invlists.resize(nlist);
        codes.resize(nlist);
        
        for (int list_id = 0; list_id < nlist; list_id++) {
            // 读取倒排表大小
            int list_size = 0;
            fin.read(reinterpret_cast<char*>(&list_size), sizeof(int));
            
            // 读取倒排表ID
            invlists[list_id].resize(list_size);
            fin.read(reinterpret_cast<char*>(invlists[list_id].data()), 
                    list_size * sizeof(int));
            
            // 读取PQ编码
            codes[list_id].resize(list_size);
            for (int i = 0; i < list_size; i++) {
                codes[list_id][i].resize(M);
                fin.read(reinterpret_cast<char*>(codes[list_id][i].data()), 
                        M * sizeof(uint8_t));
            }
        }
        
        fin.close();
        return true;
    }
    
    // 保存索引到文件
    void save(const std::string& filename) const {
        std::ofstream fout(filename, std::ios::binary);
        
        // 写入基础参数
        fout.write(reinterpret_cast<const char*>(&nlist), sizeof(int));
        fout.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        fout.write(reinterpret_cast<const char*>(&M), sizeof(int));
        fout.write(reinterpret_cast<const char*>(&K), sizeof(int));
        
        // 写入聚类中心
        for (int i = 0; i < nlist; ++i) {
            fout.write(reinterpret_cast<const char*>(centroids[i].data()), 
                    dim * sizeof(float));
        }
        
        // 写入PQ码本
        for (int list_id = 0; list_id < nlist; list_id++) {
            for (int m = 0; m < M; m++) {
                for (int k = 0; k < K; k++) {
                    fout.write(reinterpret_cast<const char*>(codebooks[list_id][m][k].data()), 
                            sub_dim * sizeof(float));
                }
            }
        }
        
        // 写入倒排表和PQ编码
        for (int list_id = 0; list_id < nlist; list_id++) {
            // 写入倒排表大小
            int list_size = static_cast<int>(invlists[list_id].size());
            fout.write(reinterpret_cast<const char*>(&list_size), sizeof(int));
            
            // 写入倒排表ID
            fout.write(reinterpret_cast<const char*>(invlists[list_id].data()), 
                    list_size * sizeof(int));
            
            // 写入PQ编码
            for (int i = 0; i < list_size; i++) {
                fout.write(reinterpret_cast<const char*>(codes[list_id][i].data()), 
                        M * sizeof(uint8_t));
            }
        }
        
        fout.close();
    }
};

// 全局IVFPQ索引
extern IVFPQIndex g_ivfpq_index;

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
/*                     IVFPQ搜索函数                          */
/* ----------------------------------------------------------- */
inline std::priority_queue<std::pair<float, int>>
ivfpq_search(const float* base,           // 原始数据矩阵
           const float* query,          // 查询向量
           size_t       base_number,    // 数据行数
           size_t       dim,            // 维度
           size_t       k,              // 需要的最近邻个数
           int          nprobe,         // 要探测的IVF聚类中心数
           int          rerank_k = 0)   // 使用原始向量重排的候选数量，0表示不重排
{
    using DistIdx = std::pair<float, int>;
    
    //------------------------------------------------------------------
    // 1. 找到查询向量最近的nprobe个聚类中心
    //------------------------------------------------------------------
    std::vector<DistIdx> cent_dists;
    cent_dists.reserve(g_ivfpq_index.nlist);
    
    for (int cid = 0; cid < g_ivfpq_index.nlist; ++cid) {
        float dist = l2_distance(query, g_ivfpq_index.centroids[cid].data(), dim);
        cent_dists.emplace_back(dist, cid);
    }
    
    std::partial_sort(cent_dists.begin(),
                      cent_dists.begin() + std::min(nprobe, g_ivfpq_index.nlist),
                      cent_dists.end());
    
    //------------------------------------------------------------------
    // 2. 在选中的倒排表中使用PQ编码进行搜索
    //------------------------------------------------------------------
    // 确定PQ搜索阶段需要获取的候选者数量
    // 如果 rerank_k > 0, PQ阶段获取 rerank_k 个候选者
    // 否则 (rerank_k == 0), PQ阶段获取 k 个候选者
    size_t num_candidates_for_pq_search = (rerank_k > 0) ? static_cast<size_t>(rerank_k) : k;
    
    std::priority_queue<DistIdx> topk;  // 大顶堆，维护当前最近的 num_candidates_for_pq_search 个向量
    
    std::vector<float> query_residual(dim); // 用于存储查询向量的残差

    for (int p = 0; p < nprobe && p < g_ivfpq_index.nlist; ++p) {
        int list_id = cent_dists[p].second;
        const auto& invlist = g_ivfpq_index.invlists[list_id];
        
        if (invlist.empty()) continue;

        // 计算查询向量相对于当前粗聚类中心的残差
        const std::vector<float>& coarse_centroid = g_ivfpq_index.centroids[list_id];
        for (size_t d_idx = 0; d_idx < dim; ++d_idx) {
            query_residual[d_idx] = query[d_idx] - coarse_centroid[d_idx];
        }
        
        // 计算当前倒排表的PQ距离表 (使用查询残差)
        g_ivfpq_index.compute_distance_table(query_residual.data(), list_id);
        
        // 遍历倒排表中的所有向量
        for (size_t i = 0; i < invlist.size(); ++i) {
            int idx = invlist[i];
            float approx_dist = 0;
            
            // 计算PQ近似距离
            for (int m = 0; m < g_ivfpq_index.M; m++) {
                uint8_t code = g_ivfpq_index.codes[list_id][i][m];
                approx_dist += g_ivfpq_index.dist_tables[m * g_ivfpq_index.K + code];
            }
            
            // 维护topk
            if (topk.size() < num_candidates_for_pq_search) { // 修改：使用 num_candidates_for_pq_search
                topk.emplace(approx_dist, idx);
            } else if (approx_dist < topk.top().first) {
                topk.pop();
                topk.emplace(approx_dist, idx);
            }
        }
    }
    
    //------------------------------------------------------------------
    // 3. 如果需要重排序，使用原始向量计算精确距离
    //------------------------------------------------------------------
    if (rerank_k > 0) { // 修改：只要 rerank_k > 0 就进行重排
        // 收集需要重排的向量ID (从PQ搜索得到的 num_candidates_for_pq_search 个候选)
        std::vector<DistIdx> candidates;
        candidates.reserve(topk.size());
        
        while (!topk.empty()) {
            candidates.push_back(topk.top());
            topk.pop(); // topk 现在为空，后续将用精确距离结果重新填充至 k 个
        }
        
        // 限制重排数量 (实际上是遍历所有从PQ阶段获取的 rerank_k 个候选者)
        // candidates.size() 此时等于 num_candidates_for_pq_search (即 rerank_k)
        // actual_rerank 将等于 rerank_k
        int actual_rerank = std::min(static_cast<int>(candidates.size()), rerank_k);
        
        // 使用原始向量重新计算距离
        // 注意：candidates 是从大顶堆中取出，顺序可能不是最优的，但我们会遍历所有actual_rerank个
        for (int i = 0; i < actual_rerank; ++i) {
            int idx = candidates[i].second;
            const float* vec = base + static_cast<size_t>(idx) * dim;
            float exact_dist = l2_distance(query, vec, dim);
            
            // topk 被重新用来构建最终的 k 个结果
            topk.emplace(exact_dist, idx);
            
            // 保持堆大小不超过k
            if (topk.size() > k) topk.pop();
        }
    }
    
    //------------------------------------------------------------------
    // 4. 返回结果，为保持与其他实现的兼容性，将距离取负
    //------------------------------------------------------------------
    std::priority_queue<DistIdx> results;
    while (!topk.empty()) {
        results.emplace(-topk.top().first, topk.top().second);
        topk.pop();
    }
    
    return results;  // 注意：距离是负值
}
