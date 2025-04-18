#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <arm_neon.h>  // ARM NEON 指令集
#include <cassert>
#include <queue>
#include <cstring>
#include <algorithm>
#include <utility> // Added for std::pair

// PQ配置参数
constexpr int PQ_M = 16;  // 子空间数量，
constexpr int PQ_K = 256;  // 聚类数量，

// PQ索引结构
struct PQIndex {
    int M;              // 子空间数量
    int K;              // 每个子空间的聚类数
    int dim;            // 原始向量维度
    int sub_dim;        // 每个子空间的维度 (dim / M)
    
    std::vector<std::vector<std::vector<float>>> codebooks;  // 码本 [M][K][sub_dim]
    std::vector<std::vector<uint8_t>> codes;                // 编码后的数据 [n][M]
    
    // 查询时用的距离表
    float* dist_tables;  // 大小为 M * K
    // 添加原始数据指针以进行重排
    const float* base_data; 
    size_t base_num;

    PQIndex(int m = PQ_M, int k = PQ_K) : M(m), K(k), dist_tables(nullptr), base_data(nullptr), base_num(0) {}
    
    ~PQIndex() {
        if (dist_tables) {
            delete[] dist_tables;
        }
    }
    
    //使用NEON加速计算距离表
    void compute_distance_table_neon(const float* query) {
        if (!dist_tables) {
            dist_tables = new float[M * K];
        }
        // 为每个子空间计算距离表
        for (int m = 0; m < M; m++) {
            const float* query_sub = query + m * sub_dim;
            for (int k = 0; k < K; k++) {
                const float* centroid = codebooks[m][k].data();
                float dist = 0.0f;
                // 累加子空间内的欧几里得距离平方
                for (int d = 0; d < sub_dim; d++) {
                    float diff = query_sub[d] - centroid[d];
                    dist += diff * diff;
                }
                dist_tables[m * K + k] = dist;
            }
        }
    }
    
    // 加载索引
    bool load(const std::string& filename) {
        std::ifstream fin(filename, std::ios::binary);
        if (!fin) {
            return false;
        }
        
        fin.read(reinterpret_cast<char*>(&M), sizeof(int));
        fin.read(reinterpret_cast<char*>(&K), sizeof(int));
        fin.read(reinterpret_cast<char*>(&dim), sizeof(int));
        sub_dim = dim / M;
        
        // 读取码本
        codebooks.resize(M);
        for (int m = 0; m < M; m++) {
            codebooks[m].resize(K);
            for (int k = 0; k < K; k++) {
                codebooks[m][k].resize(sub_dim);
                fin.read(reinterpret_cast<char*>(codebooks[m][k].data()), sub_dim * sizeof(float));
            }
        }
        
        // 读取编码数据
        int n;
        fin.read(reinterpret_cast<char*>(&n), sizeof(int));
        codes.resize(n);
        for (int i = 0; i < n; i++) {
            codes[i].resize(M);
            fin.read(reinterpret_cast<char*>(codes[i].data()), M * sizeof(uint8_t));
        }
        
        fin.close();
        return true;
    }

    // 设置原始数据指针，用于重排
    void set_base_data(const float* base, size_t n) {
        base_data = base;
        base_num = n;
    }

    // 计算两个向量之间的精确平方欧氏距离
    float compute_exact_distance(const float* vec1, const float* vec2, int dimension) {
        float dist = 0.0f;
        // 可以添加 SIMD 优化
        for (int d = 0; d < dimension; ++d) {
            float diff = vec1[d] - vec2[d];
            dist += diff * diff;
        }
        return dist;
    }
    
    // 使用预计算的距离表进行查询，并进行重排
    std::priority_queue<std::pair<float, uint32_t>> query(const float* query_vec, int k, int rerank_k) {
        if (!base_data) {
             std::cerr << "Error: Base data not set for reranking." << std::endl;
             // 返回空结果或者抛出异常
             return std::priority_queue<std::pair<float, uint32_t>>();
        }
        if (rerank_k < k) {
            rerank_k = k; // 确保 rerank_k 至少为 k
            std::cerr << "Warning: rerank_k is less than k. Setting rerank_k = k." << std::endl;
        }
        if (rerank_k > codes.size()) {
             rerank_k = codes.size(); // rerank_k 不能超过总数据量
             std::cerr << "Warning: rerank_k is greater than the number of base vectors. Setting rerank_k to base size." << std::endl;
        }


        // 1. 计算距离表
        compute_distance_table_neon(query_vec);
        
        // 2. 使用 PQ 近似距离进行初步检索，获取 top rerank_k 候选
        std::priority_queue<std::pair<float, uint32_t>> approx_result_pq; // Max-heap for approx distances
        
        for (size_t i = 0; i < codes.size(); i++) {
            float approx_dist = 0;
            // 累加每个子空间的距离
            for (int m = 0; m < M; m++) {
                uint8_t code = codes[i][m];
                approx_dist += dist_tables[m * K + code];
            }
            
            if (approx_result_pq.size() < rerank_k) {
                approx_result_pq.push({approx_dist, (uint32_t)i});
            } else if (approx_dist < approx_result_pq.top().first) {
                approx_result_pq.pop();
                approx_result_pq.push({approx_dist, (uint32_t)i});
            }
        }

        // 3. 提取 rerank_k 候选 ID
        std::vector<uint32_t> candidate_ids;
        candidate_ids.reserve(approx_result_pq.size());
        while (!approx_result_pq.empty()) {
            candidate_ids.push_back(approx_result_pq.top().second);
            approx_result_pq.pop();
        }
        // 反转，因为 priority_queue 是最大堆，我们想要距离最近的
        std::reverse(candidate_ids.begin(), candidate_ids.end());


        // 4. 计算精确距离并进行重排
        std::priority_queue<std::pair<float, uint32_t>> final_result_pq; // Max-heap for exact distances

        for (uint32_t id : candidate_ids) {
             if (id >= base_num) { // 安全检查
                 std::cerr << "Error: Candidate ID " << id << " out of bounds (" << base_num << ")" << std::endl;
                 continue;
             }
            const float* base_vec = base_data + (size_t)id * dim; // 获取原始向量指针
            float exact_dist = compute_exact_distance(query_vec, base_vec, dim);

            if (final_result_pq.size() < k) {
                final_result_pq.push({exact_dist, id});
            } else if (exact_dist < final_result_pq.top().first) {
                final_result_pq.pop();
                final_result_pq.push({exact_dist, id});
            }
        }
        
        // 5. 返回最终 top-k 结果 (priority_queue 内部已按距离排序)
        return final_result_pq;
    }
};

// 全局PQ索引变量
static PQIndex g_pq_index;


// 更新 pq_search 包装函数以接受 rerank_k 参数
std::priority_queue<std::pair<float, uint32_t>> pq_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, int rerank_k) {
    // 确保在查询前设置了原始数据指针
    g_pq_index.set_base_data(base, base_number); 
    // 使用PQ索引查询并进行重排
    return g_pq_index.query(query, k, rerank_k);
}