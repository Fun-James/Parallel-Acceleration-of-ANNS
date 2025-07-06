#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#include <cassert>
#include <queue>
#include <cstring>
#include <algorithm>
#include <utility> // Added for std::pair
#include <omp.h> // OpenMP支持

// PQ配置参数
constexpr int PQ_M = 16;  // 子空间数量，
constexpr int PQ_K = 16;  // 聚类数量，

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
    
    //使用OpenMP加速计算距离表
    void compute_distance_table_avx2(const float* query) {
        if (!dist_tables) {
            dist_tables = new float[M * K];
        }
        // 使用OpenMP并行化子空间的距离表计算
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; m++) {
            const float* query_sub = query + m * sub_dim;
            for (int k = 0; k < K; k++) {
                const float* centroid = codebooks[m][k].data();
                float dist = 0.0f;
                
                // 标量计算
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
        
        // 标量计算
        for (int d = 0; d < dimension; ++d) {
            float diff = vec1[d] - vec2[d];
            dist += diff * diff;
        }
        return dist;
    }
    
    // 使用预计算的距离表进行查询，并进行重排（OpenMP优化版本）
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

        // 1. 计算距离表（已经OpenMP优化）
        compute_distance_table_avx2(query_vec);
        
        // 2. 使用 PQ 近似距离进行初步检索，获取 top rerank_k 候选
        // 为了并行化，我们使用分块策略
        const int num_threads = omp_get_max_threads();
        const size_t chunk_size = (codes.size() + num_threads - 1) / num_threads;
        
        // 每个线程维护自己的top-k结果
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> thread_results(num_threads);
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            size_t start_idx = thread_id * chunk_size;
            size_t end_idx = std::min(start_idx + chunk_size, codes.size());
            
            auto& local_pq = thread_results[thread_id];
            
            for (size_t i = start_idx; i < end_idx; i++) {
                float approx_dist = 0;
                // 累加每个子空间的距离
                for (int m = 0; m < M; m++) {
                    uint8_t code = codes[i][m];
                    approx_dist += dist_tables[m * K + code];
                }
                
                if (local_pq.size() < rerank_k) {
                    local_pq.push({approx_dist, (uint32_t)i});
                } else if (approx_dist < local_pq.top().first) {
                    local_pq.pop();
                    local_pq.push({approx_dist, (uint32_t)i});
                }
            }
        }
        
        // 合并线程结果
        std::priority_queue<std::pair<float, uint32_t>> approx_result_pq;
        for (int t = 0; t < num_threads; t++) {
            while (!thread_results[t].empty()) {
                auto item = thread_results[t].top();
                thread_results[t].pop();
                
                if (approx_result_pq.size() < rerank_k) {
                    approx_result_pq.push(item);
                } else if (item.first < approx_result_pq.top().first) {
                    approx_result_pq.pop();
                    approx_result_pq.push(item);
                }
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

        // 4. 计算精确距离并进行重排（OpenMP并行化）
        // 每个线程维护自己的top-k结果
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> rerank_thread_results(num_threads);
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            size_t thread_chunk_size = (candidate_ids.size() + num_threads - 1) / num_threads;
            size_t start_idx = thread_id * thread_chunk_size;
            size_t end_idx = std::min(start_idx + thread_chunk_size, candidate_ids.size());
            
            auto& local_pq = rerank_thread_results[thread_id];
            
            for (size_t idx = start_idx; idx < end_idx; idx++) {
                uint32_t id = candidate_ids[idx];
                if (id >= base_num) { // 安全检查
                    continue;
                }
                
                const float* base_vec = base_data + (size_t)id * dim; // 获取原始向量指针
                float exact_dist = compute_exact_distance(query_vec, base_vec, dim);

                if (local_pq.size() < k) {
                    local_pq.push({exact_dist, id});
                } else if (exact_dist < local_pq.top().first) {
                    local_pq.pop();
                    local_pq.push({exact_dist, id});
                }
            }
        }
        
        // 合并重排结果
        std::priority_queue<std::pair<float, uint32_t>> final_result_pq;
        for (int t = 0; t < num_threads; t++) {
            while (!rerank_thread_results[t].empty()) {
                auto item = rerank_thread_results[t].top();
                rerank_thread_results[t].pop();
                
                if (final_result_pq.size() < k) {
                    final_result_pq.push(item);
                } else if (item.first < final_result_pq.top().first) {
                    final_result_pq.pop();
                    final_result_pq.push(item);
                }
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