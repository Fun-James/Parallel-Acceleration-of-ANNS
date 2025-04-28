#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <immintrin.h> // x86 AVX2 指令集
#include <cassert>
#include <queue>
#include <cstring>
#include <algorithm>
#include <utility> // Added for std::pair

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

    // 使用AVX2加速计算距离表
    void compute_distance_table_avx2(const float* query) {
        if (!dist_tables) {
            dist_tables = new float[M * K];
        }
        // 为每个子空间计算距离表
        for (int m = 0; m < M; m++) {
            const float* query_sub = query + m * sub_dim;
            for (int k = 0; k < K; k++) {
                const float* centroid = codebooks[m][k].data();
                dist_tables[m * K + k] = compute_exact_distance(query_sub, centroid, sub_dim);
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

    // 计算两个向量之间的精确平方欧氏距离 (使用 AVX2 优化)
    float compute_exact_distance(const float* vec1, const float* vec2, int dimension) {
        float dist = 0.0f;
        __m256 sum_vec = _mm256_setzero_ps();
        int d = 0;

        // 一次处理 8 个浮点数
        for (; d <= dimension - 8; d += 8) {
            __m256 v1 = _mm256_loadu_ps(vec1 + d); 
            __m256 v2 = _mm256_loadu_ps(vec2 + d);
            __m256 diff = _mm256_sub_ps(v1, v2);
            __m256 diff_sq = _mm256_mul_ps(diff, diff);
            sum_vec = _mm256_add_ps(sum_vec, diff_sq);
        }

        // 水平求和 AVX 寄存器中的结果
        float sum_arr[8];
        _mm256_storeu_ps(sum_arr, sum_vec);
        dist = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] +
               sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];

        // 处理剩余不足 8 个的元素
        for (; d < dimension; ++d) {
            float diff = vec1[d] - vec2[d];
            dist += diff * diff;
        }
        return dist;
    }
    
    // 使用预计算的距离表进行查询，并进行重排
    std::priority_queue<std::pair<float, uint32_t>> query(const float* query_vec, int k, int rerank_k) {
        if (!base_data) {
             std::cerr << "Error: Base data not set for reranking." << std::endl;
             return std::priority_queue<std::pair<float, uint32_t>>();
        }
        if (rerank_k < k) {
            rerank_k = k;
            std::cerr << "Warning: rerank_k is less than k. Setting rerank_k = k." << std::endl;
        }
        size_t num_codes = codes.size(); // Use consistent size variable
        if (rerank_k > num_codes) {
             rerank_k = num_codes;
             std::cerr << "Warning: rerank_k is greater than the number of base vectors. Setting rerank_k to base size." << std::endl;
        }

        // 1. 计算距离表 (使用 AVX2 版本)
        compute_distance_table_avx2(query_vec);

        // 2. 使用 PQ 近似距离进行初步检索，获取 top rerank_k 候选
        std::priority_queue<std::pair<float, uint32_t>> approx_result_pq;
        size_t i = 0;

        // Process 8 vectors at a time using AVX2 gather
        for (; i <= num_codes - 8; i += 8) {
            __m256 approx_dist_vec = _mm256_setzero_ps(); // Accumulator for 8 distances

            for (int m = 0; m < M; m++) {
                // Prepare indices for gather
                int32_t indices[8];
                indices[0] = (int32_t)codes[i + 0][m];
                indices[1] = (int32_t)codes[i + 1][m];
                indices[2] = (int32_t)codes[i + 2][m];
                indices[3] = (int32_t)codes[i + 3][m];
                indices[4] = (int32_t)codes[i + 4][m];
                indices[5] = (int32_t)codes[i + 5][m];
                indices[6] = (int32_t)codes[i + 6][m];
                indices[7] = (int32_t)codes[i + 7][m];

                // Load indices into AVX register
                __m256i indices_vec = _mm256_loadu_si256((__m256i*)indices);

                // Base pointer for the current sub-quantizer's distance table
                const float* table_base_ptr = dist_tables + m * K;

                // Gather distances from the LUT based on indices
                __m256 gathered_dists = _mm256_i32gather_ps(table_base_ptr, indices_vec, sizeof(float));

                // Accumulate the gathered distances
                approx_dist_vec = _mm256_add_ps(approx_dist_vec, gathered_dists);
            }

            // Store the 8 accumulated distances
            float approx_dists_arr[8];
            _mm256_storeu_ps(approx_dists_arr, approx_dist_vec);

            // Push results into the priority queue
            for (int j = 0; j < 8; ++j) {
                float approx_dist = approx_dists_arr[j];
                uint32_t current_id = (uint32_t)(i + j);
                if (approx_result_pq.size() < rerank_k) {
                    approx_result_pq.push({approx_dist, current_id});
                } else if (approx_dist < approx_result_pq.top().first) {
                    approx_result_pq.pop();
                    approx_result_pq.push({approx_dist, current_id});
                }
            }
        }

        // Process remaining vectors (less than 8) using scalar approach
        for (; i < num_codes; ++i) {
            float approx_dist = 0;
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
        std::reverse(candidate_ids.begin(), candidate_ids.end());

        // 4. 计算精确距离并进行重排
        std::priority_queue<std::pair<float, uint32_t>> final_result_pq;

        for (uint32_t id : candidate_ids) {
             if (id >= base_num) {
                 std::cerr << "Error: Candidate ID " << id << " out of bounds (" << base_num << ")" << std::endl;
                 continue;
             }
            const float* base_vec = base_data + (size_t)id * dim;
            float exact_dist = compute_exact_distance(query_vec, base_vec, dim);

            if (final_result_pq.size() < k) {
                final_result_pq.push({exact_dist, id});
            } else if (exact_dist < final_result_pq.top().first) {
                final_result_pq.pop();
                final_result_pq.push({exact_dist, id});
            }
        }
        
        return final_result_pq;
    }
};

// 全局PQ索引变量
static PQIndex g_pq_index;

// 更新 pq_search 包装函数以接受 rerank_k 参数
std::priority_queue<std::pair<float, uint32_t>> pq_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, int rerank_k) {
    g_pq_index.set_base_data(base, base_number); 
    return g_pq_index.query(query, k, rerank_k);
}