#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <arm_neon.h> // ARM NEON 指令集
#include <cassert>
#include <queue>
#include <cstring>
#include <algorithm>
#include <utility> // Added for std::pair
#include <limits> // Added for std::numeric_limits

// PQ配置参数
constexpr int PQ_M = 16;  // 子空间数量
constexpr int PQ_K = 16; // 聚类数量

// SIMD宽度 (处理4个float或4个int)
constexpr int SIMD_WIDTH = 4;

// PQ索引结构
struct PQIndex {
    int M; // 子空间数量
    int K; // 每个子空间的聚类数
    int dim; // 原始向量维度
    int sub_dim; // 每个子空间的维度 (dim / M)

    std::vector<std::vector<std::vector<float>>> codebooks;  // 码本 [M][K][sub_dim]
    std::vector<std::vector<uint8_t>> codes;                // 编码后的数据 [n][M]

    // 查询时用的距离表 - M * K floats, aligned for potential future optimization
    float* dist_tables;

    // 添加原始数据指针以进行重排
    const float* base_data;
    size_t base_num;

    PQIndex(int m = PQ_M, int k = PQ_K) : M(m), K(k), dim(0), sub_dim(0), dist_tables(nullptr), base_data(nullptr), base_num(0) {}

    ~PQIndex() {
        // Use aligned free if allocated aligned memory
        // For simplicity now, using standard delete[]
        if (dist_tables) {
            delete[] dist_tables;
        }
    }

    // 使用NEON加速计算距离表 (欧氏距离平方)
    void compute_distance_table_neon(const float* query) {
        if (!dist_tables) {
            
            dist_tables = new float[M * K];
            if (!dist_tables) {
                 std::cerr << "Error: Failed to allocate memory for dist_tables." << std::endl;
                
                 exit(1); // Or other error handling
            }
        }

        // 计算每个子空间的距离表
        for (int m = 0; m < M; ++m) {
            const float* query_sub = query + m * sub_dim;
            float* current_dist_table = dist_tables + m * K;

            for (int k = 0; k < K; ++k) {
                const float* centroid = codebooks[m][k].data();
                float dist_sq = 0.0f;

                
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                int d = 0;
               
                for (; d <= sub_dim - SIMD_WIDTH; d += SIMD_WIDTH) {
                    float32x4_t q_vec = vld1q_f32(query_sub + d);
                    float32x4_t c_vec = vld1q_f32(centroid + d);
                    float32x4_t diff = vsubq_f32(q_vec, c_vec);
                    
                     sum_vec = vaddq_f32(sum_vec, vmulq_f32(diff, diff));
                }

                #ifdef __aarch64__
                    dist_sq = vaddvq_f32(sum_vec);
                #else
                    
                    float32x2_t sum_pair = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
                    dist_sq = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
                #endif


               
                for (; d < sub_dim; ++d) {
                    float diff = query_sub[d] - centroid[d];
                    dist_sq += diff * diff;
                }
                current_dist_table[k] = dist_sq;
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
        if (M <= 0 || K <= 0 || dim <= 0 || dim % M != 0) {
             std::cerr << "Error: Invalid PQ parameters loaded from file." << std::endl;
             fin.close();
             return false;
        }
        sub_dim = dim / M;

        // 读取码本
        codebooks.resize(M);
        for (int m = 0; m < M; m++) {
            codebooks[m].resize(K);
            for (int k = 0; k < K; k++) {
                codebooks[m][k].resize(sub_dim);
                fin.read(reinterpret_cast<char*>(codebooks[m][k].data()), sub_dim * sizeof(float));
                 if (!fin) {
                    std::cerr << "Error reading codebook data from file." << std::endl;
                    return false;
                 }
            }
        }

        // 读取编码数据
        size_t n_read; // Use size_t for consistency
        fin.read(reinterpret_cast<char*>(&n_read), sizeof(size_t)); // Assuming size_t was written
         // If int was written:
         // int n_int; fin.read(reinterpret_cast<char*>(&n_int), sizeof(int)); n_read = n_int;
        if (!fin || n_read > 100000000) { // Basic sanity check on size
            std::cerr << "Error reading or invalid number of codes from file." << std::endl;
            return false;
        }
        codes.resize(n_read);
        for (size_t i = 0; i < n_read; i++) {
            codes[i].resize(M);
            fin.read(reinterpret_cast<char*>(codes[i].data()), M * sizeof(uint8_t));
            if (!fin) {
                std::cerr << "Error reading codes data for vector " << i << " from file." << std::endl;
                return false;
            }
        }

        fin.close();
        std::cout << "PQ Index loaded: M=" << M << ", K=" << K << ", dim=" << dim << ", sub_dim=" << sub_dim << ", num_codes=" << codes.size() << std::endl;
        return true;
    }

    // 设置原始数据指针，用于重排
    void set_base_data(const float* base, size_t n) {
        base_data = base;
        base_num = n;
        // Safety check: Ensure loaded codes match base_num if possible
        if (base_num != codes.size() && !codes.empty()) {
            std::cerr << "Warning: Base data size (" << n << ") does not match loaded codes size (" << codes.size() << ")." << std::endl;
            
        }
    }

    // 计算两个向量之间的精确平方欧氏距离 (NEON accelerated)
    float compute_exact_distance_neon(const float* vec1, const float* vec2, int dimension) {
        float dist_sq = 0.0f;
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        int d = 0;

        for (; d <= dimension - SIMD_WIDTH; d += SIMD_WIDTH) {
            float32x4_t v1 = vld1q_f32(vec1 + d);
            float32x4_t v2 = vld1q_f32(vec2 + d);
            float32x4_t diff = vsubq_f32(v1, v2);
            // sum_vec = vmlaq_f32(sum_vec, diff, diff); // FMA
             sum_vec = vaddq_f32(sum_vec, vmulq_f32(diff, diff)); // Non-FMA
        }

        #ifdef __aarch64__
            dist_sq = vaddvq_f32(sum_vec);
        #else
            float32x2_t sum_pair = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
            dist_sq = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
        #endif


        for (; d < dimension; ++d) {
            float diff = vec1[d] - vec2[d];
            dist_sq += diff * diff;
        }
        return dist_sq;
    }

    // 使用预计算的距离表进行查询，并进行重排 (NEON accelerated approximate search)
    std::priority_queue<std::pair<float, uint32_t>> query(const float* query_vec, int k, int rerank_k) {
        if (!base_data) {
            std::cerr << "Error: Base data not set for reranking." << std::endl;
            return {}; // Return empty queue
        }
         if (codes.empty()) {
            std::cerr << "Error: PQ codes are empty." << std::endl;
            return {};
         }

        size_t num_vectors = codes.size(); // Use the actual number of loaded codes

         if (rerank_k < k) {
             std::cerr << "Warning: rerank_k (" << rerank_k << ") is less than k (" << k << "). Setting rerank_k = k." << std::endl;
             rerank_k = k;
         }
         if (rerank_k > num_vectors) {
             std::cerr << "Warning: rerank_k (" << rerank_k << ") is greater than the number of base vectors (" << num_vectors << "). Setting rerank_k to base size." << std::endl;
             rerank_k = num_vectors;
         }
         if (rerank_k <= 0) { // Handle cases where rerank_k might become 0 or negative
              rerank_k = std::min((size_t)k, num_vectors); // Default to k or num_vectors if k is too large
              if (rerank_k <= 0) return {}; // If k is also 0 or negative, return empty
         }


        // 1. 计算距离表 (NEON accelerated)
        compute_distance_table_neon(query_vec);

        // 2. 使用 PQ 近似距离进行初步检索 (NEON accelerated loop)
        // Use a max-heap (default priority_queue) but store negative distance to find smallest distances effectively?
        // Or stick to standard max-heap and manage size. Let's stick to standard.
        std::priority_queue<std::pair<float, uint32_t>> approx_result_pq; // Max-heap for approx distances


        size_t i = 0;
        // Process 4 vectors at a time using NEON
        for (; i <= num_vectors - SIMD_WIDTH; i += SIMD_WIDTH) {
            // Accumulators for 4 vectors, initialized to 0
            float32x4_t dist_accum = vdupq_n_f32(0.0f);

            // Iterate through sub-quantizers
            for (int m = 0; m < M; ++m) {
                const float* current_dist_table = dist_tables + m * K;

                
                uint8_t code0 = codes[i + 0][m];
                uint8_t code1 = codes[i + 1][m];
                uint8_t code2 = codes[i + 2][m];
                uint8_t code3 = codes[i + 3][m];

               
                float d0 = current_dist_table[code0];
                float d1 = current_dist_table[code1];
                float d2 = current_dist_table[code2];
                float d3 = current_dist_table[code3];

               
                float32x4_t dist_vec = vdupq_n_f32(0.0f); // Initialize
                dist_vec = vsetq_lane_f32(d0, dist_vec, 0);
                dist_vec = vsetq_lane_f32(d1, dist_vec, 1);
                dist_vec = vsetq_lane_f32(d2, dist_vec, 2);
                dist_vec = vsetq_lane_f32(d3, dist_vec, 3);


                // Accumulate distances
                dist_accum = vaddq_f32(dist_accum, dist_vec);
            }

            // Store the accumulated distances for the 4 vectors
            float approx_dists[SIMD_WIDTH];
            vst1q_f32(approx_dists, dist_accum);

            // Update the priority queue for each of the 4 results
            for (int j = 0; j < SIMD_WIDTH; ++j) {
                 uint32_t current_id = static_cast<uint32_t>(i + j);
                 float current_dist = approx_dists[j];

                 if (approx_result_pq.size() < rerank_k) {
                     approx_result_pq.push({current_dist, current_id});
                 } else if (current_dist < approx_result_pq.top().first) {
                     approx_result_pq.pop();
                     approx_result_pq.push({current_dist, current_id});
                 }
            }
        }

        // Process remaining vectors (< SIMD_WIDTH) with scalar code
        for (; i < num_vectors; ++i) {
            float approx_dist = 0;
            for (int m = 0; m < M; ++m) {
                uint8_t code = codes[i][m];
                approx_dist += dist_tables[m * K + code];
            }

             uint32_t current_id = static_cast<uint32_t>(i);
             if (approx_result_pq.size() < rerank_k) {
                 approx_result_pq.push({approx_dist, current_id});
             } else if (approx_dist < approx_result_pq.top().first) {
                 approx_result_pq.pop();
                 approx_result_pq.push({approx_dist, current_id});
             }
        }


        // 3. Extract rerank_k candidate IDs
         // Ensure we don't try to extract more than available or more than rerank_k
         size_t num_candidates = std::min((size_t)rerank_k, approx_result_pq.size());
         std::vector<uint32_t> candidate_ids;
         candidate_ids.reserve(num_candidates);
        
         std::vector<std::pair<float, uint32_t>> temp_candidates;
         temp_candidates.reserve(approx_result_pq.size());
         while (!approx_result_pq.empty()) {
             temp_candidates.push_back(approx_result_pq.top());
             approx_result_pq.pop();
         }
         // Sort by distance ascending (pairs sort by first element by default)
         std::sort(temp_candidates.begin(), temp_candidates.end());
         // Take the top 'num_candidates' (which are the first ones after sorting)
         for (size_t j = 0; j < num_candidates; ++j) {
              candidate_ids.push_back(temp_candidates[j].second);
         }
        // No need to reverse now, as we sorted ascending


        // 4. 计算精确距离并进行重排 (NEON accelerated)
        std::priority_queue<std::pair<float, uint32_t>> final_result_pq; // Max-heap for exact distances

        for (uint32_t id : candidate_ids) {
            if (id >= base_num) { // Safety check against base_data bounds
                std::cerr << "Error: Candidate ID " << id << " out of bounds (" << base_num << ")" << std::endl;
                continue;
            }
            const float* base_vec = base_data + (size_t)id * dim; // Get original vector pointer
            float exact_dist = compute_exact_distance_neon(query_vec, base_vec, dim);

            if (final_result_pq.size() < k) {
                final_result_pq.push({exact_dist, id});
            } else if (exact_dist < final_result_pq.top().first) {
                final_result_pq.pop();
                final_result_pq.push({exact_dist, id});
            }
        }

        // 5. 返回最终 top-k 结果
        return final_result_pq;
    }
};

// 全局PQ索引变量
static PQIndex g_pq_index;

// 更新 pq_search 包装函数以接受 rerank_k 参数
std::priority_queue<std::pair<float, uint32_t>> pq_search(
    float* base,        
    float* query,      
    size_t base_number,
    size_t vecdim,    
    size_t k,
    int rerank_k)
{
   
    
    g_pq_index.set_base_data(base, base_number);

  
    if (g_pq_index.dim != 0 && g_pq_index.dim != vecdim) {
         std::cerr << "Error: Query dimension (" << vecdim
                   << ") does not match loaded PQ index dimension (" << g_pq_index.dim << ")" << std::endl;
         return {}; 
    }
    if (g_pq_index.dim == 0) { 
         std::cerr << "Error: PQ index dimension is zero. Was the index loaded/built?" << std::endl;
         return {};
    }

    // Use PQ索引查询并进行重排
    return g_pq_index.query(query, k, rerank_k);
}
