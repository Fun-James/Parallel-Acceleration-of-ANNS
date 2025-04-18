#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <arm_neon.h>  // ARM NEON 指令集
#include <cassert>
#include <queue>
#include <cstring>

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
    
    PQIndex(int m = PQ_M, int k = PQ_K) : M(m), K(k), dist_tables(nullptr) {}
    
    ~PQIndex() {
        if (dist_tables) {
            delete[] dist_tables;
        }
    }

    // 使用NEON加速计算内积
    float compute_inner_product_neon(const float* a, const float* b, int n) {
        float32x4_t sum_vec = vdupq_n_f32(0);
        int i = 0;
        
        // 每次处理4个元素
        for (; i <= n - 4; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            sum_vec = vmlaq_f32(sum_vec, va, vb);
        }
        
        // 水平求和
        float sum = vgetq_lane_f32(sum_vec, 0) + 
                   vgetq_lane_f32(sum_vec, 1) +
                   vgetq_lane_f32(sum_vec, 2) + 
                   vgetq_lane_f32(sum_vec, 3);
        
        // 处理剩余元素
        for (; i < n; i++) {
            sum += a[i] * b[i];
        }
        
        return sum;
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
    
    // 使用预计算的距离表进行查询
    std::priority_queue<std::pair<float, uint32_t>> query(const float* query_vec, int k) {
        // 计算距离表
        compute_distance_table_neon(query_vec);
        
        // 使用距离表进行查询
        std::priority_queue<std::pair<float, uint32_t>> result;
        
        for (size_t i = 0; i < codes.size(); i++) {
            float dist = 0;
            
            // 累加每个子空间的距离
            for (int m = 0; m < M; m++) {
                uint8_t code = codes[i][m];
                dist += dist_tables[m * K + code];
            }
            
            if (result.size() < k) {
                result.push(std::make_pair(dist, i));
            } else if (dist < result.top().first) {
                result.push(std::make_pair(dist, i));
                result.pop();
            }
        }
        
        return result;
    }
};

// 全局PQ索引变量
static PQIndex g_pq_index;


std::priority_queue<std::pair<float, uint32_t>> pq_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    // 使用PQ索引查询
    return g_pq_index.query(query, k);
}