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
// 替换ARM NEON头文件为x86 SIMD头文件
#include <immintrin.h> // AVX/SSE指令集

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

extern IVFPQIndex g_ivfpq_index;

/* ---------------- SIMD 版 L2 距离 ---------------- */
static inline float l2_avx(const float* a, const float* b, int dim)
{
    float sum = 0.f;
    int i = 0;
    
    // 直接使用AVX进行向量化计算
    __m256 sum_avx = _mm256_setzero_ps();
    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum_avx = _mm256_add_ps(sum_avx, _mm256_mul_ps(diff, diff));
    }
    // 水平求和
    __m128 sum_low = _mm256_extractf128_ps(sum_avx, 0);
    __m128 sum_high = _mm256_extractf128_ps(sum_avx, 1);
    sum_low = _mm_add_ps(sum_low, sum_high);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    sum += _mm_cvtss_f32(sum_low);
    
    // 处理剩余元素
    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    return sum;
}

/* ============================================================== */
/*                  OpenMP + AVX  查询主函数                      */
/* ============================================================== */
/*

*/
inline std::priority_queue<std::pair<float,int>>
ivfpq_search(const float* base,
             const float* query,
             size_t       /*base_n*/,
             size_t       dim,
             size_t       k,
             int          nprobe,
             int          rerank_k = 0)
{
    using DistIdx = std::pair<float,int>;

    const int M  = g_ivfpq_index.M;
    const int K_pq  = g_ivfpq_index.K; // Renamed K to K_pq to avoid conflict with parameter k
    const int sub_dim = dim / M;

    /* ------------ 质心距离（并行 + SIMD） ------------*/
    std::vector<DistIdx> cent_dists(g_ivfpq_index.nlist);
    #pragma omp parallel for schedule(static)
    for (int cid = 0; cid < g_ivfpq_index.nlist; ++cid) {
        float d = l2_avx(query,  // 使用l2_avx替代l2_sse
                        g_ivfpq_index.centroids[cid].data(),
                        static_cast<int>(dim));
        cent_dists[cid] = {d, cid};
    }
    const int real_probe = std::min(nprobe, g_ivfpq_index.nlist);
    std::partial_sort(cent_dists.begin(),
                      cent_dists.begin()+real_probe,
                      cent_dists.end());

    std::vector<int> probe_ids(real_probe);
    for (int i=0;i<real_probe;++i) probe_ids[i]=cent_dists[i].second;

    /* ------------ 并行扫描倒排表 ----------------------*/
    const int T   = omp_get_max_threads();
    const size_t keep_pq = (rerank_k? rerank_k : k);
    std::vector<std::priority_queue<DistIdx>> local_heaps(T);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::vector<float> residual(dim);
        std::vector<float> lut(M*K_pq); // Use K_pq
        auto& heap = local_heaps[tid];

        #pragma omp for schedule(dynamic)
        for (int pi = 0; pi < real_probe; ++pi) {
            int list_id = probe_ids[pi];
            const auto& ids = g_ivfpq_index.invlists[list_id];
            if (ids.empty()) continue;

            /* --- 残差 --- */
            const float* cen = g_ivfpq_index.centroids[list_id].data();
            #pragma omp simd
            for (int d=0; d<dim; ++d) residual[d] = query[d] - cen[d];

            /* --- LUT (AVX + SIMD) --- */
            for (int m=0; m<M; ++m) {
                const float* qs = residual.data() + m*sub_dim;
                for (int k2=0; k2<K_pq; ++k2) { // Use K_pq
                    const float* c = g_ivfpq_index.codebooks[list_id][m][k2].data();
                    // 使用 l2_avx 优化子距离计算
                    lut[m*K_pq + k2] = l2_avx(qs, c, sub_dim); // 使用l2_avx替代l2_sse
                }
            }

            /* --- 扫描表 --- */
            const auto& pcodes = g_ivfpq_index.codes[list_id];
            for (size_t i=0;i<ids.size();++i) {
                const uint8_t* code = pcodes[i].data();
                float adist = 0.f;
                #pragma omp simd reduction(+:adist)
                for (int m=0;m<M;++m)
                    adist += lut[m*K_pq + code[m]]; // Use K_pq

                if (heap.size() < keep_pq)               heap.emplace(adist, ids[i]);
                else if (adist < heap.top().first) {     heap.pop(); heap.emplace(adist, ids[i]); }
            }
        }
    }

    /* ------------ 合并局部堆 --------------------------*/
    std::priority_queue<DistIdx> approx_heap;
    for (auto& h : local_heaps) {
        while(!h.empty()){
            const auto& e = h.top();
            if (approx_heap.size()<keep_pq)              approx_heap.push(e);
            else if (e.first < approx_heap.top().first){ approx_heap.pop(); approx_heap.push(e); }
            h.pop();
        }
    }

    /* ------------ 可选精排 ----------------------------*/
    std::priority_queue<DistIdx> final_heap;
    if (rerank_k > 0 && k > 0) { // Ensure rerank_k and k are positive for meaningful reranking
        std::vector<DistIdx> cand;
        cand.reserve(approx_heap.size());
        while(!approx_heap.empty()){ 
            cand.push_back(approx_heap.top()); 
            approx_heap.pop(); 
        }
        // cand 中的元素按近似距离降序排列，我们需要按升序排列以获取最佳候选项
        std::sort(cand.begin(), cand.end(), [](const DistIdx& a, const DistIdx& b){
            return a.first < b.first; // 按近似距离升序排序
        });

        int R = std::min<int>(cand.size(), rerank_k); // 确定实际要重排序的候选数量
        
        std::vector<DistIdx> rerank_results_vec(R);

        #pragma omp parallel for schedule(static) // 并行计算精确距离
        for (int i=0;i<R;++i){
            int id = cand[i].second;
            const float* vec = base + (size_t)id*dim;
            float dist = l2_avx(query, vec, static_cast<int>(dim)); // 使用l2_avx替代l2_sse
            rerank_results_vec[i] = {dist, id}; // 存储真实距离和id
        }

        // 从 rerank_results_vec 构建 final_heap (取前 k 个)
        for(int i=0; i<R; ++i) {
            if (final_heap.size() < k) {
                final_heap.push(rerank_results_vec[i]);
            } else if (rerank_results_vec[i].first < final_heap.top().first) {
                final_heap.pop();
                final_heap.push(rerank_results_vec[i]);
            }
        }
    } else { // 无重排序 (rerank_k = 0) 或 k = 0
        // 如果 k=0，最终结果应为空。如果 rerank_k=0，approx_heap 已包含按近似距离排序的 top-k 结果。
        if (k > 0) {
            while(!approx_heap.empty()){
                final_heap.emplace(approx_heap.top());
                approx_heap.pop();
                if (final_heap.size()>k) final_heap.pop();
            }
        }
        // 如果 k=0, final_heap 保持为空
    }

    /* ------------ 返回 (-dist,id) ---------------------*/
    std::priority_queue<DistIdx> ret;
    while(!final_heap.empty()){
        ret.emplace(-final_heap.top().first, final_heap.top().second);
        final_heap.pop();
    }
    return ret;
}