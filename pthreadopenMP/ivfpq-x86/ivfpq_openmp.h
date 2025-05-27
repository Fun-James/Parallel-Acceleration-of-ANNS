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

extern IVFPQIndex g_ivfpq_index;

/* -------------------- 工具函数 -------------------- */
inline float l2_distance(const float* a, const float* b, int d) {
    float s = 0.f;
    for (int i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        s += diff * diff;
    }
    return s;
}

/* -------------------- 并行搜索 -------------------- */
inline std::priority_queue<std::pair<float,int>>
ivfpq_search(const float* base,
             const float* query,
             size_t       /*base_n*/,     // 保留接口，内部未用
             size_t       dim,
             size_t       k,
             int          nprobe,
             int          rerank_k = 0)   // 0 = 不精排
{
    using DistIdx = std::pair<float,int>;
    const int M   = g_ivfpq_index.M;
    const int K   = g_ivfpq_index.K;
    const int sub_dim = dim / M;

    /* =============================================================
     * 1. 计算 query→centroid 距离，并选 nprobe 个最小者
     * ============================================================*/
    std::vector<DistIdx> cent_dists(g_ivfpq_index.nlist);

    #pragma omp parallel for schedule(static)
    for (int cid=0; cid<g_ivfpq_index.nlist; ++cid) {
        float d = l2_distance(query,
                              g_ivfpq_index.centroids[cid].data(),
                              static_cast<int>(dim));
        cent_dists[cid] = {d, cid};
    }

    const int real_probe = std::min(nprobe, g_ivfpq_index.nlist);
    std::partial_sort(cent_dists.begin(),
                      cent_dists.begin() + real_probe,
                      cent_dists.end());

    /* 提取要扫描的簇 id 列表 */
    std::vector<int> probe_ids(real_probe);
    for (int i=0;i<real_probe;++i) probe_ids[i] = cent_dists[i].second;

    /* PQ 阶段需要保留的候选数 */
    const size_t keep_pq = (rerank_k>0) ? (size_t)rerank_k : k;

    /* 线程局部堆数组 */
    const int T = omp_get_max_threads();
    std::vector<std::priority_queue<DistIdx>> local_heaps(T);

    /* =============================================================
     * 2. 并行扫描 nprobe 个倒排表
     * ============================================================*/
    #pragma omp parallel
{
    /* ------------- 线程私有缓冲一次分配 ------------- */
    std::vector<float> q_residual(dim);
    std::vector<float> lut(M*K);

    #pragma omp for schedule(dynamic)
    for (int pi = 0; pi < real_probe; ++pi) {
        int list_id = probe_ids[pi];
        const auto& ids   = g_ivfpq_index.invlists[list_id];
        if (ids.empty()) continue;

        /* (a) 计算残差 */
        const float* cen = g_ivfpq_index.centroids[list_id].data();
        #pragma omp simd
        for (int d=0; d<dim; ++d) q_residual[d] = query[d] - cen[d];

        /* (b) 构 LUT */
        for (int m=0; m<M; ++m) {
            const float* qs = q_residual.data() + m*sub_dim;
            for (int k2=0; k2<K; ++k2) {
                const float* c = g_ivfpq_index.codebooks[list_id][m][k2].data();
                float dist = 0.f;
                #pragma omp simd reduction(+:dist)
                for (int d=0; d<sub_dim; ++d) {
                    float diff = qs[d] - c[d];
                    dist += diff * diff;
                }
                lut[m*K + k2] = dist;
            }
        }

        /* (c) 扫描倒排表 */
        auto& heap = local_heaps[omp_get_thread_num()];
        const auto& pcodes = g_ivfpq_index.codes[list_id];
        for (size_t i=0; i<ids.size(); ++i) {
            const uint8_t* code = pcodes[i].data();
            float adist = 0.f;
            #pragma omp simd reduction(+:adist)
            for (int m=0;m<M;++m)
                adist += lut[m*K + code[m]];

            if (heap.size() < keep_pq)              heap.emplace(adist, ids[i]);
            else if (adist < heap.top().first) {    heap.pop(); heap.emplace(adist, ids[i]); }
        }
    }
}

    /* =============================================================
     * 3. 合并局部堆 → approx_heap
     * ============================================================*/
    std::priority_queue<DistIdx> approx_heap;
    for (auto& h : local_heaps) {
        while (!h.empty()) {
            const auto& e = h.top();
            if (approx_heap.size() < keep_pq)             approx_heap.push(e);
            else if (e.first < approx_heap.top().first) { approx_heap.pop(); approx_heap.push(e); }
            h.pop();
        }
    }

    /* =============================================================
     * 4. 可选精排 rerank_k
     * ============================================================*/
    std::priority_queue<DistIdx> final_heap;

    if (rerank_k > 0) {
        std::vector<DistIdx> cand;
        cand.reserve(approx_heap.size());
        while(!approx_heap.empty()){ cand.push_back(approx_heap.top()); approx_heap.pop(); }

        // 注意：approx_heap 是最大堆（存储负距离）或最小堆（存储正距离）
        // 之前的代码 approx_heap 存储的是正距离，值越小越好
        // cand 是从 approx_heap 顶部（最小距离）开始取的，所以 cand[0] 是最近似的
        // 但 priority_queue 默认是最大堆，std::pair 的比较默认比较 first
        // 为了保持与原始 final_heap 逻辑一致（值越小越好），这里的 cand 是从小到大（距离）排序的
        // 如果 approx_heap 是最小堆，那么 cand 已经是按距离升序排列的
        // 如果 cand 是从大顶堆取的，需要反转一下顺序，但这里是从小顶堆取的，所以顺序是对的

        int R = std::min<int>(cand.size(), rerank_k); // R 是实际要精排的数量

        // 为精排阶段创建线程局部堆
        std::vector<std::priority_queue<DistIdx>> local_rerank_heaps(T);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < R; ++i) {
            int id = cand[i].second;
            const float* vec = base + (size_t)id * dim;
            float dist = l2_distance(query, vec, static_cast<int>(dim));
            
            auto& current_thread_heap = local_rerank_heaps[omp_get_thread_num()];
            if (current_thread_heap.size() < k) {
                current_thread_heap.emplace(dist, id);
            } else if (dist < current_thread_heap.top().first) {
                current_thread_heap.pop();
                current_thread_heap.emplace(dist, id);
            }
        }

        // 合并精排阶段的局部堆到 final_heap
        for (auto& h : local_rerank_heaps) {
            while (!h.empty()) {
                const auto& e = h.top();
                if (final_heap.size() < k) {
                    final_heap.push(e);
                } else if (e.first < final_heap.top().first) {
                    final_heap.pop();
                    final_heap.push(e);
                }
                h.pop();
            }
        }

    } else {
        /* 不精排：取 approx_heap 前 k */
        // approx_heap 是最小堆，直接转移元素，并保持 final_heap 为最小堆且大小为 k
        while(!approx_heap.empty()) {
            // final_heap 也应该是最小堆，存储实际距离
            if (final_heap.size() < k) {
                final_heap.push(approx_heap.top());
            } else if (approx_heap.top().first < final_heap.top().first) {
                final_heap.pop();
                final_heap.push(approx_heap.top());
            }
            approx_heap.pop();
        }
    }

    /* =============================================================
     * 5. 打包返回 (-dist,id)   与旧接口兼容
     * ============================================================*/
    std::priority_queue<DistIdx> ret;
    while(!final_heap.empty()) {
        ret.emplace(-final_heap.top().first, final_heap.top().second);
        final_heap.pop();
    }
    return ret;
}