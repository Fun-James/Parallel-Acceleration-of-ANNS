#pragma once
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <pthread.h>

/* ---------------- IVF 索引结构 ---------------- */
struct IVFIndex {
    int nlist = 0;   // 聚类中心数
    int dim   = 0;   // 维度

    std::vector<std::vector<float>> centroids;  // [nlist][dim]
    std::vector<std::vector<int>>   invlists;   // 倒排表
    std::vector<float> rearranged_base;         // 重排后的向量数据
    std::vector<int> id_map;                    // 重排ID到原始ID的映射

    bool load(const std::string& fn) {
        std::ifstream fin(fn, std::ios::binary);
        if (!fin) return false;
        fin.read(reinterpret_cast<char*>(&nlist), sizeof(int));
        fin.read(reinterpret_cast<char*>(&dim),   sizeof(int));

        centroids.assign(nlist, std::vector<float>(dim));
        for (int i = 0; i < nlist; ++i)
            fin.read(reinterpret_cast<char*>(centroids[i].data()),
                     dim*sizeof(float));

        invlists.resize(nlist);
        for (int i = 0; i < nlist; ++i){
            int sz=0; fin.read(reinterpret_cast<char*>(&sz), sizeof(int));
            invlists[i].resize(sz);
            fin.read(reinterpret_cast<char*>(invlists[i].data()),
                     sz*sizeof(int));
        }
        
        // 尝试加载重排数据
        int total_vectors = 0;
        if (fin.read(reinterpret_cast<char*>(&total_vectors), sizeof(int))) {
            if (total_vectors > 0) {
                rearranged_base.resize(total_vectors * dim);
                fin.read(reinterpret_cast<char*>(rearranged_base.data()), 
                        total_vectors * dim * sizeof(float));
                        
                id_map.resize(total_vectors);
                fin.read(reinterpret_cast<char*>(id_map.data()),
                        total_vectors * sizeof(int));
            }
        }
        
        fin.close(); return true;
    }

    void save(const std::string& fn) const {
        std::ofstream fout(fn, std::ios::binary);
        fout.write(reinterpret_cast<const char*>(&nlist), sizeof(int));
        fout.write(reinterpret_cast<const char*>(&dim),   sizeof(int));

        for (int i = 0; i < nlist; ++i)
            fout.write(reinterpret_cast<const char*>(centroids[i].data()),
                       dim*sizeof(float));

        for (int i = 0; i < nlist; ++i){
            int sz = (int)invlists[i].size();
            fout.write(reinterpret_cast<const char*>(&sz), sizeof(int));
            fout.write(reinterpret_cast<const char*>(invlists[i].data()),
                       sz*sizeof(int));
        }
        
        // 保存重排数据
        int total_vectors = (int)rearranged_base.size() / dim;
        fout.write(reinterpret_cast<const char*>(&total_vectors), sizeof(int));
        if (total_vectors > 0) {
            fout.write(reinterpret_cast<const char*>(rearranged_base.data()),
                      total_vectors * dim * sizeof(float));
            fout.write(reinterpret_cast<const char*>(id_map.data()),
                      total_vectors * sizeof(int));
        }
        
        fout.close();
    }
};

extern IVFIndex g_ivf_index;     // 在 main.cc 定义一次

/* ------------------------------------------------------------ */
/*                      utility functions                       */
/* ------------------------------------------------------------ */
inline float l2_distance(const float* a,const float* b,int dim){
    float s=0.f; for(int i=0;i<dim;++i){ float d=a[i]-b[i]; s+=d*d; } return s;
}

/* ------------------------------------------------------------ */
/*                     static-thread framework                  */
/* ------------------------------------------------------------ */
static const int NUM_THREADS = 8;          // 总并行线程（含主线程）
static const int NUM_CHILD   = NUM_THREADS - 1;

using DistIdx = std::pair<float,int>;

/* ---------- 全局共享状态（阶段参数） ---------- */
enum Stage { STAGE_IDLE, STAGE_CENTER, STAGE_SCAN, STAGE_EXIT };
static Stage                 g_stage           = STAGE_IDLE;
static const float*          g_query           = nullptr;
static const float*          g_base            = nullptr;
static size_t                g_dim             = 0;
static size_t                g_k               = 0;
static int                   g_real_probe      = 0;
static std::vector<DistIdx>* g_cent_dists_ptr  = nullptr;
static std::priority_queue<DistIdx>* g_local_heaps = nullptr;

/* ---------- 同步原语 ---------- */
static pthread_barrier_t g_barrier;

/* ---------- 工作线程函数 ---------- */
void* worker_loop(void* arg)
{
    long tid = (long)arg;                 // 线程编号 [1..7]，主线程是 0
    while(true){
        // 等待主线程下达"阶段开始"信号
        pthread_barrier_wait(&g_barrier);

        if(g_stage == STAGE_EXIT) break;

        /* ------------------ Stage-A: centroid 距离 ------------------ */
        if(g_stage == STAGE_CENTER){
            int nlist = g_ivf_index.nlist;
            int chunk = (nlist + NUM_THREADS - 1) / NUM_THREADS;
            int st = tid * chunk;
            int ed = std::min(st + chunk, nlist);

            auto& cd = *g_cent_dists_ptr;
            for(int cid = st; cid < ed; ++cid){
                float d = l2_distance(g_query,
                                      g_ivf_index.centroids[cid].data(),
                                      (int)g_dim);
                cd[cid] = {d, cid};
            }
        }

        /* ------------------ Stage-B: list 扫描 ------------------ */
        else if(g_stage == STAGE_SCAN){
            int chunk = (g_real_probe + NUM_THREADS - 1) / NUM_THREADS;
            int rs = tid * chunk;
            int re = std::min(rs + chunk,  g_real_probe);

            auto& cd = *g_cent_dists_ptr;
            auto& heap = g_local_heaps[tid];

            for(int r = rs; r < re; ++r){
                int lid = cd[r].second;
                const auto& inv = g_ivf_index.invlists[lid];
                for(int idx : inv){
                    float d = l2_distance(g_query,
                                      g_base + (size_t)idx * g_dim,
                                      (int)g_dim);
                    if(heap.size() < g_k) heap.emplace(d, idx);
                    else if(d < heap.top().first){
                        heap.pop(); heap.emplace(d, idx);
                    }
                }
            }
        }

        // 本阶段完成
        pthread_barrier_wait(&g_barrier);
    }
    return nullptr;
}

/* ---------- 初始化：创建 7 条子线程 & barrier ---------- */
inline void init_static_threads()
{
    static bool inited = false;
    if(inited) return;
    inited = true;

    pthread_barrier_init(&g_barrier, nullptr, NUM_THREADS);   // 包含主线程

    for(long t = 1; t <= NUM_CHILD; ++t){
        pthread_t th;
        pthread_create(&th, nullptr, worker_loop, (void*)t);
        pthread_detach(th);             // 常驻线程，无需 join
    }
}

/* ------------------------------------------------------------ */
/*                    parallel ivf_search()                     */
/* ------------------------------------------------------------ */
inline std::priority_queue<DistIdx>
ivf_search(const float* base,
           const float* query,
           size_t,                 // base_number, 保留
           size_t dim,
           size_t k,
           int    nprobe)
{
    init_static_threads();      // 确保线程池已建立
    g_query  = query;
    g_dim    = dim;
    // 如果有重排后的数据，使用重排数据，否则使用原始数据
    g_base   = g_ivf_index.rearranged_base.size() > 0 ? g_ivf_index.rearranged_base.data() : base;
    g_k      = k;

    /* ============ Stage-A : centroid distances ============ */
    std::vector<DistIdx> cent_dists(g_ivf_index.nlist);
    g_cent_dists_ptr = &cent_dists;
    g_stage = STAGE_CENTER;

    pthread_barrier_wait(&g_barrier);    // 所有线程(含主)同步开始

    /* ---- 主线程自己的 slice ---- */
    {
        int nlist = g_ivf_index.nlist;
        int chunk = (nlist + NUM_THREADS - 1) / NUM_THREADS;
        int st = 0 * chunk;
        int ed = std::min(st + chunk, nlist);
        for(int cid = st; cid < ed; ++cid){
            float d = l2_distance(query,
                                  g_ivf_index.centroids[cid].data(),
                                  (int)dim);
            cent_dists[cid] = {d, cid};
        }
    }

    pthread_barrier_wait(&g_barrier);    // 等所有线程结束

    std::partial_sort(cent_dists.begin(),
                      cent_dists.begin() + std::min(nprobe,g_ivf_index.nlist),
                      cent_dists.end());

    /* ============ Stage-B : scan lists ============ */
    int real_probe      = std::min(nprobe, g_ivf_index.nlist);
    g_real_probe        = real_probe;

    static std::priority_queue<DistIdx> local_heaps[NUM_THREADS];
    for(int t=0;t<NUM_THREADS;++t) { while(!local_heaps[t].empty()) local_heaps[t].pop(); }
    g_local_heaps       = local_heaps;

    g_stage = STAGE_SCAN;
    pthread_barrier_wait(&g_barrier);

    /* ---- 主线程的 list slice ---- */
    {
        int chunk = (real_probe + NUM_THREADS - 1) / NUM_THREADS;
        int rs = 0 * chunk;
        int re = std::min(rs + chunk, real_probe);
        auto& heap = local_heaps[0];

        for(int r = rs; r < re; ++r){
            int lid = cent_dists[r].second;
            const auto& inv = g_ivf_index.invlists[lid];
            for(int idx : inv){
                float d = l2_distance(query,
                                  g_base + (size_t)idx * dim,
                                  (int)dim);
                if(heap.size() < k) heap.emplace(d, idx);
                else if(d < heap.top().first){
                    heap.pop(); heap.emplace(d, idx);
                }
            }
        }
    }

    pthread_barrier_wait(&g_barrier);     // 等所有线程结束

    /* ============ merge & return ============ */
    std::priority_queue<DistIdx> topk;
    for(int t=0;t<NUM_THREADS;++t){
        auto& heap = local_heaps[t];
        while(!heap.empty()){
            auto p = heap.top(); heap.pop();
            if(topk.size() < k) topk.push(p);
            else if(p.first < topk.top().first){
                topk.pop(); topk.push(p);
            }
        }
    }

    std::priority_queue<DistIdx> results;
    // 如果有ID映射，将重排后的ID映射回原始ID
    if (!g_ivf_index.id_map.empty()) {
        while(!topk.empty()){
            results.emplace(-topk.top().first, g_ivf_index.id_map[topk.top().second]);
            topk.pop();
        }
    } else {
        while(!topk.empty()){
            results.emplace(-topk.top().first, topk.top().second);
            topk.pop();
        }
    }
    return results;
}

/* ---------- 程序退出时，可选调用销毁线程 ---------- */
inline void destroy_static_threads()
{
    if(!pthread_barrier_destroy(&g_barrier)){      // 如果没初始化直接返回
        g_stage = STAGE_EXIT;
        pthread_barrier_wait(&g_barrier);          // 唤醒所有子线程让其退出
    }
}