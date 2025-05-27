#pragma once
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
// Windows线程相关头文件
#include <windows.h>
#include <process.h>
#include <stdint.h> // 用于intptr_t

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

/* ---------- 同步原语 - 使用Windows事件而不是条件变量 ---------- */
// 简单的Windows事件基屏障(Barrier)实现
class WinBarrier {
private:
    HANDLE* threadReady;    // 线程就绪事件数组
    HANDLE* threadProceed;  // 线程可继续事件数组
    int count;              // 线程数量
    volatile int waiting;   // 等待中的线程数
    volatile bool destroyed;// 是否已销毁

public:
    WinBarrier(int count) : count(count), waiting(0), destroyed(false) {
        threadReady = new HANDLE[count];
        threadProceed = new HANDLE[count];
        
        for (int i = 0; i < count; i++) {
            // 手动重置，初始非触发
            threadReady[i] = CreateEvent(NULL, TRUE, FALSE, NULL);
            threadProceed[i] = CreateEvent(NULL, TRUE, FALSE, NULL);
        }
    }
    
    ~WinBarrier() {
        destroyed = true;
        
        for (int i = 0; i < count; i++) {
            SetEvent(threadProceed[i]); // 唤醒所有可能处于等待状态的线程
        }
        
        Sleep(10); // 给线程一点时间来退出
        
        for (int i = 0; i < count; i++) {
            CloseHandle(threadReady[i]);
            CloseHandle(threadProceed[i]);
        }
        
        delete[] threadReady;
        delete[] threadProceed;
    }
    
    // 线程调用此函数等待屏障
    bool wait(int threadId) {
        if (destroyed) return false;
        
        // 通知主线程此线程已就绪
        SetEvent(threadReady[threadId]);
        
        // 等待许可继续
        DWORD result = WaitForSingleObject(threadProceed[threadId], INFINITE);
        ResetEvent(threadProceed[threadId]); // 复位事件，为下次使用做准备
        
        return result == WAIT_OBJECT_0;
    }
    
    // 主线程调用，释放所有线程
    void release() {
        if (destroyed) return;
        
        // 等待所有工作线程就绪
        for (int i = 1; i < count; i++) {
            WaitForSingleObject(threadReady[i], INFINITE);
            ResetEvent(threadReady[i]);
        }
        
        // 通知所有线程可以继续
        for (int i = 1; i < count; i++) {
            SetEvent(threadProceed[i]);
        }
    }
};

static WinBarrier* g_barrier = nullptr;

/* ---------- 工作线程函数 ---------- */
unsigned __stdcall worker_loop(void* arg)
{
    intptr_t tid = (intptr_t)arg;          // 线程编号 [1..7]
    
    try {
        while(true) {
            // 等待主线程通知继续
            if (g_barrier && !g_barrier->wait(tid)) {
                break;
            }

            // 检查是否需要退出
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
                int re = std::min(rs + chunk, g_real_probe);

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

            // 通知主线程此阶段完成
            if (g_barrier && !g_barrier->wait(tid)) {
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Thread " << tid << " exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Thread " << tid << " unknown exception" << std::endl;
    }
    
    return 0;
}

/* ---------- 初始化：创建线程 ---------- */
inline void init_static_threads()
{
    static bool inited = false;
    if(inited) return;
    
    try {
        // 创建屏障对象
        g_barrier = new WinBarrier(NUM_THREADS);
        
        // 创建工作线程
        for(intptr_t t = 1; t <= NUM_CHILD; ++t){
            HANDLE hThread = (HANDLE)_beginthreadex(NULL, 0, worker_loop, (void*)t, 0, NULL);
            if (hThread) {
                CloseHandle(hThread);  // 分离线程
            } else {
                std::cerr << "Error creating thread " << t << ": " << GetLastError() << std::endl;
            }
        }
        
        inited = true;
    } catch (const std::exception& e) {
        std::cerr << "Thread initialization exception: " << e.what() << std::endl;
        if (g_barrier) { delete g_barrier; g_barrier = nullptr; }
        throw;
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

    // 通知所有线程开始
    g_barrier->release();

    /* ---- 主线程自己的 slice ---- */
    {
        int nlist = g_ivf_index.nlist;
        int chunk = (nlist + NUM_THREADS - 1) / NUM_THREADS;
        int st = 0;  // 主线程处理第一个chunk
        int ed = std::min(st + chunk, nlist);
        for(int cid = st; cid < ed; ++cid){
            float d = l2_distance(query,
                                g_ivf_index.centroids[cid].data(),
                                (int)dim);
            cent_dists[cid] = {d, cid};
        }
    }

    // 等待所有线程完成
    g_barrier->release();

    std::partial_sort(cent_dists.begin(),
                    cent_dists.begin() + std::min(nprobe, g_ivf_index.nlist),
                    cent_dists.end());

    /* ============ Stage-B : scan lists ============ */
    int real_probe      = std::min(nprobe, g_ivf_index.nlist);
    g_real_probe        = real_probe;

    static std::priority_queue<DistIdx> local_heaps[NUM_THREADS];
    for(int t = 0; t < NUM_THREADS; ++t) { 
        while(!local_heaps[t].empty()) local_heaps[t].pop(); 
    }
    g_local_heaps       = local_heaps;

    g_stage = STAGE_SCAN;
    // 通知所有线程开始扫描
    g_barrier->release();

    /* ---- 主线程的 list slice ---- */
    {
        int chunk = (real_probe + NUM_THREADS - 1) / NUM_THREADS;
        int rs = 0;
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

    // 等待所有线程完成扫描
    g_barrier->release();

    /* ============ merge & return ============ */
    std::priority_queue<DistIdx> topk;
    for(int t = 0; t < NUM_THREADS; ++t){
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

/* ---------- 程序退出时，销毁线程资源 ---------- */
inline void destroy_static_threads()
{
    if (g_barrier != nullptr) {
        g_stage = STAGE_EXIT;
        g_barrier->release();  // 通知所有线程退出
        
        // 给线程一点时间来退出
        Sleep(100);
        
        delete g_barrier;
        g_barrier = nullptr;
    }
}