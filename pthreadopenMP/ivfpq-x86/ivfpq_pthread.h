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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <future>

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

/* ===================== 线程池实现 ========================= */
class ThreadPool {
public:
    ThreadPool(size_t numThreads) : stop(false) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(this->queueMutex);
                        this->condition.wait(lock, [this] { 
                            return this->stop || !this->tasks.empty(); 
                        });
                        
                        if (this->stop && this->tasks.empty())
                            return;
                            
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    
                    task();
                }
            });
        }
    }
    
    // 添加任务到线程池
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }
    
    // 等待所有任务完成
    void wait() {
        std::unique_lock<std::mutex> lock(queueMutex);
        condition.wait(lock, [this] { return tasks.empty() && active_tasks == 0; });
    }
    
    // 析构函数中停止所有线程
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            if (worker.joinable())
                worker.join();
        }
    }
    
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    std::atomic<int> active_tasks{0};
    bool stop;
    
    friend void increment_active(ThreadPool* pool);
    friend void decrement_active(ThreadPool* pool);
};

// 全局线程池实例
ThreadPool g_thread_pool(7); // 7个工作线程

inline void increment_active(ThreadPool* pool) {
    pool->active_tasks++;
}

inline void decrement_active(ThreadPool* pool) {
    pool->active_tasks--;
    pool->condition.notify_all();
}

/* ===================== 工具函数 =========================== */
inline float l2_distance(const float* a, const float* b, int dim) {
    float s = 0.f;
    for (int i = 0; i < dim; ++i) {
        float d = a[i] - b[i];  s += d * d;
    }
    return s;
}

/* ===================== 线程结构体 ========================= */
struct WorkerArg {
    /* 只读 */
    const float* base;
    const float* query;
    const std::vector<int>* probe_ids;     // 需要处理的簇 id
    int dim;
    size_t k_keep;
    int begin, end;                        // probe_ids 区间

    /* 输出 */
    std::priority_queue<std::pair<float,int>> local_heap;
};

/* ------------------- 线程主函数 --------------------------- */
static void worker_entry(WorkerArg* arg)
{
    auto* wa = arg;
    const int M = g_ivfpq_index.M;
    const int K = g_ivfpq_index.K;
    const int dim = wa->dim;
    const int sub_dim = dim / M;
    const size_t kkeep = wa->k_keep;

    /* 私有缓冲 */
    std::vector<float> q_residual(dim);
    std::vector<float> dist_table(M * K);
    auto& heap = wa->local_heap;

    for (int pos = wa->begin; pos < wa->end; ++pos)
    {
        int list_id = (*wa->probe_ids)[pos];
        if (g_ivfpq_index.invlists[list_id].empty()) continue;

        /* ---- 残差 ---- */
        const auto& cen = g_ivfpq_index.centroids[list_id];
        for (int d=0; d<dim; ++d) q_residual[d] = wa->query[d] - cen[d];

        /* ---- Build LUT ---- */
        for (int m=0; m<M; ++m){
            const float* qsub = q_residual.data() + m*sub_dim;
            for (int k=0; k<K; ++k){
                float dist=0.f;
                const float* c = g_ivfpq_index.codebooks[list_id][m][k].data();
                for (int d=0; d<sub_dim; ++d){
                    float diff = qsub[d]-c[d]; dist+=diff*diff;
                }
                dist_table[m*K + k] = dist;
            }
        }

        /* ---- 扫描倒排表 ---- */
        const auto& ids   = g_ivfpq_index.invlists[list_id];
        const auto& codes = g_ivfpq_index.codes[list_id];
        for (size_t i=0;i<ids.size();++i){
            float adist=0.f;
            for (int m=0;m<M;++m){
                uint8_t code = codes[i][m];
                adist += dist_table[m*K + code];
            }
            if (heap.size()<kkeep)               heap.emplace(adist, ids[i]);
            else if (adist < heap.top().first){  heap.pop(); heap.emplace(adist, ids[i]); }
        }
    }
}

/* ======================= 对外接口 ========================= */
inline std::priority_queue<std::pair<float,int>>
ivfpq_search(const float* base,
             const float* query,
             size_t       base_number,   // 保留接口，内部未使用
             size_t       dim,
             size_t       k,
             int          nprobe,
             int          rerank_k = 0)  // 0 = 不精排
{
    using DistIdx = std::pair<float,int>;

    /* 1. 选 nprobe 质心（串行） */
    std::vector<DistIdx> cdists;
    cdists.reserve(g_ivfpq_index.nlist);
    for (int cid=0; cid<g_ivfpq_index.nlist; ++cid){
        float d = l2_distance(query, g_ivfpq_index.centroids[cid].data(), dim);
        cdists.emplace_back(d, cid);
    }
    std::partial_sort(cdists.begin(),
                      cdists.begin()+std::min(nprobe,g_ivfpq_index.nlist),
                      cdists.end());
    std::vector<int> probe_ids;
    for (int i=0;i<std::min(nprobe,g_ivfpq_index.nlist);++i)
        probe_ids.push_back(cdists[i].second);

    /* 2. 线程划分: 8 线程 (7 child + main) */
    const int THREAD_NUM = 8;
    const int child_num  = THREAD_NUM - 1;
    const size_t keep_per_thread =
        (rerank_k>0) ? (size_t)rerank_k : k;

    int chunk = (probe_ids.size() + THREAD_NUM - 1) / THREAD_NUM;
    std::vector<WorkerArg> wargs(THREAD_NUM);
    std::mutex result_mutex;
    
    // 记录已完成的线程数
    std::atomic<int> completed_threads(0);
    std::condition_variable completion_cv;
    std::mutex completion_mutex;

    /* -- 使用线程池执行子线程任务 -- */
    for (int t=0; t<child_num; ++t)
    {
        wargs[t].base   = base;
        wargs[t].query  = query;
        wargs[t].probe_ids = &probe_ids;
        wargs[t].dim    = dim;
        wargs[t].k_keep = keep_per_thread;
        wargs[t].begin  = t*chunk;
        wargs[t].end    = std::min<int>((t+1)*chunk, probe_ids.size());
        
        g_thread_pool.enqueue([t, &wargs, &completed_threads, &completion_cv, &completion_mutex]() {
            worker_entry(&wargs[t]);
            
            // 标记线程完成
            {
                std::lock_guard<std::mutex> lock(completion_mutex);
                completed_threads++;
            }
            completion_cv.notify_all();
        });
    }
    
    /* -- 主线程也当 worker -- */
    int mt = child_num;   // index of main-thread arg
    wargs[mt].base   = base;
    wargs[mt].query  = query;
    wargs[mt].probe_ids = &probe_ids;
    wargs[mt].dim    = dim;
    wargs[mt].k_keep = keep_per_thread;
    wargs[mt].begin  = mt*chunk;
    wargs[mt].end    = std::min<int>((mt+1)*chunk, probe_ids.size());
    worker_entry(&wargs[mt]);

    /* -- 等待所有子线程完成 -- */
    {
        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_cv.wait(lock, [&completed_threads, child_num]() { 
            return completed_threads >= child_num; 
        });
    }

    /* 3. 合并局部堆 */
    std::priority_queue<DistIdx> approx_heap;
    for (int t=0;t<THREAD_NUM;++t){
        auto& h = wargs[t].local_heap;
        while(!h.empty()){
            if (approx_heap.size()<keep_per_thread)    approx_heap.push(h.top());
            else if (h.top().first < approx_heap.top().first){
                approx_heap.pop(); approx_heap.push(h.top());
            }
            h.pop();
        }
    }

    /* 4. 可选精排 */
    std::priority_queue<DistIdx> final_heap;
    if (rerank_k>0){
        std::vector<DistIdx> cand;
        cand.reserve(approx_heap.size());
        while(!approx_heap.empty()){ cand.push_back(approx_heap.top()); approx_heap.pop(); }

        int R = std::min<int>(cand.size(), rerank_k);
        for (int i=0;i<R;++i){
            int id = cand[i].second;
            const float* vec = base + (size_t)id*dim;
            float dist = l2_distance(query, vec, dim);

            if (final_heap.size()<k)                final_heap.emplace(dist,id);
            else if (dist < final_heap.top().first){
                final_heap.pop(); final_heap.emplace(dist,id);
            }
        }
    }else{
        while(!approx_heap.empty()){
            final_heap.push(approx_heap.top()); approx_heap.pop();
            if (final_heap.size()>k) final_heap.pop();
        }
    }

    /* 5. 返回 (-dist,id) 与旧接口保持一致 */
    std::priority_queue<DistIdx> ret;
    while(!final_heap.empty()){
        ret.emplace(-final_heap.top().first, final_heap.top().second);
        final_heap.pop();
    }
    return ret;
}