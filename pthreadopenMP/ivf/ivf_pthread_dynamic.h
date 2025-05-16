#ifndef IVF_PTHREAD_DYNAMIC_H
#define IVF_PTHREAD_DYNAMIC_H

#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cmath>
#include <pthread.h>
#include <unistd.h>
#include <functional>
#include <atomic>
#include <mutex>

// IVF索引结构
struct IVFIndex {
    int nlist;                                   // 聚类中心数量
    int dim;                                     // 向量维度
    std::vector<std::vector<float>> centroids;   // 聚类中心
    std::vector<std::vector<int>> invlists;      // 倒排列表

    // 保存索引到文件
    void save(const std::string& filename) {
        std::ofstream fout(filename, std::ios::binary);
        fout.write(reinterpret_cast<const char*>(&nlist), sizeof(int));
        fout.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        
        // 保存聚类中心
        for (int i = 0; i < nlist; i++) {
            fout.write(reinterpret_cast<const char*>(centroids[i].data()), dim * sizeof(float));
        }
        
        // 保存倒排列表
        for (int i = 0; i < nlist; i++) {
            int list_size = invlists[i].size();
            fout.write(reinterpret_cast<const char*>(&list_size), sizeof(int));
            fout.write(reinterpret_cast<const char*>(invlists[i].data()), list_size * sizeof(int));
        }
        
        fout.close();
    }
    
    // 从文件加载索引
    bool load(const std::string& filename) {
        std::ifstream fin(filename, std::ios::binary);
        if (!fin.is_open()) {
            return false;
        }
        
        fin.read(reinterpret_cast<char*>(&nlist), sizeof(int));
        fin.read(reinterpret_cast<char*>(&dim), sizeof(int));
        
        // 加载聚类中心
        centroids.resize(nlist);
        for (int i = 0; i < nlist; i++) {
            centroids[i].resize(dim);
            fin.read(reinterpret_cast<char*>(centroids[i].data()), dim * sizeof(float));
        }
        
        // 加载倒排列表
        invlists.resize(nlist);
        for (int i = 0; i < nlist; i++) {
            int list_size;
            fin.read(reinterpret_cast<char*>(&list_size), sizeof(int));
            invlists[i].resize(list_size);
            fin.read(reinterpret_cast<char*>(invlists[i].data()), list_size * sizeof(int));
        }
        
        fin.close();
        return true;
    }
};

// 使用全局变量存储索引
extern IVFIndex g_ivf_index;

// 线程任务结构体
struct SearchTask {
    int cluster_idx;              // 聚类中心索引
    const float* query;           // 查询向量
    const float* base;            // 基础向量集
    int dim;                      // 向量维度
    std::vector<std::pair<float, int>>* local_results; // 线程局部结果
};

// 线程参数结构体
struct ThreadArg {
    std::vector<SearchTask> tasks;  // 线程要处理的任务
    int thread_id;                  // 线程ID
};

// 计算L2距离
inline float compute_l2_distance(const float* a, const float* b, int dim) {
    float dist = 0;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// 线程工作函数
void* search_thread(void* arg) {
    ThreadArg* thread_arg = static_cast<ThreadArg*>(arg);
    
    for (auto& task : thread_arg->tasks) {
        int cluster_idx = task.cluster_idx;
        const float* query = task.query;
        const float* base = task.base;
        int dim = task.dim;
        
        // 获取当前聚类的倒排列表
        const std::vector<int>& ids = g_ivf_index.invlists[cluster_idx];
        
        // 计算查询向量与列表中所有向量的距离
        for (int id : ids) {
            float dist = compute_l2_distance(query, base + id * dim, dim);
            task.local_results->push_back(std::make_pair(dist, id));
        }
    }
    
    return nullptr;
}

// 合并所有线程的结果
std::priority_queue<std::pair<float, int>> merge_results(
    std::vector<std::vector<std::pair<float, int>>>& all_results, int k) {
    
    // 创建最大堆进行Top-K选择
    std::priority_queue<std::pair<float, int>> top_results;
    
    // 将所有结果合并
    for (auto& results : all_results) {
        for (auto& result : results) {
            top_results.push(result);
            if (top_results.size() > k) {
                top_results.pop();  // 保持堆大小为k
            }
        }
    }
    
    return top_results;
}

// 主搜索函数 - 使用pthread动态线程
std::priority_queue<std::pair<float, int>> ivf_search(
    const float* base,    // 基础向量集
    const float* query,   // 查询向量
    size_t base_number,   // 基础向量数量
    size_t dim,           // 向量维度
    int k,                // 返回的最近邻数量
    int nprobe            // 要检查的聚类中心数量
) {
    // 确保nprobe不超过可用的聚类中心数量
    nprobe = std::min(nprobe, g_ivf_index.nlist);
    
    // 找到最近的nprobe个聚类中心
    std::vector<std::pair<float, int>> cluster_dists;
    for (int i = 0; i < g_ivf_index.nlist; i++) {
        float dist = compute_l2_distance(query, g_ivf_index.centroids[i].data(), dim);
        cluster_dists.push_back(std::make_pair(dist, i));
    }
    
    // 按距离排序
    std::sort(cluster_dists.begin(), cluster_dists.end());
    
    // 获取系统核心数量，确定线程数
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    int num_threads = std::min(num_cores, nprobe);  // 线程数不超过nprobe
    
    // 为每个线程准备任务和结果
    std::vector<ThreadArg> thread_args(num_threads);
    std::vector<std::vector<std::pair<float, int>>> all_results(nprobe);
    
    // 分配任务给线程
    for (int i = 0; i < nprobe; i++) {
        int cluster_idx = cluster_dists[i].second;
        int thread_idx = i % num_threads;
        
        // 创建任务
        SearchTask task;
        task.cluster_idx = cluster_idx;
        task.query = query;
        task.base = base;
        task.dim = dim;
        task.local_results = &all_results[i];
        
        // 分配给对应线程
        thread_args[thread_idx].tasks.push_back(task);
        thread_args[thread_idx].thread_id = thread_idx;
    }
    
    // 创建线程
    std::vector<pthread_t> threads(num_threads);
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], nullptr, search_thread, &thread_args[i]);
    }
    
    // 等待所有线程完成
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }
    
    // 合并结果
    return merge_results(all_results, k);
}

#endif // IVF_PTHREAD_DYNAMIC_H