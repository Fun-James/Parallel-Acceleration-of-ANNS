#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <windows.h> // 替换Linux特定头文件为Windows头文件
#include <omp.h>

// 明确启用Windows线程模型
#define _GLIBCXX_HAS_GTHREADS 1
#define _GLIBCXX_USE_C11_MUTEX 1
#include "ivf_pthread_dynamic.h"
IVFIndex g_ivf_index;          // 这里进行实例化

/********************************************************************
切换头文件就可以运行不同策略的pq查询。第一次调用会先构建预处理保存至files中，之后调用可以直接读取预处理文件。
如果需要运行NNS的flat或simd优化代码，需要切换头文件，并且更换查询函数，最好把加载索引的函数也注释掉。
 *******************************************************************/
template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(size_t i = 0; i < n; ++i){  // 从int改为size_t
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

// 构建IVF索引的函数
void build_ivf_index(float* base, size_t base_number, size_t vecdim, int nlist) {
    std::cout << "Building IVF index with " << nlist << " clusters..." << std::endl;
    
    // 初始化IVF索引
    g_ivf_index.nlist = nlist;
    g_ivf_index.dim = vecdim;
    g_ivf_index.centroids.resize(nlist);
    g_ivf_index.invlists.resize(nlist);
    
    // 随机选择初始聚类中心
    std::vector<int> centroid_indices;
    for (int i = 0; i < nlist; i++) {
        int random_idx = rand() % base_number;
        // 避免重复选择相同的中心点
        while (std::find(centroid_indices.begin(), centroid_indices.end(), random_idx) != centroid_indices.end()) {
            random_idx = rand() % base_number;
        }
        centroid_indices.push_back(random_idx);
        
        // 初始化聚类中心
        g_ivf_index.centroids[i].resize(vecdim);
        for (size_t d = 0; d < vecdim; d++) {  // 从int改为size_t
            g_ivf_index.centroids[i][d] = base[random_idx * vecdim + d];
        }
    }
    
    // K-means聚类迭代
    const int max_iter = 300;  // 对于IVF，通常不需要太多迭代
    std::vector<std::vector<int>> clusters(nlist);
    std::vector<std::vector<float>> prev_centroids(nlist);
    
    // 初始化上一轮的聚类中心
    for (int i = 0; i < nlist; i++) {
        prev_centroids[i].resize(vecdim, 0.0f);
    }
    
    const float convergence_threshold = 0.0001f; // 收敛阈值
    
    for (int iter = 0; iter < max_iter; iter++) {
        std::cout << "IVF K-means iteration " << iter+1 << "/" << max_iter << std::endl;
        
        // 保存当前的聚类中心以便检查收敛
        for (int i = 0; i < nlist; i++) {
            std::copy(g_ivf_index.centroids[i].begin(), 
                     g_ivf_index.centroids[i].end(), 
                     prev_centroids[i].begin());
        }
        
        // 清空聚类
        for (int i = 0; i < nlist; i++) {
            clusters[i].clear();
        }
        
        // 分配数据点到最近的聚类中心
        #pragma omp parallel for
        for (size_t i = 0; i < base_number; i++) {  // 从int改为size_t
            float min_dist = INFINITY;
            int best_cluster = 0;
            
            for (int j = 0; j < nlist; j++) {
                float dist = 0;
                for (size_t d = 0; d < vecdim; d++) {  // 从int改为size_t
                    float diff = base[i * vecdim + d] - g_ivf_index.centroids[j][d];
                    dist += diff * diff;
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            
            #pragma omp critical
            {
                clusters[best_cluster].push_back(i);
            }
        }
        
        // 更新聚类中心
        for (int i = 0; i < nlist; i++) {
            if (clusters[i].empty()) {
                continue;  // 跳过空聚类
            }
            
            // 重置聚类中心
            std::fill(g_ivf_index.centroids[i].begin(), g_ivf_index.centroids[i].end(), 0.0f);
            
            // 计算新的聚类中心
            for (int idx : clusters[i]) {
                for (size_t d = 0; d < vecdim; d++) {  // 从int改为size_t
                    g_ivf_index.centroids[i][d] += base[idx * vecdim + d];
                }
            }
            
            // 归一化
            for (size_t d = 0; d < vecdim; d++) {  // 从int改为size_t
                g_ivf_index.centroids[i][d] /= clusters[i].size();
            }
        }
        
        // 检查收敛性
        bool converged = true;
        for (int i = 0; i < nlist; i++) {
            float diff_sum = 0.0f;
            for (size_t d = 0; d < vecdim; d++) {  // 从int改为size_t
                float diff = g_ivf_index.centroids[i][d] - prev_centroids[i][d];
                diff_sum += diff * diff;
            }
            if (diff_sum > convergence_threshold) {
                converged = false;
                break;
            }
        }
        
        if (converged) {
            std::cout << "IVF K-means converged after " << iter+1 << " iterations." << std::endl;
            break;
        }
    }
    
    // 构建倒排列表
    std::cout << "Building inverted lists..." << std::endl;
    for (int i = 0; i < nlist; i++) {
        g_ivf_index.invlists[i] = clusters[i];
    }
    
    // 保存索引到文件
    std::string index_path = "files/ivf1024.index";
    g_ivf_index.save(index_path);
    std::cout << "IVF index saved to " << index_path << std::endl;
}

// 获取高精度时间戳(微秒)
int64_t get_microseconds() {
    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    
    return (start.QuadPart * 1000000) / frequency.QuadPart;
}

int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "anndata\\"; // 修改为Windows路径格式
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;
    const int nlist = 1024;  // IVF的聚类中心数量
    const int nprobe = 16;  // IVF搜索时要检查的聚类中心数量

    std::vector<SearchResult> results;
    results.resize(test_number);


    // 构建或加载IVF索引
    if (!g_ivf_index.load("files\\ivf1024.index")) { // 修改为Windows路径格式
        std::cout << "Building IVF index..." << std::endl;
        build_ivf_index(base, base_number, vecdim, nlist);
        std::cout << "IVF index built and saved." << std::endl;
    } else {
        std::cout << "Loaded IVF index from file." << std::endl;
    }
    
    // 查询测试代码
    std::cout << "Starting IVF search (k=" << k << ", nprobe=" << nprobe << ")..." << std::endl;
    
    try {
        for(size_t i = 0; i < test_number; ++i) {  // 从int改为size_t
            int64_t start_time = get_microseconds();

            // 使用IVF搜索
            auto res = ivf_search(base, test_query + i*vecdim, base_number, vecdim, k, nprobe);

            int64_t end_time = get_microseconds();
            int64_t diff = end_time - start_time;

            std::set<uint32_t> gtset;
            for(size_t j = 0; j < k; ++j){  // 从int改为size_t
                int t = test_gt[j + i*test_gt_d];
                gtset.insert(t);
            }

            size_t acc = 0;
            while (res.size()) {   
                int x = res.top().second;
                if(gtset.find(x) != gtset.end()){
                    ++acc;
                }
                res.pop();
            }
            float recall = (float)acc/k;

            results[i] = {recall, diff};
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
    }

    float avg_recall = 0, avg_latency = 0;
    for(size_t i = 0; i < test_number; ++i) {  // 从int改为size_t
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";

    // 释放内存
    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    // 确保在程序结束前清理线程资源
    //destroy_static_threads();

    return 0;
}
