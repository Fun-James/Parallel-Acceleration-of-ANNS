#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
// Windows替代sys/time.h
#include <windows.h>
#include <omp.h>
#include <cassert>
#include <direct.h> // 用于创建目录

#include "ivf_pthread_static_cache.h"
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
    for(size_t i = 0; i < n; ++i){  // 改为size_t类型
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
void build_ivf_index(float* base,
                     size_t base_number,
                     size_t vecdim,
                     int    nlist)
{
    std::cout << "Building IVF index  (N=" << base_number
              << ", dim=" << vecdim << ", nlist=" << nlist << ") ...\n";

    /* -------------------------- 1. 初始化 -------------------------- */
    g_ivf_index.nlist = nlist;
    g_ivf_index.dim   = (int)vecdim;
    g_ivf_index.centroids.assign(nlist, std::vector<float>(vecdim));
    g_ivf_index.invlists.assign (nlist, {});

    /* ------------------ 2. 选取初始中心 (随机) --------------------- */
    std::vector<int> init_ids;
    init_ids.reserve(nlist);
    while ((int)init_ids.size() < nlist) {
        int id = rand() % base_number;
        if (std::find(init_ids.begin(), init_ids.end(), id) == init_ids.end())
            init_ids.push_back(id);
    }
    for (int c = 0; c < nlist; ++c) {
        std::memcpy(g_ivf_index.centroids[c].data(),
                    base + (size_t)init_ids[c] * vecdim,
                    vecdim * sizeof(float));
    }

    /* ----------------------- 3. K-means --------------------------- */
    const int   max_iter = 200;
    const float eps      = 1e-4f;

    std::vector<std::vector<int>>   clusters(nlist);
    std::vector<std::vector<float>> prev_centroids(nlist,
                                    std::vector<float>(vecdim, 0.f));

    for (int it = 0; it < max_iter; ++it) {
        std::cout << "  k-means iter " << it+1 << "/" << max_iter << std::endl;

        for (auto& v : clusters) v.clear();          // 清空簇

        /* ----------- 3.1 assignment (OMP 并行) ----------- */
#pragma omp parallel for
        for (long i = 0; i < (long)base_number; ++i) {
            float best_d = std::numeric_limits<float>::infinity();
            int   best_c = 0;
            for (int c = 0; c < nlist; ++c) {
                float d = 0.f;
                const float* cen = g_ivf_index.centroids[c].data();
                const float* vec = base + (size_t)i * vecdim;
                for (size_t k = 0; k < vecdim; ++k) {
                    float diff = vec[k] - cen[k];
                    d += diff * diff;
                }
                if (d < best_d) { best_d = d; best_c = c; }
            }
#pragma omp critical
            { clusters[best_c].push_back((int)i); }
        }

        /* ----------- 3.2 update ----------- */
        prev_centroids = g_ivf_index.centroids;      // 备份旧中心

#pragma omp parallel for
        for (int c = 0; c < nlist; ++c) {
            if (clusters[c].empty()) continue;
            std::fill(g_ivf_index.centroids[c].begin(),
                      g_ivf_index.centroids[c].end(), 0.f);

            for (int idx : clusters[c]) {
                const float* vec = base + (size_t)idx * vecdim;
                for (size_t k = 0; k < vecdim; ++k)
                    g_ivf_index.centroids[c][k] += vec[k];
            }
            float inv_sz = 1.f / clusters[c].size();
            for (size_t k = 0; k < vecdim; ++k)
                g_ivf_index.centroids[c][k] *= inv_sz;
        }

        /* ----------- 3.3 收敛判断 ----------- */
        bool converged = true;
#pragma omp parallel for reduction(&:converged)
        for (int c = 0; c < nlist; ++c) {
            float sum = 0.f;
            for (size_t k = 0; k < vecdim; ++k) {
                float diff = g_ivf_index.centroids[c][k] - prev_centroids[c][k];
                sum += diff * diff;
            }
            if (sum > eps) converged = false;
        }
        if (converged) {
            std::cout << "  k-means converged at iter "
                      << it+1 << std::endl;
            break;
        }
    }

    /* -------------------- 4. 连续内存重排 -------------------- */
    std::cout << "  Reordering vectors by cluster ..." << std::endl;

    std::vector<float> new_base(base_number * vecdim); // 连续存储
    std::vector<int>   new2old(base_number);           // new_id → old_id

    size_t offset = 0;
    for (int c = 0; c < nlist; ++c) {
        g_ivf_index.invlists[c].clear();
        g_ivf_index.invlists[c].reserve(clusters[c].size());

        for (int old_id : clusters[c]) {
            std::memcpy(new_base.data() + offset * vecdim,
                        base           + (size_t)old_id * vecdim,
                        vecdim * sizeof(float));

            g_ivf_index.invlists[c].push_back((int)offset); // list 存 new_id
            new2old[offset] = old_id;
            ++offset;
        }
    }
    assert(offset == base_number && "vector count mismatch");

    /* 将连续内存和映射表 move 给索引结构 */
    g_ivf_index.rearranged_base.swap(new_base);
    g_ivf_index.id_map.swap(new2old);      // new_id → old_id

    /* ----------------------- 5. 保存 ----------------------- */
    const std::string fn =
        "files/ivf" + std::to_string(nlist) + "cache.index";
    g_ivf_index.save(fn);
    std::cout << "  IVF index saved to " << fn << std::endl << std::flush;
}

// 添加gettimeofday实现
int gettimeofday(struct timeval* tp, void* tzp) {
    // 注意：tzp参数被忽略
    LARGE_INTEGER frequency;
    LARGE_INTEGER current;
    
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&current);
    
    long long microseconds = current.QuadPart * 1000000ll / frequency.QuadPart;
    tp->tv_sec = (long)(microseconds / 1000000ll);
    tp->tv_usec = (long)(microseconds % 1000000ll);
    
    return 0;
}

int main(int argc, char *argv[])
{
    // 创建files目录（如果不存在）
    _mkdir("files");

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "anndata/"; // 调整为Windows相对路径
    
    try {
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
        if (!g_ivf_index.load("files/ivf1024cache.index")) {
            std::cout << "Building IVF index..." << std::endl;
            build_ivf_index(base, base_number, vecdim, nlist);
            std::cout << "IVF index built and saved." << std::endl;
        } else {
            std::cout << "Loaded IVF index from file." << std::endl;
        }
        
        // 查询测试代码
        std::cout << "Starting IVF search (k=" << k << ", nprobe=" << nprobe << ")..." << std::endl;
        
        for(size_t i = 0; i < test_number; ++i) {
            const unsigned long Converter = 1000 * 1000;
            struct timeval val;
            int ret = gettimeofday(&val, NULL);

            // 使用IVF搜索
            auto res = ivf_search(base, test_query + i*vecdim, base_number, vecdim, k, nprobe);

            struct timeval newVal;
            ret = gettimeofday(&newVal, NULL);
            int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

            std::set<uint32_t> gtset;
            for(size_t j = 0; j < k; ++j){  // 改为size_t类型
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

        float avg_recall = 0, avg_latency = 0;
        for(size_t i = 0; i < test_number; ++i) {
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
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        destroy_static_threads(); // 确保在异常情况下也清理资源
    } catch (...) {
        std::cerr << "Unknown exception" << std::endl;
        destroy_static_threads(); // 确保在异常情况下也清理资源
    }

    return 0;
}
