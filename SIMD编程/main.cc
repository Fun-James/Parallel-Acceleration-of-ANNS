#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
// 可以自行添加需要的头文件
#include "simd.h"
#include "pq.h"
using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
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

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}
void build_pq_index(float *base, size_t base_number, size_t vecdim)
{
    // 初始化PQ索引
    g_pq_index.dim = vecdim;
    g_pq_index.sub_dim = vecdim / PQ_M;
    g_pq_index.codebooks.resize(PQ_M);

    // 为每个子空间分配内存并初始化聚类中心
    for (int m = 0; m < PQ_M; m++)
    {
        g_pq_index.codebooks[m].resize(PQ_K);
        for (int k = 0; k < PQ_K; k++)
        {
            g_pq_index.codebooks[m][k].resize(g_pq_index.sub_dim);
        }
    }

    // 计算每个维度的均值作为初始聚类中心
    std::vector<std::vector<float>> sub_means(PQ_M);
    for (int m = 0; m < PQ_M; m++)
    {
        sub_means[m].resize(g_pq_index.sub_dim, 0.0f);

        // 计算子空间的均值
        for (size_t i = 0; i < base_number; i++)
        {
            for (int d = 0; d < g_pq_index.sub_dim; d++)
            {
                sub_means[m][d] += base[i * vecdim + m * g_pq_index.sub_dim + d];
            }
        }

        // 归一化均值
        for (int d = 0; d < g_pq_index.sub_dim; d++)
        {
            sub_means[m][d] /= base_number;
            // 使用均值初始化第一个聚类中心
            g_pq_index.codebooks[m][0][d] = sub_means[m][d];
        }

        // 随机初始化其他聚类中心
        for (int k = 1; k < PQ_K; k++)
        {
            size_t rand_idx = rand() % base_number;
            for (int d = 0; d < g_pq_index.sub_dim; d++)
            {
                g_pq_index.codebooks[m][k][d] = base[rand_idx * vecdim + m * g_pq_index.sub_dim + d];
            }
        }
    }

    // K-means聚类迭代
    const int max_iter = 300;
    std::vector<std::vector<std::vector<size_t>>> clusters(PQ_M);
    std::vector<std::vector<std::vector<float>>> prev_codebooks(PQ_M);

    // 为prev_codebooks分配内存
    for (int m = 0; m < PQ_M; m++)
    {
        prev_codebooks[m].resize(PQ_K);
        for (int k = 0; k < PQ_K; k++)
        {
            prev_codebooks[m][k].resize(g_pq_index.sub_dim, 0.0f);
        }
    }
    const float convergence_threshold = 0.0001f; // 收敛阈值

    for (int iter = 0; iter < max_iter; iter++)
    {
        // 保存当前的codebooks用于后面比较
        for (int m = 0; m < PQ_M; m++)
        {
            for (int k = 0; k < PQ_K; k++)
            {
                std::copy(g_pq_index.codebooks[m][k].begin(),
                          g_pq_index.codebooks[m][k].end(),
                          prev_codebooks[m][k].begin());
            }
        }

        // 清空clusters
        for (int m = 0; m < PQ_M; m++)
        {
            clusters[m].clear();
            clusters[m].resize(PQ_K);
        }

        // 分配数据点到最近的聚类中心
        for (size_t i = 0; i < base_number; i++)
        {
            for (int m = 0; m < PQ_M; m++)
            {
                float min_dist = INFINITY;
                int best_k = 0;

                // 找到最近的聚类中心
                for (int k = 0; k < PQ_K; k++)
                {
                    float dist = 0;
                    for (int d = 0; d < g_pq_index.sub_dim; d++)
                    {
                        float diff = base[i * vecdim + m * g_pq_index.sub_dim + d] - g_pq_index.codebooks[m][k][d];
                        dist += diff * diff;
                    }
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        best_k = k;
                    }
                }

                clusters[m][best_k].push_back(i);
            }
        }

        // 更新聚类中心
        for (int m = 0; m < PQ_M; m++)
        {
            for (int k = 0; k < PQ_K; k++)
            {
                if (clusters[m][k].empty())
                    continue;

                // 重置聚类中心
                std::fill(g_pq_index.codebooks[m][k].begin(), g_pq_index.codebooks[m][k].end(), 0.0f);

                // 计算新的聚类中心
                for (size_t idx : clusters[m][k])
                {
                    for (int d = 0; d < g_pq_index.sub_dim; d++)
                    {
                        g_pq_index.codebooks[m][k][d] += base[idx * vecdim + m * g_pq_index.sub_dim + d];
                    }
                }

                // 归一化
                for (int d = 0; d < g_pq_index.sub_dim; d++)
                {
                    g_pq_index.codebooks[m][k][d] /= clusters[m][k].size();
                }
            }
        }
        bool converged = true;
        for (int m = 0; m < PQ_M; m++) {
            for (int k = 0; k < PQ_K; k++) {
                float diff_sum = 0.0f;
                for (int d = 0; d < g_pq_index.sub_dim; d++) {
                    float diff = g_pq_index.codebooks[m][k][d] - prev_codebooks[m][k][d];
                    diff_sum += diff * diff;
                }
                if (diff_sum > convergence_threshold) {
                    converged = false;
                    break;
                }
            }
            if (!converged) break;
        }
        
        if (converged) {
            std::cout << "K-means converged after " << iter+1 << " iterations." << std::endl;
            break;
        }
    }

    // 对所有数据点进行编码
    g_pq_index.codes.resize(base_number);
    for (size_t i = 0; i < base_number; i++)
    {
        g_pq_index.codes[i].resize(PQ_M);
        for (int m = 0; m < PQ_M; m++)
        {
            float min_dist = INFINITY;
            int best_k = 0;

            for (int k = 0; k < PQ_K; k++)
            {
                float dist = 0;
                for (int d = 0; d < g_pq_index.sub_dim; d++)
                {
                    float diff = base[i * vecdim + m * g_pq_index.sub_dim + d] - g_pq_index.codebooks[m][k][d];
                    dist += diff * diff;
                }
                if (dist < min_dist)
                {
                    min_dist = dist;
                    best_k = k;
                }
            }
            g_pq_index.codes[i][m] = best_k;
        }
    }

    // 保存索引到文件
    std::ofstream fout("files/pq.index16256", std::ios::binary);
    fout.write(reinterpret_cast<const char *>(&PQ_M), sizeof(int));
    fout.write(reinterpret_cast<const char *>(&PQ_K), sizeof(int));
    fout.write(reinterpret_cast<const char *>(&vecdim), sizeof(int));

    // 保存码本
    for (int m = 0; m < PQ_M; m++)
    {
        for (int k = 0; k < PQ_K; k++)
        {
            fout.write(reinterpret_cast<const char *>(g_pq_index.codebooks[m][k].data()),
                       g_pq_index.sub_dim * sizeof(float));
        }
    }

    // 保存编码数据
    size_t n = g_pq_index.codes.size();
    fout.write(reinterpret_cast<const char *>(&n), sizeof(int));
    for (size_t i = 0; i < n; i++)
    {
        fout.write(reinterpret_cast<const char *>(g_pq_index.codes[i].data()), PQ_M * sizeof(uint8_t));
    }
    fout.close();
}


int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;
    const int rerank_k = 100; // 定义 rerank 候选数量，例如 k 的 10 倍

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    // build_index(base, base_number, vecdim);
     // 在查询之前构建或加载PQ索引
     if (!g_pq_index.load("files/pq.index16256")) {
        std::cout << "Building PQ index..." << std::endl;
        build_pq_index(base, base_number, vecdim);
        std::cout << "PQ index built and saved." << std::endl;
    } else {
        std::cout << "Loaded PQ index from file." << std::endl;
    }
    // 设置原始数据指针，这也可以在循环外完成一次
    // g_pq_index.set_base_data(base, base_number); // 在 pq_search 内部调用了

    
    // 查询测试代码
    std::cout << "Starting PQ search with rerank (k=" << k << ", rerank_k=" << rerank_k << ")..." << std::endl;
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
        auto res = pq_search(base, test_query + i*vecdim, base_number, vecdim, k, rerank_k);

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
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
    for(int i = 0; i < test_number; ++i) {
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

    return 0;
}
