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
#include "ivfpq_openmp_simd.h"

// 全局IVFPQ索引
IVFPQIndex g_ivfpq_index;

template<typename T>
T* LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"加载数据 "<<data_path<<"\n";
    std::cerr<<"维度: "<<d<<"  数量:"<<n<<"  单元素大小:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

// 构建IVFPQ索引
void build_ivfpq_index(float* base, size_t base_number, size_t vecdim, int nlist, int m = IVFPQ_M, int k = IVFPQ_K) {
    std::cout << "构建IVFPQ索引，聚类数: " << nlist << ", 子空间数: " << m << ", 每子空间聚类数: " << k << std::endl;
    
    // 1. 初始化IVFPQ索引参数
    g_ivfpq_index.nlist = nlist;
    g_ivfpq_index.dim = vecdim;
    g_ivfpq_index.M = m;
    g_ivfpq_index.K = k;
    g_ivfpq_index.sub_dim = vecdim / m;
    
    // 2. 第一步：IVF聚类 - 将数据分配到不同聚类中
    // 2.1 初始化聚类中心
    g_ivfpq_index.centroids.resize(nlist);
    for (int i = 0; i < nlist; ++i) {
        g_ivfpq_index.centroids[i].resize(vecdim);
        // 随机选择初始聚类中心
        size_t random_idx = rand() % base_number;
        for (size_t d = 0; d < vecdim; ++d) {
            g_ivfpq_index.centroids[i][d] = base[random_idx * vecdim + d];
        }
    }
    
    // 2.2 K-means聚类
    const int max_iter = 200;  // IVF聚类的最大迭代次数
    std::vector<std::vector<int>> clusters(nlist);
    std::vector<std::vector<float>> prev_centroids(nlist);
    
    for (int i = 0; i < nlist; ++i) {
        prev_centroids[i].resize(vecdim, 0.0f);
    }
    
    const float convergence_threshold = 0.001f;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        std::cout << "IVF聚类迭代: " << iter + 1 << "/" << max_iter << std::endl;
        
        // 保存当前聚类中心
        for (int i = 0; i < nlist; ++i) {
            std::copy(g_ivfpq_index.centroids[i].begin(), 
                     g_ivfpq_index.centroids[i].end(), 
                     prev_centroids[i].begin());
        }
        
        // 清空聚类
        for (int i = 0; i < nlist; ++i) {
            clusters[i].clear();
        }
        
        // 分配数据点到最近的聚类
        #pragma omp parallel for
        for (size_t i = 0; i < base_number; ++i) {
            float min_dist = INFINITY;
            int best_cluster = 0;
            
            for (int j = 0; j < nlist; ++j) {
                float dist = 0;
                for (size_t d = 0; d < vecdim; ++d) {
                    float diff = base[i * vecdim + d] - g_ivfpq_index.centroids[j][d];
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
        for (int i = 0; i < nlist; ++i) {
            if (clusters[i].empty()) {
                // 如果聚类为空，随机选择一个新中心点
                size_t random_idx = rand() % base_number;
                for (size_t d = 0; d < vecdim; ++d) {
                    g_ivfpq_index.centroids[i][d] = base[random_idx * vecdim + d];
                }
                continue;
            }
            
            // 重置聚类中心
            std::fill(g_ivfpq_index.centroids[i].begin(), g_ivfpq_index.centroids[i].end(), 0.0f);
            
            // 计算新的聚类中心
            for (int idx : clusters[i]) {
                for (size_t d = 0; d < vecdim; ++d) {
                    g_ivfpq_index.centroids[i][d] += base[idx * vecdim + d];
                }
            }
            
            // 归一化
            for (size_t d = 0; d < vecdim; ++d) {
                g_ivfpq_index.centroids[i][d] /= clusters[i].size();
            }
        }
        
        // 检查收敛性
        bool converged = true;
        for (int i = 0; i < nlist; ++i) {
            float diff_sum = 0.0f;
            for (size_t d = 0; d < vecdim; ++d) {
                float diff = g_ivfpq_index.centroids[i][d] - prev_centroids[i][d];
                diff_sum += diff * diff;
            }
            if (diff_sum > convergence_threshold) {
                converged = false;
                break;
            }
        }
        
        if (converged) {
            std::cout << "IVF聚类在 " << iter + 1 << " 次迭代后收敛" << std::endl;
            break;
        }
    }
    
    // 构建倒排表
    g_ivfpq_index.invlists = clusters;
    std::cout << "IVF聚类完成，共 " << nlist << " 个聚类" << std::endl;
    
    // 3. 在每个聚类内部进行PQ编码
    std::cout << "开始对每个聚类进行PQ编码..." << std::endl;
    g_ivfpq_index.codebooks.resize(nlist);
    g_ivfpq_index.codes.resize(nlist);
    
    // 为每个聚类训练PQ码本并编码
    #pragma omp parallel for schedule(dynamic)
    for (int list_id = 0; list_id < nlist; ++list_id) {
        const auto& invlist = g_ivfpq_index.invlists[list_id];
        if (invlist.empty()) {
            continue;  // 跳过空聚类
        }
        
        // 获取当前粗聚类的中心
        const std::vector<float>& current_coarse_centroid = g_ivfpq_index.centroids[list_id];

        // 初始化当前聚类的码本
        g_ivfpq_index.codebooks[list_id].resize(m);
        for (int subq = 0; subq < m; ++subq) {
            g_ivfpq_index.codebooks[list_id][subq].resize(k);
            for (int centroid = 0; centroid < k; ++centroid) {
                g_ivfpq_index.codebooks[list_id][subq][centroid].resize(g_ivfpq_index.sub_dim);
            }
        }
        
        // 为每个子空间计算初始中心
        for (int subq = 0; subq < m; ++subq) {
            // 计算子空间的均值作为第一个中心
            std::vector<float> subq_mean(g_ivfpq_index.sub_dim, 0.0f);
            for (int idx : invlist) {
                for (int d = 0; d < g_ivfpq_index.sub_dim; ++d) {
                    float base_val = base[idx * vecdim + subq * g_ivfpq_index.sub_dim + d];
                    float centroid_val = current_coarse_centroid[subq * g_ivfpq_index.sub_dim + d];
                    subq_mean[d] += (base_val - centroid_val);
                }
            }
            
            for (int d = 0; d < g_ivfpq_index.sub_dim; ++d) {
                subq_mean[d] /= invlist.size();
                g_ivfpq_index.codebooks[list_id][subq][0][d] = subq_mean[d];
            }
            
            // 随机初始化其他中心
            for (int centroid = 1; centroid < k; ++centroid) {
                int rand_idx_original = invlist[rand() % invlist.size()]; // Get original index from invlist
                for (int d = 0; d < g_ivfpq_index.sub_dim; ++d) {
                    float base_val = base[rand_idx_original * vecdim + subq * g_ivfpq_index.sub_dim + d];
                    float centroid_val = current_coarse_centroid[subq * g_ivfpq_index.sub_dim + d];
                    g_ivfpq_index.codebooks[list_id][subq][centroid][d] = (base_val - centroid_val);
                }
            }
        }
        
        // 对每个子空间进行K-means聚类
        for (int subq = 0; subq < m; ++subq) {
            const int pq_max_iter = 200;  // PQ聚类的最大迭代次数
            std::vector<std::vector<int>> subq_clusters(k);
            
            for (int iter = 0; iter < pq_max_iter; ++iter) {
                // 清空聚类
                for (int centroid = 0; centroid < k; ++centroid) {
                    subq_clusters[centroid].clear();
                }
                
                // 分配数据点到最近的中心
                for (int idx : invlist) { // idx is original base index
                    float min_dist = INFINITY;
                    int best_centroid = 0;
                    
                    for (int centroid = 0; centroid < k; ++centroid) {
                        float dist = 0;
                        for (int d = 0; d < g_ivfpq_index.sub_dim; ++d) {
                            float base_val = base[idx * vecdim + subq * g_ivfpq_index.sub_dim + d];
                            float coarse_centroid_val = current_coarse_centroid[subq * g_ivfpq_index.sub_dim + d];
                            float residual_val = base_val - coarse_centroid_val;
                            float diff = residual_val - g_ivfpq_index.codebooks[list_id][subq][centroid][d];
                            dist += diff * diff;
                        }
                        
                        if (dist < min_dist) {
                            min_dist = dist;
                            best_centroid = centroid;
                        }
                    }
                    
                    subq_clusters[best_centroid].push_back(idx); // Store original base index
                }
                
                // 更新中心
                bool any_empty = false;
                for (int centroid = 0; centroid < k; ++centroid) {
                    if (subq_clusters[centroid].empty()) {
                        any_empty = true;
                        continue;
                    }
                    
                    // 重置中心
                    std::fill(g_ivfpq_index.codebooks[list_id][subq][centroid].begin(), 
                             g_ivfpq_index.codebooks[list_id][subq][centroid].end(), 0.0f);
                    
                    // 计算新的中心
                    for (int idx : subq_clusters[centroid]) { // idx is original base index
                        for (int d = 0; d < g_ivfpq_index.sub_dim; ++d) {
                            float base_val = base[idx * vecdim + subq * g_ivfpq_index.sub_dim + d];
                            float coarse_centroid_val = current_coarse_centroid[subq * g_ivfpq_index.sub_dim + d];
                            float residual_val = base_val - coarse_centroid_val;
                            g_ivfpq_index.codebooks[list_id][subq][centroid][d] += residual_val;
                        }
                    }
                    
                    // 归一化
                    for (int d = 0; d < g_ivfpq_index.sub_dim; ++d) {
                        g_ivfpq_index.codebooks[list_id][subq][centroid][d] /= subq_clusters[centroid].size();
                    }
                }
                
                // 如果有空聚类，重新初始化
                if (any_empty) {
                    for (int centroid = 0; centroid < k; ++centroid) {
                        if (subq_clusters[centroid].empty()) {
                            int rand_idx_original = invlist[rand() % invlist.size()]; // Get original index
                            for (int d = 0; d < g_ivfpq_index.sub_dim; ++d) {
                                float base_val = base[rand_idx_original * vecdim + subq * g_ivfpq_index.sub_dim + d];
                                float coarse_centroid_val = current_coarse_centroid[subq * g_ivfpq_index.sub_dim + d];
                                g_ivfpq_index.codebooks[list_id][subq][centroid][d] = (base_val - coarse_centroid_val);
                            }
                        }
                    }
                }
            }
        }
        
        // 对当前聚类内的所有数据点进行PQ编码
        g_ivfpq_index.codes[list_id].resize(invlist.size());
        for (size_t i = 0; i < invlist.size(); ++i) {
            int idx = invlist[i]; // Original base index
            g_ivfpq_index.codes[list_id][i].resize(m);
            
            for (int subq = 0; subq < m; ++subq) {
                float min_dist = INFINITY;
                int best_centroid = 0;
                
                for (int centroid = 0; centroid < k; ++centroid) {
                    float dist = 0;
                    for (int d = 0; d < g_ivfpq_index.sub_dim; ++d) {
                        float base_val = base[idx * vecdim + subq * g_ivfpq_index.sub_dim + d];
                        float coarse_centroid_val = current_coarse_centroid[subq * g_ivfpq_index.sub_dim + d];
                        float residual_val = base_val - coarse_centroid_val;
                        float diff = residual_val - g_ivfpq_index.codebooks[list_id][subq][centroid][d];
                        dist += diff * diff;
                    }
                    
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_centroid = centroid;
                    }
                }
                
                g_ivfpq_index.codes[list_id][i][subq] = best_centroid;
            }
        }
        

    }
    
    // 保存索引到文件
    std::string index_path = "files/ivfpq" + std::to_string(nlist) + "_" + 
                            std::to_string(m) + "x" + std::to_string(k) + ".index";
    g_ivfpq_index.save(index_path);
    std::cout << "IVFPQ索引已保存到 " << index_path << std::endl;
}

int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;          // 返回的最近邻个数
    const int nlist = 1024;       // IVF聚类中心数
    const int nprobe = 16;        // 查询时检查的聚类数
    const int rerank_k = 110;     // 重排的向量数，0表示不重排

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 构建或加载IVFPQ索引
    std::string index_path = "files/ivfpq" + std::to_string(nlist) + "_" + 
                            std::to_string(IVFPQ_M) + "x" + std::to_string(IVFPQ_K) + ".index";
    
    if (!g_ivfpq_index.load(index_path)) {
        std::cout << "IVFPQ索引文件不存在，开始构建..." << std::endl;
        build_ivfpq_index(base, base_number, vecdim, nlist);
    } else {
        std::cout << "已从文件加载IVFPQ索引" << std::endl;
    }
    
    // 开始查询测试
    std::cout << "开始IVFPQ搜索测试 (k=" << k << ", nprobe=" << nprobe << ", rerank_k=" << rerank_k << ")..." << std::endl;
    
    for (int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 使用IVFPQ搜索
        auto res = ivfpq_search(base, test_query + i*vecdim, base_number, vecdim, k, nprobe, rerank_k);

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        // 计算召回率
        std::set<uint32_t> gtset;
        for (int j = 0; j < k; ++j) {
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if (gtset.find(x) != gtset.end()) {
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    // 统计结果
    float avg_recall = 0, avg_latency = 0;
    for (int i = 0; i < test_number; ++i) {
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
