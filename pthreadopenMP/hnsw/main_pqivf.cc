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
#include "pqivf.h"

// 全局PQIVF索引
PQIVFIndex g_pqivf_index;

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

// 构建PQIVF索引
void build_pqivf_index(float* base, size_t base_number, size_t vecdim, int nlist, int m = PQIVF_M, int k = PQIVF_K) {
    std::cout << "构建PQIVF索引，聚类数: " << nlist << ", 子空间数: " << m << ", 每子空间聚类数: " << k << std::endl;
    
    // 1. 初始化PQIVF索引参数
    g_pqivf_index.nlist = nlist;
    g_pqivf_index.dim = vecdim;
    g_pqivf_index.M = m;
    g_pqivf_index.K = k;
    g_pqivf_index.sub_dim = vecdim / m;
    
    // 2. 先训练PQ码本
    std::cout << "训练PQ码本..." << std::endl;
    
    // 初始化码本
    g_pqivf_index.codebooks.resize(m);
    for (int subq = 0; subq < m; ++subq) {
        g_pqivf_index.codebooks[subq].resize(k);
        for (int centroid = 0; centroid < k; ++centroid) {
            g_pqivf_index.codebooks[subq][centroid].resize(g_pqivf_index.sub_dim);
        }
    }
    
    // 设置收敛阈值
    const float convergence_threshold = 0.0001f;
    
    // 训练每个子空间的码本
    for (int subq = 0; subq < m; ++subq) {
        std::cout << "训练子空间 " << subq + 1 << "/" << m << std::endl;
        
        // 初始化第一个中心为随机选择的向量
        std::vector<size_t> random_indices;
        for (size_t i = 0; i < k; ++i) {
            random_indices.push_back(rand() % base_number);
        }
        
        for (int centroid = 0; centroid < k; ++centroid) {
            size_t random_idx = random_indices[centroid];
            for (int d = 0; d < g_pqivf_index.sub_dim; ++d) {
                g_pqivf_index.codebooks[subq][centroid][d] = base[random_idx * vecdim + subq * g_pqivf_index.sub_dim + d];
            }
        }
        
        // K-means聚类
        const int max_iter = 300;
        std::vector<std::vector<size_t>> clusters(k);
        
        // 保存上一次迭代的聚类中心
        std::vector<std::vector<float>> prev_centroids(k, std::vector<float>(g_pqivf_index.sub_dim, 0.0f));
        
        for (int iter = 0; iter < max_iter; ++iter) {
            // 保存当前聚类中心用于检查收敛性
            for (int centroid = 0; centroid < k; ++centroid) {
                prev_centroids[centroid] = g_pqivf_index.codebooks[subq][centroid];
            }
            
            // 清空聚类
            for (int centroid = 0; centroid < k; ++centroid) {
                clusters[centroid].clear();
            }
            
            // 分配数据点到最近的中心
            #pragma omp parallel for
            for (size_t i = 0; i < base_number; ++i) {
                float min_dist = INFINITY;
                int best_centroid = 0;
                
                for (int centroid = 0; centroid < k; ++centroid) {
                    float dist = 0;
                    for (int d = 0; d < g_pqivf_index.sub_dim; ++d) {
                        float diff = base[i * vecdim + subq * g_pqivf_index.sub_dim + d] - 
                                    g_pqivf_index.codebooks[subq][centroid][d];
                        dist += diff * diff;
                    }
                    
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_centroid = centroid;
                    }
                }
                
                #pragma omp critical
                {
                    clusters[best_centroid].push_back(i);
                }
            }
            
            // 更新中心
            bool any_empty = false;
            for (int centroid = 0; centroid < k; ++centroid) {
                if (clusters[centroid].empty()) {
                    any_empty = true;
                    continue;
                }
                
                // 重置中心
                std::fill(g_pqivf_index.codebooks[subq][centroid].begin(), 
                        g_pqivf_index.codebooks[subq][centroid].end(), 0.0f);
                
                // 计算新的中心
                for (size_t idx : clusters[centroid]) {
                    for (int d = 0; d < g_pqivf_index.sub_dim; ++d) {
                        g_pqivf_index.codebooks[subq][centroid][d] += 
                            base[idx * vecdim + subq * g_pqivf_index.sub_dim + d];
                    }
                }
                
                // 归一化
                for (int d = 0; d < g_pqivf_index.sub_dim; ++d) {
                    g_pqivf_index.codebooks[subq][centroid][d] /= clusters[centroid].size();
                }
            }
            
            // 如果有空聚类，重新初始化
            if (any_empty) {
                for (int centroid = 0; centroid < k; ++centroid) {
                    if (clusters[centroid].empty()) {
                        size_t random_idx = rand() % base_number;
                        for (int d = 0; d < g_pqivf_index.sub_dim; ++d) {
                            g_pqivf_index.codebooks[subq][centroid][d] = 
                                base[random_idx * vecdim + subq * g_pqivf_index.sub_dim + d];
                        }
                    }
                }
            }
            
            // 检查收敛性
            bool converged = true;
            for (int centroid = 0; centroid < k; ++centroid) {
                if (clusters[centroid].empty()) continue;
                
                float diff_sum = 0.0f;
                for (int d = 0; d < g_pqivf_index.sub_dim; ++d) {
                    float diff = g_pqivf_index.codebooks[subq][centroid][d] - prev_centroids[centroid][d];
                    diff_sum += diff * diff;
                }
                
                if (diff_sum > convergence_threshold) {
                    converged = false;
                    break;
                }
            }
            
            if (converged) {
                std::cout << "子空间 " << subq + 1 << " 的K-means在第 " << iter + 1 << " 次迭代后收敛" << std::endl;
                break;
            }
        }
    }
    
    // 3. 对所有数据进行PQ编码
    std::cout << "对所有数据进行PQ编码..." << std::endl;
    g_pqivf_index.codes.resize(base_number);
    
    #pragma omp parallel for
    for (size_t i = 0; i < base_number; ++i) {
        g_pqivf_index.codes[i].resize(m);
        
        for (int subq = 0; subq < m; ++subq) {
            float min_dist = INFINITY;
            int best_centroid = 0;
            
            for (int centroid = 0; centroid < k; ++centroid) {
                float dist = 0;
                for (int d = 0; d < g_pqivf_index.sub_dim; ++d) {
                    float diff = base[i * vecdim + subq * g_pqivf_index.sub_dim + d] - 
                                g_pqivf_index.codebooks[subq][centroid][d];
                    dist += diff * diff;
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_centroid = centroid;
                }
            }
            
            g_pqivf_index.codes[i][subq] = best_centroid;
        }
    }
    
    // 4. 在PQ编码空间上进行IVF聚类
    std::cout << "在PQ编码空间上进行IVF聚类..." << std::endl;
    g_pqivf_index.centroids.resize(nlist, std::vector<uint8_t>(m)); // <--- 修改类型
    g_pqivf_index.invlists.resize(nlist);
    
    // 4.1 随机选择初始聚类中心
    std::vector<std::vector<uint8_t>> initial_centroids;
    for (int i = 0; i < nlist; ++i) {
        size_t random_idx = rand() % base_number;
        initial_centroids.push_back(g_pqivf_index.codes[random_idx]);
    }
    
    // 4.2 K-means聚类
    const int max_iter = 300;
    std::vector<std::vector<size_t>> clusters(nlist);
    
    // 保存上一次迭代的聚类中心用于检查收敛性
    std::vector<std::vector<uint8_t>> prev_centroids = initial_centroids;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        std::cout << "IVF聚类迭代: " << iter + 1 << "/" << max_iter << std::endl;
        
        // 保存当前聚类中心
        prev_centroids = initial_centroids;
        
        // 清空聚类
        for (int i = 0; i < nlist; ++i) {
            clusters[i].clear();
        }
        
        // 分配数据点到最近的聚类中心
        #pragma omp parallel for
        for (size_t i = 0; i < base_number; ++i) {
            float min_dist = INFINITY;
            int best_cluster = 0;
            
            for (int j = 0; j < nlist; ++j) {
                float dist = 0;
                for (int m = 0; m < g_pqivf_index.M; ++m) {
                    // 使用PQ子空间欧氏距离
                    uint8_t code1 = g_pqivf_index.codes[i][m];
                    uint8_t code2 = initial_centroids[j][m];
                    for (int d = 0; d < g_pqivf_index.sub_dim; ++d) {
                        float diff = g_pqivf_index.codebooks[m][code1][d] - 
                                    g_pqivf_index.codebooks[m][code2][d];
                        dist += diff * diff;
                    }
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
        bool any_empty = false;
        for (int i = 0; i < nlist; ++i) {
            if (clusters[i].empty()) {
                any_empty = true;
                continue;
            }
            
            // 计算每个子空间中出现频率最高的码
            std::vector<std::vector<int>> code_freq(m, std::vector<int>(k, 0));
            
            for (size_t idx : clusters[i]) {
                for (int j = 0; j < m; ++j) {
                    code_freq[j][g_pqivf_index.codes[idx][j]]++;
                }
            }
            
            // 更新中心为众数
            for (int j = 0; j < g_pqivf_index.M; ++j) {
                int max_freq = -1;
                int max_code = 0;
                
                for (int c = 0; c < g_pqivf_index.K; ++c) {
                    if (code_freq[j][c] > max_freq) {
                        max_freq = code_freq[j][c];
                        max_code = c;
                    }
                }
                
                initial_centroids[i][j] = max_code;
            }
        }
        
        // 如果有空聚类，重新初始化
        if (any_empty) {
            for (int i = 0; i < nlist; ++i) {
                if (clusters[i].empty()) {
                    size_t random_idx = rand() % base_number;
                    initial_centroids[i] = g_pqivf_index.codes[random_idx];
                }
            }
        }
        
        // 检查收敛性
        bool converged = true;
        for (int i = 0; i < nlist; ++i) {
            if (clusters[i].empty()) continue;
            
            int diff_count = 0;
            for (int j = 0; j < m; ++j) {
                if (initial_centroids[i][j] != prev_centroids[i][j]) {
                    diff_count++;
                }
            }
            
            // 如果中心点有变化（差异比例超过阈值），则未收敛
            float diff_ratio = static_cast<float>(diff_count) / m;
            if (diff_ratio > 0.01f) {  // 允许1%的变化
                converged = false;
                break;
            }
        }
        
        if (converged) {
            std::cout << "IVF聚类在第 " << iter + 1 << " 次迭代后收敛" << std::endl;
            break;
        }
    }
    
    // 5. 设置最终的聚类中心和倒排表
    for (int i = 0; i < nlist; ++i) {
        // 保存聚类中心
        for (int j = 0; j < m; ++j) {
            g_pqivf_index.centroids[i][j] = initial_centroids[i][j];
        }
        
        // 保存倒排表
        g_pqivf_index.invlists[i] = std::vector<int>(clusters[i].begin(), clusters[i].end());
    }
    
    // 保存索引到文件
    std::string index_path = "files/pqivf" + std::to_string(nlist) + "_" + 
                            std::to_string(m) + "x" + std::to_string(k) + ".index";
    g_pqivf_index.save(index_path);
    std::cout << "PQIVF索引已保存到 " << index_path << std::endl;
}

int main(int argc, char *argv[])
{
    // 为rand()函数设置种子，以便每次运行K-means时获得不同的随机初始化
    // 如果希望结果可复现（用于调试），可以注释掉此行或使用固定种子
    srand(time(NULL)); 

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
    const int nprobe = 24;        // 查询时检查的聚类数
    const int rerank_k = 200;     // 重排的向量数，0表示不重排

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 构建或加载PQIVF索引
    std::string index_path = "files/pqivf" + std::to_string(nlist) + "_" + 
                            std::to_string(PQIVF_M) + "x" + std::to_string(PQIVF_K) + ".index";
    
    if (!g_pqivf_index.load(index_path)) {
        std::cout << "PQIVF索引文件不存在，开始构建..." << std::endl;
        build_pqivf_index(base, base_number, vecdim, nlist);
    } else {
        std::cout << "已从文件加载PQIVF索引" << std::endl;
    }
    
    // 开始查询测试
    std::cout << "开始PQIVF搜索测试 (k=" << k << ", nprobe=" << nprobe << ", rerank_k=" << rerank_k << ")..." << std::endl;
    
    for (int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 使用PQIVF搜索
        auto res = pqivf_search(base, test_query + i*vecdim, base_number, vecdim, k, nprobe, rerank_k);

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

    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";

    // 释放内存
    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    return 0;
}
