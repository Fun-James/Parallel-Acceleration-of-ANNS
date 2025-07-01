/*********************************************************************
*  IVF CPU串行实现 - 用于与GPU版本性能对比
*  编译命令：g++ ivf_cpu.cpp -O2 -o ivf_cpu -std=c++14
*  
*  实现特点：
*  1. 纯串行处理，无并行优化
*  2. 标准欧几里得距离计算
*  3. 单线程K-means聚类
*  4. 内存友好的数据布局
*  5. 与GPU版本相同的算法逻辑
*********************************************************************/
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <memory>
#include <cmath>
#include <cstring>      // for memcpy

/* ================================================================
   数据结构定义
   ============================================================== */
struct IVFIndex {
    int num_clusters;
    int dimension;
    int num_vectors;
    float* centroids;           // [num_clusters, dimension]
    float* cluster_data;        // 所有簇的数据连续存储
    int* cluster_offsets;       // 每个簇在cluster_data中的起始位置
    int* cluster_sizes;         // 每个簇的向量数量
    int* vector_ids;           // 原始向量ID映射
};

struct QueryGroup {
    std::vector<int> query_indices;     // 查询索引
    std::vector<int> probe_clusters;    // 需要探索的簇
    int max_cluster_size;               // 最大簇大小
};

/* ================================================================
   数据读取函数
   ============================================================== */
template<typename T>
T* LoadData(const std::string& data_path, size_t& n, size_t& d) {
    std::ifstream fin(data_path, std::ios::binary);
    if (!fin) {
        std::cerr << "Cannot open " << data_path << "\n";
        exit(-1);
    }
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    T* data = new T[n * d];
    fin.read((char*)data, n * d * sizeof(T));
    fin.close();
    
    std::cerr << "Loaded: " << data_path << " (" << n << " x " << d << ")\n";
    return data;
}

/* ================================================================
   标准欧几里得距离计算
   ============================================================== */
inline float ComputeL2Distance(const float* a, const float* b, int dim) {
    float result = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

/* ================================================================
   K-means聚类实现
   ============================================================== */
void SimpleKMeans(float* data, int n, int d, int k, float* centroids, int* assignments) {
    // 随机初始化质心
    srand(42);
    std::cout << "  Initializing " << k << " centroids randomly..." << std::endl;
    for (int i = 0; i < k; i++) {
        int rand_idx = rand() % n;
        std::memcpy(&centroids[i * d], &data[rand_idx * d], d * sizeof(float));
    }
    
    // 迭代优化
    std::cout << "  Starting K-means iterations (serial)..." << std::endl;
    
    const float convergence_threshold = 0.0001f;
    const int max_iterations = 300;
    
    // 保存上一次的质心用于收敛检测
    std::vector<float> prev_centroids(k * d);
    std::copy(centroids, centroids + k * d, prev_centroids.begin());
    
    for (int iter = 0; iter < max_iterations; iter++) {
        std::cout << "    Iteration " << (iter + 1) << "/" << max_iterations << std::endl;
        
        // 分配点到最近的质心 - 串行处理
        for (int i = 0; i < n; i++) {
            float min_dist = 1e10f;
            int best_cluster = 0;
            
            for (int c = 0; c < k; c++) {
                float dist = ComputeL2Distance(&data[i * d], &centroids[c * d], d);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            assignments[i] = best_cluster;
        }
        
        // 更新质心 - 串行处理
        std::vector<int> cluster_counts(k, 0);
        std::fill(centroids, centroids + k * d, 0.0f);
        
        // 累加每个簇的向量
        for (int i = 0; i < n; i++) {
            int c = assignments[i];
            cluster_counts[c]++;
            for (int j = 0; j < d; j++) {
                centroids[c * d + j] += data[i * d + j];
            }
        }
        
        // 计算最终质心
        for (int c = 0; c < k; c++) {
            if (cluster_counts[c] > 0) {
                for (int j = 0; j < d; j++) {
                    centroids[c * d + j] /= cluster_counts[c];
                }
            }
        }
        
        // 计算质心变化量（收敛检测）
        float total_change = 0.0f;
        for (int i = 0; i < k * d; i++) {
            float diff = centroids[i] - prev_centroids[i];
            total_change += diff * diff;
        }
        float avg_change = sqrt(total_change / (k * d));
        
        // 每5次迭代或最后几次迭代输出统计信息
        if ((iter + 1) % 5 == 0 || avg_change < convergence_threshold) {
            int empty_clusters = 0;
            for (int c = 0; c < k; c++) {
                if (cluster_counts[c] == 0) empty_clusters++;
            }
            std::cout << "      - Empty clusters: " << empty_clusters << std::endl;
            std::cout << "      - Average centroid change: " << avg_change << std::endl;
        }
        
        // 检查收敛
        if (avg_change < convergence_threshold) {
            std::cout << "    Converged after " << (iter + 1) << " iterations (change < " 
                      << convergence_threshold << ")" << std::endl;
            break;
        }
        
        // 保存当前质心作为下一次的上一次质心
        std::copy(centroids, centroids + k * d, prev_centroids.begin());
    }
    std::cout << "  K-means iterations completed." << std::endl;
}

/* ================================================================
   保存和加载IVF索引
   ============================================================== */
void SaveIVFIndex(IVFIndex* index, const std::string& index_path) {
    std::ofstream fout(index_path, std::ios::binary);
    if (!fout) {
        std::cerr << "Cannot create index file: " << index_path << std::endl;
        return;
    }
    
    // 保存基本信息
    fout.write((char*)&index->num_clusters, sizeof(int));
    fout.write((char*)&index->dimension, sizeof(int));
    fout.write((char*)&index->num_vectors, sizeof(int));
    
    // 保存质心
    fout.write((char*)index->centroids, index->num_clusters * index->dimension * sizeof(float));
    
    // 保存簇信息
    fout.write((char*)index->cluster_offsets, (index->num_clusters + 1) * sizeof(int));
    fout.write((char*)index->cluster_sizes, index->num_clusters * sizeof(int));
    
    // 保存向量数据和ID映射
    fout.write((char*)index->cluster_data, index->num_vectors * index->dimension * sizeof(float));
    fout.write((char*)index->vector_ids, index->num_vectors * sizeof(int));
    
    fout.close();
    std::cout << "Index saved to: " << index_path << std::endl;
}

IVFIndex* LoadIVFIndex(const std::string& index_path) {
    std::ifstream fin(index_path, std::ios::binary);
    if (!fin) {
        return nullptr;
    }
    
    auto index = new IVFIndex();
    
    // 读取基本信息
    fin.read((char*)&index->num_clusters, sizeof(int));
    fin.read((char*)&index->dimension, sizeof(int));
    fin.read((char*)&index->num_vectors, sizeof(int));
    
    // 分配内存
    index->centroids = new float[index->num_clusters * index->dimension];
    index->cluster_offsets = new int[index->num_clusters + 1];
    index->cluster_sizes = new int[index->num_clusters];
    index->vector_ids = new int[index->num_vectors];
    index->cluster_data = new float[index->num_vectors * index->dimension];
    
    // 读取质心
    fin.read((char*)index->centroids, index->num_clusters * index->dimension * sizeof(float));
    
    // 读取簇信息
    fin.read((char*)index->cluster_offsets, (index->num_clusters + 1) * sizeof(int));
    fin.read((char*)index->cluster_sizes, index->num_clusters * sizeof(int));
    
    // 读取向量数据和ID映射
    fin.read((char*)index->cluster_data, index->num_vectors * index->dimension * sizeof(float));
    fin.read((char*)index->vector_ids, index->num_vectors * sizeof(int));
    
    fin.close();
    std::cout << "Index loaded from: " << index_path << std::endl;
    return index;
}

/* ================================================================
   构建IVF索引
   ============================================================== */
IVFIndex* BuildIVFIndex(float* data, int n, int d, int num_clusters) {
    auto index = new IVFIndex();
    index->num_clusters = num_clusters;
    index->dimension = d;
    index->num_vectors = n;
    
    // 分配内存
    index->centroids = new float[num_clusters * d];
    index->cluster_offsets = new int[num_clusters + 1];
    index->cluster_sizes = new int[num_clusters];
    index->vector_ids = new int[n];
    
    std::cout << "Starting K-means clustering with " << num_clusters << " clusters..." << std::endl;
    
    // 执行K-means聚类
    std::vector<int> assignments(n);
    SimpleKMeans(data, n, d, num_clusters, index->centroids, assignments.data());
    
    std::cout << "K-means clustering completed. Organizing data into clusters..." << std::endl;
    
    // 统计每个簇的大小
    std::fill(index->cluster_sizes, index->cluster_sizes + num_clusters, 0);
    for (int i = 0; i < n; i++) {
        index->cluster_sizes[assignments[i]]++;
    }
    
    // 输出簇大小统计
    int min_size = index->cluster_sizes[0], max_size = index->cluster_sizes[0];
    int empty_clusters = 0;
    for (int i = 0; i < num_clusters; i++) {
        if (index->cluster_sizes[i] == 0) empty_clusters++;
        min_size = std::min(min_size, index->cluster_sizes[i]);
        max_size = std::max(max_size, index->cluster_sizes[i]);
    }
    
    std::cout << "Cluster statistics:" << std::endl;
    std::cout << "  - Min cluster size: " << min_size << std::endl;
    std::cout << "  - Max cluster size: " << max_size << std::endl;
    std::cout << "  - Average cluster size: " << (float)n / num_clusters << std::endl;
    std::cout << "  - Empty clusters: " << empty_clusters << std::endl;
    
    // 计算偏移量
    index->cluster_offsets[0] = 0;
    for (int i = 0; i < num_clusters; i++) {
        index->cluster_offsets[i + 1] = index->cluster_offsets[i] + index->cluster_sizes[i];
    }
    
    // 分配聚类数据内存
    index->cluster_data = new float[n * d];
    
    std::cout << "Reorganizing vectors into clusters (serial)..." << std::endl;
    
    // 重新组织数据到簇中 - 串行处理
    std::vector<int> cluster_counters(num_clusters, 0);
    
    // 首先统计每个簇应该放置的向量索引
    std::vector<std::vector<int>> cluster_indices(num_clusters);
    for (int c = 0; c < num_clusters; c++) {
        cluster_indices[c].reserve(index->cluster_sizes[c]);
    }
    
    for (int i = 0; i < n; i++) {
        int cluster_id = assignments[i];
        cluster_indices[cluster_id].push_back(i);
    }
    
    // 串行处理每个簇的数据复制
    for (int c = 0; c < num_clusters; c++) {
        int offset = index->cluster_offsets[c];
        for (size_t i = 0; i < cluster_indices[c].size(); i++) {
            int original_idx = cluster_indices[c][i];
            int pos = offset + i;
            
            // 复制向量数据
            std::memcpy(&index->cluster_data[pos * d], &data[original_idx * d], d * sizeof(float));
            index->vector_ids[pos] = original_idx;
        }
        
        // 输出进度
        if (c % (num_clusters / 10) == 0) {
            std::cout << "  Progress: " << (c * 100 / num_clusters) << "%" << std::endl;
        }
    }
    
    std::cout << "IVF index construction completed!" << std::endl;
    return index;
}

/* ================================================================
   分组策略：基于簇重合度的查询分组
   ============================================================== */
std::vector<QueryGroup> GroupQueriesByClusterOverlap(int* query_clusters, 
                                                     int num_queries, int nprobe,
                                                     IVFIndex* index) {
    std::vector<QueryGroup> groups;
    std::vector<bool> processed(num_queries, false);
    
    for (int i = 0; i < num_queries; i++) {
        if (processed[i]) continue;
        
        QueryGroup group;
        group.query_indices.push_back(i);
        processed[i] = true;
        
        // 获取当前查询的簇列表
        std::set<int> base_clusters;
        for (int j = 0; j < nprobe; j++) {
            base_clusters.insert(query_clusters[i * nprobe + j]);
        }
        
        // 寻找具有高重合度的其他查询
        for (int j = i + 1; j < num_queries; j++) {
            if (processed[j] || group.query_indices.size() >= 32) break; // 限制组大小
            
            // 计算重合度
            std::set<int> other_clusters;
            for (int k = 0; k < nprobe; k++) {
                other_clusters.insert(query_clusters[j * nprobe + k]);
            }
            
            // 计算交集
            std::vector<int> intersection;
            std::set_intersection(base_clusters.begin(), base_clusters.end(),
                                other_clusters.begin(), other_clusters.end(),
                                std::back_inserter(intersection));
            
            // 如果重合度超过阈值，加入组中
            if (intersection.size() >= nprobe * 0.5) { // 50%重合度阈值
                group.query_indices.push_back(j);
                processed[j] = true;
                
                // 更新组的簇列表（取并集）
                base_clusters.insert(other_clusters.begin(), other_clusters.end());
            }
        }
        
        // 确定最终的探索簇列表和最大簇大小
        group.probe_clusters.assign(base_clusters.begin(), base_clusters.end());
        group.max_cluster_size = 0;
        for (int cluster_id : group.probe_clusters) {
            group.max_cluster_size = std::max(group.max_cluster_size, 
                                            index->cluster_sizes[cluster_id]);
        }
        
        groups.push_back(group);
    }
    
    std::cout << "Created " << groups.size() << " query groups with average size: "
              << (float)num_queries / groups.size() << std::endl;
    
    return groups;
}

/* ================================================================
   CPU串行IVF搜索
   ============================================================== */
void SearchIVFParallel(IVFIndex* index, float* queries, int num_queries,
                      int nprobe, int k, std::vector<std::vector<int>>& results) {
    
    // 第一阶段：计算查询到质心的距离并选择top-nprobe个簇
    std::vector<int> query_clusters(num_queries * nprobe);
    
    std::cout << "Phase 1: Computing query-to-centroid distances (serial)..." << std::endl;
    
    for (int q = 0; q < num_queries; q++) {
        std::vector<std::pair<float, int>> centroid_distances;
        centroid_distances.reserve(index->num_clusters);
        
        // 计算到所有质心的距离
        for (int c = 0; c < index->num_clusters; c++) {
            float dist = ComputeL2Distance(&queries[q * index->dimension], 
                                         &index->centroids[c * index->dimension], 
                                         index->dimension);
            centroid_distances.push_back({dist, c});
        }
        
        // 选择top-nprobe个最近的簇
        std::nth_element(centroid_distances.begin(), 
                        centroid_distances.begin() + nprobe, 
                        centroid_distances.end());
        
        for (int i = 0; i < nprobe; i++) {
            query_clusters[q * nprobe + i] = centroid_distances[i].second;
        }
    }
    
    std::cout << "Phase 2: Grouping queries by cluster overlap..." << std::endl;
    
    // 第二阶段：基于簇重合度进行查询分组
    auto query_groups = GroupQueriesByClusterOverlap(query_clusters.data(), num_queries, nprobe, index);
    
    // 初始化结果存储
    results.resize(num_queries);
    for (auto& result : results) {
        result.resize(k);
    }
    
    std::cout << "Phase 3: Serial search within groups..." << std::endl;
    
    // 第三阶段：对每个组进行串行搜索
    for (size_t g = 0; g < query_groups.size(); g++) {
        const auto& group = query_groups[g];
        
        // 为组内每个查询分别收集候选者
        std::vector<std::vector<std::pair<float, int>>> query_candidates(group.query_indices.size());
        
        // 对组内每个簇进行搜索
        for (int cluster_id : group.probe_clusters) {
            int cluster_size = index->cluster_sizes[cluster_id];
            if (cluster_size == 0) continue;
            
            // 为每个查询分别计算距离并收集候选者
            for (size_t q = 0; q < group.query_indices.size(); q++) {
                int query_idx = group.query_indices[q];
                
                // 计算查询向量到簇内所有向量的欧几里得距离
                for (int v = 0; v < cluster_size; v++) {
                    int vector_offset = (index->cluster_offsets[cluster_id] + v) * index->dimension;
                    int query_offset = query_idx * index->dimension;
                    
                    // 使用标准距离计算
                    float dist = ComputeL2Distance(&queries[query_offset], 
                                                  &index->cluster_data[vector_offset], 
                                                  index->dimension);
                    
                    int original_vector_id = index->vector_ids[index->cluster_offsets[cluster_id] + v];
                    query_candidates[q].push_back({dist, original_vector_id});
                }
            }
        }
        
        // 为组内每个查询选择top-k结果
        for (size_t q = 0; q < group.query_indices.size(); q++) {
            int query_idx = group.query_indices[q];
            auto& candidates = query_candidates[q];
            
            // 排序并选择top-k
            if (candidates.size() > k) {
                std::nth_element(candidates.begin(), candidates.begin() + k, candidates.end());
                candidates.resize(k);
            }
            std::sort(candidates.begin(), candidates.end());
            
            // 存储结果
            for (size_t i = 0; i < candidates.size(); i++) {
                results[query_idx][i] = candidates[i].second;
            }
            
            // 如果候选者不足k个，填充-1
            for (size_t i = candidates.size(); i < k; i++) {
                results[query_idx][i] = -1;
            }
        }
    }
}

/* ================================================================
   主函数
   ============================================================== */
int main(int argc, char *argv[]) {
    std::cout << "Running CPU serial version (no parallelization)" << std::endl;
    
    // 数据加载
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;
    
    std::string data_path = "./anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    // 限制测试数量
    test_number = std::min((size_t)1000, test_number);
    
    // IVF参数
    const int num_clusters = 1024;
    const int nprobe = 4;
    const int k = 10;
    
    // 构建索引文件路径 - 与GPU版本共用同一个索引文件
    std::string index_filename = "ivf_index_" + std::to_string(num_clusters) + "_" + 
                                std::to_string(base_number) + "_" + std::to_string(vecdim) + ".bin";
    std::string index_path = "./files/" + index_filename;
    
    // 检查files目录是否存在，不存在则创建
    std::string files_dir = "./files/";
    if (system(("mkdir -p " + files_dir).c_str()) != 0) {
        std::cerr << "Warning: Could not create files directory" << std::endl;
    }
    
    IVFIndex* index = nullptr;
    
    // 尝试加载现有索引 - 可能是GPU版本创建的
    std::cout << "Checking for existing index (shared with GPU): " << index_path << std::endl;
    index = LoadIVFIndex(index_path);
    
    if (index != nullptr) {
        // 验证索引参数是否匹配
        if (index->num_clusters == num_clusters && 
            index->dimension == vecdim && 
            index->num_vectors == base_number) {
            std::cout << "Successfully loaded existing index (shared with GPU)!" << std::endl;
        } else {
            std::cout << "Index parameters mismatch, rebuilding..." << std::endl;
            delete[] index->centroids;
            delete[] index->cluster_data;
            delete[] index->cluster_offsets;
            delete[] index->cluster_sizes;
            delete[] index->vector_ids;
            delete index;
            index = nullptr;
        }
    }
    
    if (index == nullptr) {
        std::cout << "Building new CPU IVF index with " << num_clusters << " clusters..." << std::endl;
        auto start_build = std::chrono::high_resolution_clock::now();
        
        // 构建IVF索引
        index = BuildIVFIndex(base, base_number, vecdim, num_clusters);
        
        auto end_build = std::chrono::high_resolution_clock::now();
        auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build);
        std::cout << "CPU Index built in " << build_time.count() << " ms" << std::endl;
        
        // 保存索引到文件
        std::cout << "Saving CPU index to file..." << std::endl;
        SaveIVFIndex(index, index_path);
    }
    
    // 执行搜索
    std::cout << "Searching with nprobe=" << nprobe << ", k=" << k << "...\n";
    auto start_search = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<int>> search_results;
    SearchIVFParallel(index, test_query, test_number, nprobe, k, search_results);
    
    auto end_search = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search);
    
    // 计算recall
    float total_recall = 0.0f;
    for (size_t i = 0; i < test_number; i++) {
        std::set<int> gt_set;
        for (int j = 0; j < k; j++) {
            gt_set.insert(test_gt[i * test_gt_d + j]);
        }
        
        int hits = 0;
        for (int j = 0; j < k && j < (int)search_results[i].size(); j++) {
            if (gt_set.count(search_results[i][j]) > 0) {
                hits++;
            }
        }
        
        total_recall += (float)hits / k;
    }
    
    float avg_recall = total_recall / test_number;
    
    // 输出结果
    std::cout << "\n=== IVF CPU Serial Search Results ===\n";
    std::cout << "Number of clusters: " << num_clusters << "\n";
    std::cout << "nprobe: " << nprobe << "\n";
    std::cout << "Average recall@" << k << ": " << avg_recall << "\n";
    std::cout << "Search time: " << search_time.count() << " us\n";
    std::cout << "Average latency: " << search_time.count() / (float)test_number << " us/query\n";
    std::cout << "QPS: " << test_number * 1000000.0 / search_time.count() << " queries/sec\n";
    std::cout << "Processing mode: Serial (no parallelization)\n";
    
    // 清理内存
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    delete[] index->centroids;
    delete[] index->cluster_data;
    delete[] index->cluster_offsets;
    delete[] index->cluster_sizes;
    delete[] index->vector_ids;
    delete index;
    
    return 0;
}
