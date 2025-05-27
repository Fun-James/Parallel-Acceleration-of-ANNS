#pragma once

#include <vector>
#include <algorithm>
#include <random>
#include <omp.h>
#include <chrono>
#include <queue>
#include <set>
#include <limits>
#include <cmath>
#include <immintrin.h>  // x86 SSE/AVX 头文件
#include "hnswlib/hnswlib/hnswlib.h"

// 优化的欧氏距离计算函数 - 一次处理8个浮点数 (使用SSE)
inline float sse_l2sqr(const float* x, const float* y, size_t d) {
    __m128 sum_vec1 = _mm_setzero_ps();
    __m128 sum_vec2 = _mm_setzero_ps();
    
    // 每次处理8个元素 (使用两个寄存器)
    size_t i = 0;
    for (; i + 7 < d; i += 8) {
        // 第一个寄存器处理4个元素
        __m128 x_vec1 = _mm_loadu_ps(x + i);
        __m128 y_vec1 = _mm_loadu_ps(y + i);
        __m128 diff1 = _mm_sub_ps(x_vec1, y_vec1);
        
        // 第二个寄存器处理另外4个元素
        __m128 x_vec2 = _mm_loadu_ps(x + i + 4);
        __m128 y_vec2 = _mm_loadu_ps(y + i + 4);
        __m128 diff2 = _mm_sub_ps(x_vec2, y_vec2);
        
        // 分别累加平方差
        sum_vec1 = _mm_add_ps(sum_vec1, _mm_mul_ps(diff1, diff1));
        sum_vec2 = _mm_add_ps(sum_vec2, _mm_mul_ps(diff2, diff2));
    }
    
    // 处理剩余的4个元素
    for (; i + 3 < d; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128 y_vec = _mm_loadu_ps(y + i);
        __m128 diff = _mm_sub_ps(x_vec, y_vec);
        sum_vec1 = _mm_add_ps(sum_vec1, _mm_mul_ps(diff, diff));
    }
    
    // 合并两个寄存器的结果
    sum_vec1 = _mm_add_ps(sum_vec1, sum_vec2);
    
    // 水平相加得到最终结果
    float sum[4];
    _mm_storeu_ps(sum, sum_vec1);
    float result = sum[0] + sum[1] + sum[2] + sum[3];
    
    // 处理剩余元素
    for (; i < d; i++) {
        float diff = x[i] - y[i];
        result += diff * diff;
    }
    
    return result;
}

// 通过内联函数启用或禁用SIMD
inline float compute_distance(const float* x, const float* y, size_t d, bool use_simd) {
    if (use_simd) {
        return sse_l2sqr(x, y, d);
    } else {
        float dist = 0.0f;
        for (size_t k = 0; k < d; k++) {
            float diff = x[k] - y[k];
            dist += diff * diff;
        }
        return dist;
    }
}

class IVF_HNSW {
public:
    // 构造函数
    IVF_HNSW(size_t numClusters, size_t dim, size_t M = 11, size_t efConstruction = 150, 
             float convergence_threshold = 0.0001, bool use_simd = true) 
        : numClusters(numClusters), dim(dim), M(M), efConstruction(efConstruction), 
          convergence_threshold(convergence_threshold), use_simd(use_simd) {
        centroids.resize(numClusters, std::vector<float>(dim, 0));
        indices.resize(numClusters, nullptr);
        is_index_built = false;
    }
    
    // 析构函数
    ~IVF_HNSW() {
        for (auto index : indices) {
            if (index) delete index;
        }
    }
    
    // 使用K-means构建索引
    void build(float* data, size_t numPoints, size_t dim, int max_iterations = 200) {
        if (this->dim != dim) {
            throw std::runtime_error("Dimension mismatch");
        }
        
        std::cout << "Building IVF+HNSW index with " << numClusters << " clusters" << std::endl;
        std::cout << "Using convergence threshold: " << convergence_threshold << std::endl;
        if (use_simd) {
            std::cout << "SIMD optimization (x86 SSE, 8-float processing) enabled" << std::endl;
        }
        
        // 初始化聚类中心 (使用随机选择的数据点)
        std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
        std::vector<int> sampling_indices(numPoints);
        for (size_t i = 0; i < numPoints; i++) sampling_indices[i] = i;
        std::shuffle(sampling_indices.begin(), sampling_indices.end(), rng);
        
        for (size_t i = 0; i < numClusters && i < numPoints; i++) {
            for (size_t j = 0; j < dim; j++) {
                centroids[i][j] = data[sampling_indices[i] * dim + j];
            }
        }
        
        // 执行K-means聚类 (添加收敛判断)
        bool converged = false;
        int iterations = 0;
        
        while (!converged && iterations < max_iterations) {
            iterations++;
            std::cout << "K-means iteration " << iterations << "/" << max_iterations << std::endl;
            
            // 保存旧的聚类中心，用于计算变化量
            std::vector<std::vector<float>> old_centroids = centroids;
            
            // 分配点到最近的聚类中心
            std::vector<std::vector<size_t>> clusters(numClusters);
            
            #pragma omp parallel for
            for (size_t i = 0; i < numPoints; i++) {
                float min_dist = std::numeric_limits<float>::max();
                size_t closest_centroid = 0;
                
                for (size_t j = 0; j < numClusters; j++) {
                    float dist = compute_distance(&data[i * dim], centroids[j].data(), dim, use_simd);
                    
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest_centroid = j;
                    }
                }
                
                #pragma omp critical
                {
                    clusters[closest_centroid].push_back(i);
                }
            }
            
            // 更新聚类中心
            for (size_t i = 0; i < numClusters; i++) {
                if (clusters[i].empty()) continue;
                
                std::vector<float> new_centroid(dim, 0.0f);
                for (size_t idx : clusters[i]) {
                    for (size_t k = 0; k < dim; k++) {
                        new_centroid[k] += data[idx * dim + k];
                    }
                }
                
                for (size_t k = 0; k < dim; k++) {
                    centroids[i][k] = new_centroid[k] / clusters[i].size();
                }
            }
            
            // 计算聚类中心的变化量 (使用欧氏距离平均值)
            float total_change = 0.0f;
            for (size_t i = 0; i < numClusters; i++) {
                float centroid_change = 0.0f;
                for (size_t j = 0; j < dim; j++) {
                    float diff = centroids[i][j] - old_centroids[i][j];
                    centroid_change += diff * diff;
                }
                centroid_change = std::sqrt(centroid_change);
                total_change += centroid_change;
            }
            float avg_change = total_change / numClusters;
            std::cout << "Average centroid change: " << avg_change << std::endl;
            
            // 检查是否收敛
            if (avg_change < convergence_threshold) {
                std::cout << "K-means converged after " << iterations << " iterations" << std::endl;
                converged = true;
            }
        }
        
        // 最终分配点到聚类
        cluster_assignments.resize(numPoints);
        std::vector<std::vector<size_t>> clusters(numClusters);
        
        #pragma omp parallel for
        for (size_t i = 0; i < numPoints; i++) {
            float min_dist = std::numeric_limits<float>::max();
            size_t closest_centroid = 0;
            
            for (size_t j = 0; j < numClusters; j++) {
                float dist = compute_distance(&data[i * dim], centroids[j].data(), dim, use_simd);
                
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = j;
                }
            }
            
            cluster_assignments[i] = closest_centroid;
            
            #pragma omp critical
            {
                clusters[closest_centroid].push_back(i);
            }
        }
        
        // 为每个聚类构建HNSW索引
        #pragma omp parallel for
        for (size_t i = 0; i < numClusters; i++) {
            std::cout << "Building HNSW index for cluster " << i << " with " << clusters[i].size() << " points" << std::endl;
            
            if (clusters[i].empty()) {
                continue;  // 对于空聚类，indices[i]已经是nullptr
            }
            
            // 创建L2空间
            hnswlib::L2Space space(dim);
            
            // 创建HNSW索引
            indices[i] = new hnswlib::HierarchicalNSW<float>(&space, clusters[i].size() + 1, M, efConstruction);
            
            // 添加点到索引
            for (size_t j = 0; j < clusters[i].size(); j++) {
                size_t idx = clusters[i][j];
                indices[i]->addPoint(&data[idx * dim], idx);  // 使用原始ID作为标签
            }
        }
        
        is_index_built = true;
    }
    
    // 加载已构建的索引
    void load(const std::string& filename, hnswlib::SpaceInterface<float>* space) {
        std::ifstream input(filename, std::ios::binary);
        if (!input.is_open()) {
            throw std::runtime_error("Cannot open file");
        }
        
        // 读取基本参数
        size_t saved_clusters, saved_dim;
        input.read(reinterpret_cast<char*>(&saved_clusters), sizeof(saved_clusters));
        input.read(reinterpret_cast<char*>(&saved_dim), sizeof(saved_dim));
        
        if (saved_clusters != numClusters || saved_dim != dim) {
            throw std::runtime_error("Index parameters mismatch");
        }
        
        // 读取聚类中心
        for (size_t i = 0; i < numClusters; i++) {
            input.read(reinterpret_cast<char*>(centroids[i].data()), dim * sizeof(float));
        }
        
        // 为每个聚类加载HNSW索引
        for (size_t i = 0; i < numClusters; i++) {
            size_t cluster_size;
            input.read(reinterpret_cast<char*>(&cluster_size), sizeof(cluster_size));
            
            if (cluster_size == 0) {
                indices[i] = nullptr;
                continue;
            }
            
            std::string index_filename = filename + "_cluster_" + std::to_string(i);
            indices[i] = new hnswlib::HierarchicalNSW<float>(space, index_filename, false);
        }
        
        // 读取点到聚类的分配
        size_t num_points;
        input.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
        cluster_assignments.resize(num_points);
        input.read(reinterpret_cast<char*>(cluster_assignments.data()), num_points * sizeof(size_t));
        
        input.close();
        is_index_built = true;
    }
    
    // 保存索引
    void save(const std::string& filename) {
        if (!is_index_built) {
            throw std::runtime_error("Index not built yet");
        }
        
        std::ofstream output(filename, std::ios::binary);
        
        // 写入基本参数
        output.write(reinterpret_cast<const char*>(&numClusters), sizeof(numClusters));
        output.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        
        // 写入聚类中心
        for (size_t i = 0; i < numClusters; i++) {
            output.write(reinterpret_cast<const char*>(centroids[i].data()), dim * sizeof(float));
        }
        
        // 为每个聚类保存HNSW索引
        for (size_t i = 0; i < numClusters; i++) {
            size_t cluster_size = indices[i] ? indices[i]->getCurrentElementCount() : 0;
            output.write(reinterpret_cast<const char*>(&cluster_size), sizeof(cluster_size));
            
            if (indices[i]) {
                std::string index_filename = filename + "_cluster_" + std::to_string(i);
                indices[i]->saveIndex(index_filename);
            }
        }
        
        // 写入点到聚类的分配
        size_t num_points = cluster_assignments.size();
        output.write(reinterpret_cast<const char*>(&num_points), sizeof(num_points));
        output.write(reinterpret_cast<const char*>(cluster_assignments.data()), num_points * sizeof(size_t));
        
        output.close();
    }
    
    // KNN查询
    std::vector<std::pair<float, hnswlib::labeltype>> query(const float* query_point, size_t k, size_t efSearch = 50, size_t nprobe = 1) {
        if (!is_index_built) {
            throw std::runtime_error("Index not built yet");
        }
        
        // 找到最近的nprobe个聚类中心
        std::priority_queue<std::pair<float, size_t>> nearest_clusters;
        for (size_t i = 0; i < numClusters; i++) {
            float dist = compute_distance(query_point, centroids[i].data(), dim, use_simd);
            nearest_clusters.emplace(dist, i);
            
            if (nearest_clusters.size() > nprobe) {
                nearest_clusters.pop();
            }
        }
        
        // 在最近的nprobe个聚类中并行搜索
        std::vector<size_t> search_clusters;
        while (!nearest_clusters.empty()) {
            search_clusters.push_back(nearest_clusters.top().second);
            nearest_clusters.pop();
        }
        
        std::vector<std::vector<std::pair<float, hnswlib::labeltype>>> cluster_results(search_clusters.size());
        
        #pragma omp parallel for
        for (size_t i = 0; i < search_clusters.size(); i++) {
            size_t cluster_id = search_clusters[i];
            if (!indices[cluster_id] || indices[cluster_id]->getCurrentElementCount() == 0) {
                continue;
            }
            
            // 设置搜索参数
            indices[cluster_id]->setEf(efSearch);
            
            // 执行搜索
            auto result = indices[cluster_id]->searchKnn(query_point, k);
            
            std::vector<std::pair<float, hnswlib::labeltype>> cluster_result;
            while (!result.empty()) {
                cluster_result.push_back(std::make_pair(result.top().first, result.top().second));
                result.pop();
            }
            
            cluster_results[i] = std::move(cluster_result);
        }
        
        // 合并各聚类的搜索结果
        std::vector<std::pair<float, hnswlib::labeltype>> final_results;
        for (const auto& result : cluster_results) {
            final_results.insert(final_results.end(), result.begin(), result.end());
        }
        
        // 排序并只保留前k个
        std::sort(final_results.begin(), final_results.end());
        if (final_results.size() > k) {
            final_results.resize(k);
        }
        
        return final_results;
    }
    
private:
    // 聚类中心
    std::vector<std::vector<float>> centroids;
    
    // 每个聚类的HNSW索引 - 确保这是指针类型
    std::vector<hnswlib::HierarchicalNSW<float>*> indices;
    
    // 每个点属于哪个聚类
    std::vector<size_t> cluster_assignments;
    
    size_t numClusters;  // 聚类数量
    size_t dim;          // 向量维度
    size_t M;            // HNSW参数
    size_t efConstruction; // HNSW参数
    float convergence_threshold; // 聚类收敛阈值
    bool is_index_built;  // 索引是否已构建
    bool use_simd;  // 是否使用SIMD加速
};
