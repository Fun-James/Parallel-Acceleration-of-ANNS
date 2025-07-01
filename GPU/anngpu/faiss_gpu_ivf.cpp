/*********************************************************************
*  Faiss-GPU IVF实现 - 高性能ANN GPU加速
*  编译命令：g++ faiss_gpu_ivf.cpp -O2 -o faiss_gpu_ivf -std=c++14 -I/usr/local/include -L/usr/local/lib -lfaiss -lcuda -lcublas -lopenblas -lgomp
*  
*  实现特点：
*  1. 使用Faiss-GPU的高度优化IVF实现
*  2. GPU加速的聚类和搜索
*  3. 自动批处理和内存管理
*  4. 简化的GPU专用接口
*********************************************************************/
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <chrono>
#include <algorithm>
#include <memory>
#include <string>

// Faiss头文件
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>
#include <faiss/index_io.h>

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
   Faiss GPU IVF实现类
   ============================================================== */
class FaissGpuIVF {
private:
    std::unique_ptr<faiss::gpu::StandardGpuResources> gpu_resources;
    std::unique_ptr<faiss::gpu::GpuIndexIVFFlat> gpu_index;
    std::unique_ptr<faiss::IndexIVFFlat> temp_cpu_index;  // 仅用于训练
    
    int dimension;
    int num_clusters;
    size_t num_vectors;
    bool is_trained;

public:
    FaissGpuIVF(int dim, int n_clusters) 
        : dimension(dim), num_clusters(n_clusters), num_vectors(0), is_trained(false) {
        
        // 初始化GPU资源
        gpu_resources = std::make_unique<faiss::gpu::StandardGpuResources>();
        
        // 创建CPU索引（仅用于训练，训练后会转移到GPU）
        auto quantizer = new faiss::IndexFlatL2(dimension);
        temp_cpu_index = std::make_unique<faiss::IndexIVFFlat>(quantizer, dimension, num_clusters);
        
        std::cout << "Faiss GPU IVF initialized with " << num_clusters 
                  << " clusters, dimension " << dimension << std::endl;
    }
    
    ~FaissGpuIVF() = default;
    
    // 训练并构建GPU索引
    void BuildIndex(float* training_data, size_t n) {
        std::cout << "Building Faiss GPU IVF index with " << n << " vectors..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        // 步骤1: 使用CPU进行训练（聚类）
        std::cout << "Step 1: Training clusters..." << std::endl;
        temp_cpu_index->train(n, training_data);
        is_trained = true;
        
        // 步骤2: 添加向量到CPU索引
        std::cout << "Step 2: Adding vectors to CPU index..." << std::endl;
        temp_cpu_index->add(n, training_data);
        num_vectors = n;
        
        // 步骤3: 转移到GPU
        std::cout << "Step 3: Copying index to GPU..." << std::endl;
        faiss::gpu::GpuClonerOptions options;
        options.useFloat16 = false;  // 使用FP32精度
        options.usePrecomputed = false;
        options.reserveVecs = num_vectors;
        options.storeTransposed = true;  // 优化GPU内存访问模式
        
        gpu_index.reset(dynamic_cast<faiss::gpu::GpuIndexIVFFlat*>(
            faiss::gpu::index_cpu_to_gpu(gpu_resources.get(), 0, temp_cpu_index.get(), &options)));
        
        // 释放临时CPU索引内存
        temp_cpu_index.reset();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "GPU index built successfully in " << duration.count() << " ms" << std::endl;
        
        // 输出聚类统计信息
        PrintClusterStats(training_data, n);
    }
    // 打印聚类统计信息
    void PrintClusterStats(float* training_data, size_t n) {
        if (!gpu_index) return;
        
        // 创建临时CPU索引来获取聚类信息
        auto temp_quantizer = new faiss::IndexFlatL2(dimension);
        auto temp_index = std::make_unique<faiss::IndexIVFFlat>(temp_quantizer, dimension, num_clusters);
        temp_index->train(n, training_data);
        
        std::vector<int> cluster_sizes(num_clusters, 0);
        std::vector<long> assign(n);
        temp_index->quantizer->assign(n, training_data, assign.data());
        
        for (size_t i = 0; i < n; i++) {
            cluster_sizes[assign[i]]++;
        }
        
        int min_size = *std::min_element(cluster_sizes.begin(), cluster_sizes.end());
        int max_size = *std::max_element(cluster_sizes.begin(), cluster_sizes.end());
        int empty_clusters = std::count(cluster_sizes.begin(), cluster_sizes.end(), 0);
        
        std::cout << "Cluster statistics:" << std::endl;
        std::cout << "  - Min cluster size: " << min_size << std::endl;
        std::cout << "  - Max cluster size: " << max_size << std::endl;
        std::cout << "  - Average cluster size: " << (float)n / num_clusters << std::endl;
        std::cout << "  - Empty clusters: " << empty_clusters << std::endl;
    }
    
    // GPU搜索
    void Search(float* queries, int nq, int nprobe, int k, 
                std::vector<std::vector<int>>& results) {
        if (!gpu_index) {
            std::cerr << "GPU index not available! Build index first." << std::endl;
            return;
        }
        
        std::cout << "Searching " << nq << " queries on GPU with nprobe=" << nprobe 
                  << ", k=" << k << "..." << std::endl;
        
        // 设置搜索参数
        gpu_index->nprobe = nprobe;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 分配结果存储
        std::vector<float> distances(nq * k);
        std::vector<faiss::idx_t> indices(nq * k);
        
        // 执行搜索
        gpu_index->search(nq, queries, k, distances.data(), indices.data());
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // 转换结果格式
        results.resize(nq);
        for (int i = 0; i < nq; i++) {
            results[i].resize(k);
            for (int j = 0; j < k; j++) {
                results[i][j] = static_cast<int>(indices[i * k + j]);
            }
        }
        
        std::cout << "GPU search completed in " << duration.count() << " microseconds" << std::endl;
        std::cout << "Average latency: " << duration.count() / (float)nq << " us/query" << std::endl;
        std::cout << "QPS: " << nq * 1000000.0 / duration.count() << " queries/sec" << std::endl;
    }
    
    // 保存索引（从GPU复制回CPU再保存）
    void SaveIndex(const std::string& filename) {
        if (!gpu_index) {
            std::cerr << "No GPU index to save!" << std::endl;
            return;
        }
        
        std::cout << "Saving GPU index to " << filename << "..." << std::endl;
        
        // 将GPU索引复制回CPU
        auto cpu_index = faiss::gpu::index_gpu_to_cpu(gpu_index.get());
        
        // 保存CPU索引
        faiss::write_index(cpu_index, filename.c_str());
        delete cpu_index;
        
        std::cout << "Index saved successfully." << std::endl;
    }
    
    // 加载索引并转移到GPU
    bool LoadIndex(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.good()) {
            return false;
        }
        
        std::cout << "Loading index from " << filename << "..." << std::endl;
        auto loaded_index = faiss::read_index(filename.c_str());
        
        // 检查索引类型
        auto ivf_index = dynamic_cast<faiss::IndexIVFFlat*>(loaded_index);
        if (!ivf_index) {
            std::cerr << "Loaded index is not an IVF index!" << std::endl;
            delete loaded_index;
            return false;
        }
        
        // 更新索引信息
        dimension = ivf_index->d;
        num_clusters = ivf_index->nlist;
        num_vectors = ivf_index->ntotal;
        is_trained = ivf_index->is_trained;
        
        // 转移到GPU
        faiss::gpu::GpuClonerOptions options;
        options.useFloat16 = false;
        options.usePrecomputed = false;
        options.reserveVecs = num_vectors;
        options.storeTransposed = true;
        
        gpu_index.reset(dynamic_cast<faiss::gpu::GpuIndexIVFFlat*>(
            faiss::gpu::index_cpu_to_gpu(gpu_resources.get(), 0, ivf_index, &options)));
        
        delete loaded_index;
        
        std::cout << "Index loaded and copied to GPU: " << num_vectors << " vectors, " 
                  << num_clusters << " clusters" << std::endl;
        return true;
    }
    
    // 获取索引信息
    void PrintIndexInfo() {
        std::cout << "\n=== Faiss GPU Index Information ===" << std::endl;
        std::cout << "Dimension: " << dimension << std::endl;
        std::cout << "Number of clusters: " << num_clusters << std::endl;
        std::cout << "Number of vectors: " << num_vectors << std::endl;
        std::cout << "Is trained: " << (is_trained ? "Yes" : "No") << std::endl;
        std::cout << "GPU index available: " << (gpu_index ? "Yes" : "No") << std::endl;
        std::cout << "====================================" << std::endl;
    }
};

/* ================================================================
   性能测试函数
   ============================================================== */
void RunPerformanceTest(FaissGpuIVF& faiss_ivf, float* queries, int* ground_truth,
                       int num_queries, int k, int gt_d, 
                       const std::vector<int>& nprobe_values) {
    
    std::cout << "\n=== GPU Performance Test ===" << std::endl;
    
    for (int nprobe : nprobe_values) {
        std::cout << "\n--- Testing with nprobe = " << nprobe << " ---" << std::endl;
        
        // GPU搜索
        std::vector<std::vector<int>> results;
        faiss_ivf.Search(queries, num_queries, nprobe, k, results);
        
        // 计算recall
        float recall = 0.0f;
        for (int i = 0; i < num_queries; i++) {
            std::set<int> gt_set;
            for (int j = 0; j < k; j++) {
                gt_set.insert(ground_truth[i * gt_d + j]);
            }
            
            int hits = 0;
            for (int j = 0; j < k; j++) {
                if (gt_set.count(results[i][j]) > 0) {
                    hits++;
                }
            }
            recall += (float)hits / k;
        }
        recall /= num_queries;
        
        std::cout << "Results for nprobe = " << nprobe << ":" << std::endl;
        std::cout << "  Recall@" << k << ": " << recall << std::endl;
    }
}

/* ================================================================
   主函数
   ============================================================== */
int main(int argc, char *argv[]) {
    std::cout << "=== Faiss-GPU IVF Performance Test ===" << std::endl;
    
    // 数据加载
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;
    
    std::string data_path = "./anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    // 限制测试数量
    test_number = std::min((size_t)2000, test_number);
    
    // 实验参数
    const int num_clusters = 1024;
    const int k = 10;
    const std::vector<int> nprobe_values = {1, 4, 8, 16, 32, 64};
    
    // 创建Faiss GPU IVF索引
    FaissGpuIVF faiss_ivf(vecdim, num_clusters);
    
    // 索引文件路径
    std::string index_filename = "faiss_gpu_ivf_" + std::to_string(num_clusters) + "_" + 
                                std::to_string(base_number) + "_" + std::to_string(vecdim) + ".faiss";
    std::string index_path = "./files/" + index_filename;
    
    // 创建files目录
    system("mkdir -p ./files/");
    
    // 尝试加载现有索引或构建新索引
    if (!faiss_ivf.LoadIndex(index_path)) {
        std::cout << "Building new Faiss GPU IVF index..." << std::endl;
        faiss_ivf.BuildIndex(base, base_number);
        faiss_ivf.SaveIndex(index_path);
    }
    
    // 显示索引信息
    faiss_ivf.PrintIndexInfo();
    
    // 运行性能测试
    RunPerformanceTest(faiss_ivf, test_query, test_gt, test_number, k, test_gt_d, nprobe_values);
    
    // 单独测试最优参数
    std::cout << "\n=== Final Test with Optimal Parameters ===" << std::endl;
    const int optimal_nprobe = 128;
    
    std::vector<std::vector<int>> final_results;
    faiss_ivf.Search(test_query, test_number, optimal_nprobe, k, final_results);
    
    // 计算最终recall
    float final_recall = 0.0f;
    for (size_t i = 0; i < test_number; i++) {
        std::set<int> gt_set;
        for (int j = 0; j < k; j++) {
            gt_set.insert(test_gt[i * test_gt_d + j]);
        }
        
        int hits = 0;
        for (int j = 0; j < k; j++) {
            if (gt_set.count(final_results[i][j]) > 0) {
                hits++;
            }
        }
        final_recall += (float)hits / k;
    }
    final_recall /= test_number;
    
    std::cout << "Final Recall@" << k << ": " << final_recall << std::endl;
    
    // 清理内存
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    return 0;
}
