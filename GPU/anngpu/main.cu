/*********************************************************************
*  编译命令：nvcc main.cu -O2 -lcublas -o main -std=c++14
*********************************************************************/
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <chrono>
#include <algorithm>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(-1); \
        } \
    } while(0)

// cuBLAS错误检查宏
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t stat = call; \
        if (stat != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(-1); \
        } \
    } while(0)

/* ================================================================
   数据读取
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
   更高效的GPU Top-k kernel (使用堆排序思想)
   ============================================================== */
__device__ void heapify_down(float* heap_dist, int* heap_idx, int heap_size, int idx) {
    int largest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
    
    if (left < heap_size && heap_dist[left] > heap_dist[largest])
        largest = left;
    
    if (right < heap_size && heap_dist[right] > heap_dist[largest])
        largest = right;
    
    if (largest != idx) {
        // Swap distances
        float temp_dist = heap_dist[idx];
        heap_dist[idx] = heap_dist[largest];
        heap_dist[largest] = temp_dist;
        
        // Swap indices
        int temp_idx = heap_idx[idx];
        heap_idx[idx] = heap_idx[largest];
        heap_idx[largest] = temp_idx;
        
        heapify_down(heap_dist, heap_idx, heap_size, largest);
    }
}

__global__ void find_topk_heap(float* distances, int* result_indices, 
                               int n, int batch_size, int k) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= batch_size) return;
    
    // 在shared memory中为每个线程创建小的堆
    extern __shared__ float shared_mem[];
    float* heap_dist = &shared_mem[threadIdx.x * k * 2];
    int* heap_idx = (int*)&heap_dist[k];
    
    // 初始化堆，用前k个元素
    for (int i = 0; i < k && i < n; i++) {
        float inner_product = distances[i * batch_size + query_idx];
        heap_dist[i] = 1.0f - inner_product;  // 转换为距离
        heap_idx[i] = i;
    }
    
    // 建立最大堆
    for (int i = k/2 - 1; i >= 0; i--) {
        heapify_down(heap_dist, heap_idx, k, i);
    }
    
    // 处理剩余元素
    for (int i = k; i < n; i++) {
        float inner_product = distances[i * batch_size + query_idx];
        float dist = 1.0f - inner_product;
        
        // 如果当前距离比堆顶小，替换堆顶
        if (dist < heap_dist[0]) {
            heap_dist[0] = dist;
            heap_idx[0] = i;
            heapify_down(heap_dist, heap_idx, k, 0);
        }
    }
    
    // 将结果复制到全局内存
    for (int i = 0; i < k; i++) {
        result_indices[query_idx * k + i] = heap_idx[i];
    }
}

/* ================================================================
   简单的GPU Top-k kernel (备用版本)
   ============================================================== */
__global__ void find_topk_simple(float* distances, int* result_indices, 
                                 int n, int batch_size, int k) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= batch_size) return;
    
    // 为每个查询找top-k
    for (int target = 0; target < k; target++) {
        float min_dist = 1e10f;
        int min_idx = -1;
        
        // 找到第target小的距离
        for (int i = 0; i < n; i++) {
            float inner_product = distances[i * batch_size + query_idx];
            float dist = 1.0f - inner_product;  // 转换为距离
            
            // 检查是否已经被选中
            bool already_selected = false;
            for (int j = 0; j < target; j++) {
                if (result_indices[query_idx * k + j] == i) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && dist < min_dist) {
                min_dist = dist;
                min_idx = i;
            }
        }
        
        result_indices[query_idx * k + target] = min_idx;
    }
}

/* ================================================================
   主函数
   ============================================================== */
int main(int argc, char *argv[]) {
    // 数据加载
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;
    
    std::string data_path = "./anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    // 参数设置
    test_number = std::min((size_t)10000, test_number);
    const int k = 10;
    const int BATCH_SIZE = 1000;

    // GPU内存分配
    float *d_base, *d_queries, *d_distances;
    int *d_indices;
    
    CUDA_CHECK(cudaMalloc(&d_base, base_number * vecdim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_queries, BATCH_SIZE * vecdim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_distances, base_number * BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, BATCH_SIZE * k * sizeof(int)));
    
    // 拷贝base数据到GPU（只需一次）
    CUDA_CHECK(cudaMemcpy(d_base, base, base_number * vecdim * sizeof(float), cudaMemcpyHostToDevice));
    
    // 创建cuBLAS句柄
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // 处理所有查询
    std::vector<float> all_recalls;
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (size_t batch_start = 0; batch_start < test_number; batch_start += BATCH_SIZE) {
        size_t current_batch_size = std::min((size_t)BATCH_SIZE, test_number - batch_start);
        
        // 拷贝查询到GPU
        CUDA_CHECK(cudaMemcpy(d_queries, test_query + batch_start * vecdim, 
                   current_batch_size * vecdim * sizeof(float), cudaMemcpyHostToDevice));
        
        // 计算内积矩阵: d_distances = d_base * d_queries^T
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    current_batch_size, base_number, vecdim,
                    &alpha,
                    d_queries, vecdim,
                    d_base, vecdim,
                    &beta,
                    d_distances, current_batch_size));
        
        // 找top-k - 尝试使用堆版本，如果shared memory不够则使用简单版本
        int threads = 128;  // 减少线程数以留出更多shared memory
        int blocks = (current_batch_size + threads - 1) / threads;
        
        // 计算所需的shared memory大小
        size_t shared_mem_size = threads * k * (sizeof(float) + sizeof(int));
        
        // 检查设备是否支持所需的shared memory
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        
        if (shared_mem_size <= prop.sharedMemPerBlock) {
            // 使用堆版本
            find_topk_heap<<<blocks, threads, shared_mem_size>>>(d_distances, d_indices, 
                                                      base_number, current_batch_size, k);
        } else {
            // 使用简单版本
            threads = 256;
            blocks = (current_batch_size + threads - 1) / threads;
            find_topk_simple<<<blocks, threads>>>(d_distances, d_indices, 
                                                  base_number, current_batch_size, k);
        }
        
        // 检查kernel执行错误
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 拷贝结果回CPU
        std::vector<int> batch_indices(current_batch_size * k);
        CUDA_CHECK(cudaMemcpy(batch_indices.data(), d_indices, 
                   current_batch_size * k * sizeof(int), cudaMemcpyDeviceToHost));
        
        // 计算recall
        for (size_t i = 0; i < current_batch_size; i++) {
            size_t query_id = batch_start + i;
            
            // Ground truth
            std::set<int> gt_set;
            for (int j = 0; j < k; j++) {
                gt_set.insert(test_gt[query_id * test_gt_d + j]);
            }
            
            // 计算命中数
            int hits = 0;
            for (int j = 0; j < k; j++) {
                if (gt_set.count(batch_indices[i * k + j]) > 0) {
                    hits++;
                }
            }
            
            all_recalls.push_back((float)hits / k);
        }
        
        // 显示进度
        std::cout << "Processed " << batch_start + current_batch_size << "/" << test_number << " queries\r" << std::flush;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    
    // 计算平均recall
    float avg_recall = 0;
    for (auto r : all_recalls) avg_recall += r;
    avg_recall /= all_recalls.size();
    
    // 输出简化结果
    std::cout << "\n\nResults:\n";
    std::cout << "Average recall" << k << ": " << avg_recall << "\n";
    std::cout << "Average latency: " << duration.count() / (float)test_number << " us/query\n";
    std::cout << "QPS: " << test_number * 1000000.0 / duration.count() << " queries/sec\n";
    
    // 清理
    cublasDestroy(handle);
    cudaFree(d_base);
    cudaFree(d_queries);
    cudaFree(d_distances);
    cudaFree(d_indices);
    
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    return 0;
}
