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
#include "ivf+hnsw.h"

// 可以自行添加需要的头文件

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

void build_ivf_hnsw_index(float* base, size_t base_number, size_t vecdim, int numClusters)
{
    // 创建IVF+HNSW索引
    IVF_HNSW ivf_hnsw(numClusters, vecdim);
    
    // 构建索引
    ivf_hnsw.build(base, base_number, vecdim);
    
    // 保存索引
    ivf_hnsw.save("files/ivf_hnsw.index");
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

    const size_t k = 10;
    const int numClusters = 64; // 聚类数量，可以根据需要调整
    const int nprobe = 8;       // 查询时探测的聚类数量

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 创建InnerProductSpace空间
    InnerProductSpace ipspace(vecdim);

    // 构建或加载IVF+HNSW索引
    IVF_HNSW* ivf_hnsw = nullptr;
    std::ifstream idxfin("files/ivf_hnsw.index", std::ios::binary);
    if (idxfin.good()) {
        std::cerr << "Loading IVF+HNSW index from files/ivf_hnsw.index" << std::endl;
        idxfin.close();
        ivf_hnsw = new IVF_HNSW(numClusters, vecdim);
        ivf_hnsw->load("files/ivf_hnsw.index", &ipspace);
    } else {
        std::cerr << "Building IVF+HNSW index and saving to files/ivf_hnsw.index" << std::endl;
        build_ivf_hnsw_index(base, base_number, vecdim, numClusters);
        
        // 加载刚刚构建的索引
        ivf_hnsw = new IVF_HNSW(numClusters, vecdim);
        ivf_hnsw->load("files/ivf_hnsw.index", &ipspace);
    }
    // 查询测试代码
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 使用IVF+HNSW进行ANN查询
        auto result = ivf_hnsw->query(test_query + i*vecdim, k, 15, nprobe);

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        for(const auto& item : result) {
            int x = item.second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
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
    std::cout << "average recall: " << avg_recall / test_number << "\n";
    std::cout << "average latency (us): " << avg_latency / test_number << "\n";
    
    delete ivf_hnsw;
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    return 0;
}
