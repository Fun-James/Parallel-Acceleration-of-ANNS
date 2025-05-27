#pragma once

#include <vector>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cmath>
#include <memory>
#include <fstream>
#include <string>
#include <omp.h>  // 添加OpenMP头文件
#include "hnswlib/hnswlib/hnswlib.h"

using namespace hnswlib;

// 串行版本的HNSW实现
class HNSW_Serial {
private:
    // 内部节点结构
    struct Node {
        std::vector<std::vector<size_t>> connections; // 每层的连接
        float* data;                                 // 向量数据
        bool valid;                                  // 节点是否有效
        size_t id;                                   // 节点ID

        Node(size_t id, float* point, size_t dim, size_t maxLevel) : id(id), valid(true) {
            connections.resize(maxLevel + 1);
            data = new float[dim];
            memcpy(data, point, dim * sizeof(float));
        }

        ~Node() {
            delete[] data;
        }
    };

    SpaceInterface<float>* space;     // 距离空间
    size_t dim;                       // 向量维度
    size_t maxElements;               // 最大元素数
    size_t curElements;               // 当前元素数
    size_t M;                         // 每个节点的最大出边数
    size_t maxM;                      // 第0层以上的最大出边数
    size_t maxM0;                     // 第0层的最大出边数
    size_t efConstruction;            // 构建时的ef参数
    float ml;                         // 层级乘数
    size_t maxLevel;                  // 当前最大层级
    std::vector<Node*> nodes;         // 所有节点
    size_t entryPointId;              // 入口点ID
    std::default_random_engine rng;   // 随机数生成器
    int num_threads;                  // OpenMP线程数

public:
    HNSW_Serial(SpaceInterface<float>* s, size_t maxElements, size_t M = 16, size_t efConstruction = 200) 
        : space(s), maxElements(maxElements), M(M), efConstruction(efConstruction) {
        dim = space->get_data_size();
        maxM = M;
        maxM0 = 2 * M;
        ml = 1.0 / log(1.0 * M);
        maxLevel = 0;
        curElements = 0;
        entryPointId = 0;
        nodes.resize(maxElements, nullptr);
        
        // 获取系统可用的线程数
        num_threads = omp_get_max_threads();
        std::cout << "Using " << num_threads << " threads for index construction" << std::endl;
    }

    ~HNSW_Serial() {
        // 清理所有节点
        for (auto node : nodes) {
            if (node) delete node;
        }
    }

    // 获取随机层级
    size_t getRandomLevel() {
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        float r = distribution(rng);
        return -log(r) * ml;
    }

    // 在特定层搜索最近邻
    std::priority_queue<std::pair<float, size_t>> searchLayer(
        const float* query, size_t ef, size_t entryPoint, size_t level) {
        std::unordered_set<size_t> visited;
        std::priority_queue<std::pair<float, size_t>> topCandidates;
        std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, std::greater<std::pair<float, size_t>>> candidates;

        float dist = space->get_dist_func()(query, nodes[entryPoint]->data, space->get_dist_func_param());
        topCandidates.emplace(dist, entryPoint);
        candidates.emplace(dist, entryPoint);
        visited.insert(entryPoint);

        while (!candidates.empty()) {
            std::pair<float, size_t> current = candidates.top();
            float lowerBound = topCandidates.top().first;
            
            if (current.first > lowerBound && topCandidates.size() >= ef) {
                break;
            }
            
            candidates.pop();
            
            // 检查当前节点的所有邻居
            for (size_t neighborId : nodes[current.second]->connections[level]) {
                if (visited.find(neighborId) == visited.end()) {
                    visited.insert(neighborId);
                    float neighborDist = space->get_dist_func()(query, nodes[neighborId]->data, space->get_dist_func_param());
                    
                    if (topCandidates.size() < ef || neighborDist < lowerBound) {
                        candidates.emplace(neighborDist, neighborId);
                        topCandidates.emplace(neighborDist, neighborId);
                        
                        if (topCandidates.size() > ef) {
                            topCandidates.pop();
                        }
                        
                        if (!topCandidates.empty()) {
                            lowerBound = topCandidates.top().first;
                        }
                    }
                }
            }
        }
        
        return topCandidates;
    }

    // 并行在特定层搜索最近邻 (仅构建时使用)
    std::priority_queue<std::pair<float, size_t>> searchLayerParallel(
        const float* query, size_t ef, size_t entryPoint, size_t level) {
        std::unordered_set<size_t> visited;
        std::priority_queue<std::pair<float, size_t>> topCandidates;
        std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, std::greater<std::pair<float, size_t>>> candidates;

        float dist = space->get_dist_func()(query, nodes[entryPoint]->data, space->get_dist_func_param());
        topCandidates.emplace(dist, entryPoint);
        candidates.emplace(dist, entryPoint);
        visited.insert(entryPoint);

        while (!candidates.empty()) {
            std::pair<float, size_t> current = candidates.top();
            float lowerBound = topCandidates.top().first;
            
            if (current.first > lowerBound && topCandidates.size() >= ef) {
                break;
            }
            
            candidates.pop();
            
            // 获取当前节点的所有邻居
            std::vector<size_t>& neighbors = nodes[current.second]->connections[level];
            size_t neighborsCount = neighbors.size();
            
            if (neighborsCount > 0) {
                // 为并行计算准备数据结构
                std::vector<float> neighborDistances(neighborsCount);
                std::vector<bool> shouldAdd(neighborsCount, false);
                
                // 并行计算距离
                #pragma omp parallel for if(neighborsCount > 100) // 只有当邻居数量足够多时才并行
                for (int i = 0; i < neighborsCount; i++) {
                    size_t neighborId = neighbors[i];
                    if (visited.find(neighborId) == visited.end()) {
                        float neighborDist = space->get_dist_func()(query, nodes[neighborId]->data, space->get_dist_func_param());
                        neighborDistances[i] = neighborDist;
                        shouldAdd[i] = true;
                    }
                }
                
                // 串行处理结果（这部分不能并行，因为涉及到共享数据结构的修改）
                for (int i = 0; i < neighborsCount; i++) {
                    if (shouldAdd[i]) {
                        size_t neighborId = neighbors[i];
                        visited.insert(neighborId);
                        float neighborDist = neighborDistances[i];
                        
                        if (topCandidates.size() < ef || neighborDist < lowerBound) {
                            candidates.emplace(neighborDist, neighborId);
                            topCandidates.emplace(neighborDist, neighborId);
                            
                            if (topCandidates.size() > ef) {
                                topCandidates.pop();
                            }
                            
                            if (!topCandidates.empty()) {
                                lowerBound = topCandidates.top().first;
                            }
                        }
                    }
                }
            }
        }
        
        return topCandidates;
    }

    // 添加一个节点到图中
    void addPoint(float* point, size_t id) {
        if (id >= maxElements) {
            throw std::runtime_error("Element id exceeds maximum capacity");
        }
        
        if (nodes[id] != nullptr) {
            throw std::runtime_error("Element already exists");
        }
        
        size_t nodeLevel = getRandomLevel();
        Node* newNode = new Node(id, point, dim, nodeLevel);
        nodes[id] = newNode;
        
        if (curElements == 0) {
            entryPointId = id;
            maxLevel = nodeLevel;
            curElements++;
            std::cout << "Added first point (ID: " << id << ") as entry point with level " << nodeLevel << std::endl;
            return;
        }
        
        // 寻找插入点
        size_t currObj = entryPointId;
        float curDist = space->get_dist_func()(point, nodes[currObj]->data, space->get_dist_func_param());
        
        if (id % 1000 == 0) {
            std::cout << "Processing point ID: " << id << " with level " << nodeLevel << std::endl;
        }
        
        // 从最高层开始向下遍历
        for (int level = maxLevel; level > nodeLevel; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t neighborId : nodes[currObj]->connections[level]) {
                    float neighborDist = space->get_dist_func()(point, nodes[neighborId]->data, space->get_dist_func_param());
                    if (neighborDist < curDist) {
                        curDist = neighborDist;
                        currObj = neighborId;
                        changed = true;
                    }
                }
            }
        }
        
        std::vector<std::priority_queue<std::pair<float, size_t>>> topCandidatesList(nodeLevel + 1);
        
        // 对每一层进行搜索和连接
        for (int level = std::min(nodeLevel, maxLevel); level >= 0; level--) {
            // 在构建时使用并行搜索
            std::priority_queue<std::pair<float, size_t>> topCandidates = searchLayerParallel(point, efConstruction, currObj, level);
            
            // 选择最近的M个节点作为连接
            size_t M_val = level == 0 ? maxM0 : maxM;
            while (topCandidates.size() > M_val) {
                topCandidates.pop();
            }
            
            topCandidatesList[level] = topCandidates;
            
            // 准备连接修剪所需的数据结构
            std::vector<std::vector<std::pair<float, size_t>>> allNeighborDistances;
            std::vector<size_t> neighborsToProcess;
            
            // 建立双向连接
            while (!topCandidates.empty()) {
                size_t neighborId = topCandidates.top().second;
                topCandidates.pop();
                
                newNode->connections[level].push_back(neighborId);
                nodes[neighborId]->connections[level].push_back(id);
                
                // 如果连接数超过限制，收集需要修剪的节点
                if (nodes[neighborId]->connections[level].size() > M_val) {
                    neighborsToProcess.push_back(neighborId);
                }
            }
            
            // 并行处理需要修剪的节点
            if (!neighborsToProcess.empty()) {
                allNeighborDistances.resize(neighborsToProcess.size());
                
                #pragma omp parallel for
                for (int i = 0; i < neighborsToProcess.size(); i++) {
                    size_t neighborId = neighborsToProcess[i];
                    std::vector<size_t>& connections = nodes[neighborId]->connections[level];
                    
                    std::vector<std::pair<float, size_t>> distances;
                    distances.reserve(connections.size());
                    
                    // 计算到所有连接节点的距离
                    for (size_t nId : connections) {
                        float dist = space->get_dist_func()(nodes[neighborId]->data, nodes[nId]->data, space->get_dist_func_param());
                        distances.emplace_back(dist, nId);
                    }
                    
                    // 排序
                    std::sort(distances.begin(), distances.end());
                    allNeighborDistances[i] = std::move(distances);
                }
                
                // 串行应用修剪结果
                for (int i = 0; i < neighborsToProcess.size(); i++) {
                    size_t neighborId = neighborsToProcess[i];
                    std::vector<size_t>& connections = nodes[neighborId]->connections[level];
                    
                    connections.clear();
                    for (size_t j = 0; j < M_val && j < allNeighborDistances[i].size(); j++) {
                        connections.push_back(allNeighborDistances[i][j].second);
                    }
                }
            }
            
            currObj = topCandidatesList[level].top().second;
        }
        
        // 更新入口点和最大层级
        if (nodeLevel > maxLevel) {
            entryPointId = id;
            maxLevel = nodeLevel;
            std::cout << "New entry point: " << id << " with max level: " << maxLevel << std::endl;
        }
        
        curElements++;
    }

    // 批量预处理向量数据（并行）
    void preprocessVectors(float* data, size_t numElements) {
        std::cout << "Preprocessing vectors in parallel..." << std::endl;
        
        #pragma omp parallel for
        for (size_t i = 0; i < numElements; i++) {
            // 这里可以进行归一化或其他预处理操作
            float* point = data + i * dim;
            
            // 例如，进行L2归一化
            float norm = 0;
            for (size_t j = 0; j < dim; j++) {
                norm += point[j] * point[j];
            }
            norm = sqrt(norm);
            
            if (norm > 0) {
                for (size_t j = 0; j < dim; j++) {
                    point[j] /= norm;
                }
            }
        }
    }

    // 建立索引
    void buildIndex(float* data, size_t numElements) {
        std::cout << "======= Starting HNSW index construction =======" << std::endl;
        std::cout << "Total points to index: " << numElements << std::endl;
        std::cout << "Vector dimension: " << dim << std::endl;
        std::cout << "M parameter: " << M << ", efConstruction: " << efConstruction << std::endl;
        std::cout << "Using " << num_threads << " threads for index construction" << std::endl;
        
        // 并行预处理向量数据
        // preprocessVectors(data, numElements);
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // 这部分必须是串行的，因为每个节点的添加依赖于前面的图结构
        for (size_t i = 0; i < numElements; i++) {
            addPoint(data + i * dim, i);
            
            // 每处理10%的数据显示一次进度
            if (i > 0 && (i % (numElements / 10) == 0 || i == numElements - 1)) {
                float progress = 100.0f * i / numElements;
                auto currentTime = std::chrono::high_resolution_clock::now();
                auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();
                
                std::cout << "Progress: " << progress << "% (" << i << "/" << numElements << " points)" << std::endl;
                std::cout << "Time elapsed: " << elapsedSeconds << " seconds" << std::endl;
                std::cout << "Current max level: " << maxLevel << std::endl;
                
                // 估算剩余时间
                if (i > 0) {
                    double pointsPerSecond = static_cast<double>(i) / elapsedSeconds;
                    double remainingSeconds = (numElements - i) / pointsPerSecond;
                    std::cout << "Estimated time remaining: " << static_cast<int>(remainingSeconds) << " seconds" << std::endl;
                }
                std::cout << "------------------------------------------------" << std::endl;
            }
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto totalSeconds = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
        std::cout << "======= HNSW index construction complete =======" << std::endl;
        std::cout << "Total construction time: " << totalSeconds << " seconds" << std::endl;
        std::cout << "Final index statistics:" << std::endl;
        std::cout << "  - Number of elements: " << curElements << std::endl;
        std::cout << "  - Maximum level: " << maxLevel << std::endl;
        std::cout << "  - Entry point ID: " << entryPointId << std::endl;
    }

    // 查询K近邻
    std::vector<std::pair<float, size_t>> searchKNN(const float* query, size_t k, size_t ef = 50) {
        if (curElements == 0) {
            return {};
        }
        
        std::vector<std::pair<float, size_t>> result;
        size_t currObj = entryPointId;
        float curDist = space->get_dist_func()(query, nodes[currObj]->data, space->get_dist_func_param());
        
        // 从最高层开始向下遍历
        for (int level = maxLevel; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t neighborId : nodes[currObj]->connections[level]) {
                    float neighborDist = space->get_dist_func()(query, nodes[neighborId]->data, space->get_dist_func_param());
                    if (neighborDist < curDist) {
                        curDist = neighborDist;
                        currObj = neighborId;
                        changed = true;
                    }
                }
            }
        }
        
        // 在底层进行精确搜索
        std::priority_queue<std::pair<float, size_t>> topCandidates = searchLayer(query, std::max(ef, k), currObj, 0);
        
        // 构建结果集
        result.resize(std::min(topCandidates.size(), k));
        for (int i = result.size() - 1; i >= 0; i--) {
            result[i] = topCandidates.top();
            topCandidates.pop();
        }
        
        return result;
    }

    // 保存索引到文件
    void saveIndex(const std::string& filename) {
        std::cout << "Saving index to " << filename << "..." << std::endl;
        std::ofstream output(filename, std::ios::binary);
        
        // 保存基本参数
        output.write((char*)&dim, sizeof(size_t));
        output.write((char*)&maxElements, sizeof(size_t));
        output.write((char*)&curElements, sizeof(size_t));
        output.write((char*)&M, sizeof(size_t));
        output.write((char*)&maxM, sizeof(size_t));
        output.write((char*)&maxM0, sizeof(size_t));
        output.write((char*)&efConstruction, sizeof(size_t));
        output.write((char*)&ml, sizeof(float));
        output.write((char*)&maxLevel, sizeof(size_t));
        output.write((char*)&entryPointId, sizeof(size_t));
        
        // 保存所有节点
        for (size_t i = 0; i < maxElements; i++) {
            bool nodeExists = (nodes[i] != nullptr);
            output.write((char*)&nodeExists, sizeof(bool));
            
            if (nodeExists) {
                // 保存节点数据
                output.write((char*)nodes[i]->data, dim * sizeof(float));
                output.write((char*)&nodes[i]->id, sizeof(size_t));
                
                // 保存节点连接
                size_t numLevels = nodes[i]->connections.size();
                output.write((char*)&numLevels, sizeof(size_t));
                
                for (size_t level = 0; level < numLevels; level++) {
                    size_t numConnections = nodes[i]->connections[level].size();
                    output.write((char*)&numConnections, sizeof(size_t));
                    
                    if (numConnections > 0) {
                        output.write((char*)&nodes[i]->connections[level][0], numConnections * sizeof(size_t));
                    }
                }
            }
        }
        
        output.close();
    }

    // 从文件加载索引
    void loadIndex(const std::string& filename, SpaceInterface<float>* s) {
        std::ifstream input(filename, std::ios::binary);
        
        // 清理现有节点
        for (auto node : nodes) {
            if (node) delete node;
        }
        nodes.clear();
        
        // 加载基本参数
        input.read((char*)&dim, sizeof(size_t));
        input.read((char*)&maxElements, sizeof(size_t));
        input.read((char*)&curElements, sizeof(size_t));
        input.read((char*)&M, sizeof(size_t));
        input.read((char*)&maxM, sizeof(size_t));
        input.read((char*)&maxM0, sizeof(size_t));
        input.read((char*)&efConstruction, sizeof(size_t));
        input.read((char*)&ml, sizeof(float));
        input.read((char*)&maxLevel, sizeof(size_t));
        input.read((char*)&entryPointId, sizeof(size_t));
        
        // 设置空间
        space = s;
        
        // 加载所有节点
        nodes.resize(maxElements, nullptr);
        
        for (size_t i = 0; i < maxElements; i++) {
            bool nodeExists;
            input.read((char*)&nodeExists, sizeof(bool));
            
            if (nodeExists) {
                float* data = new float[dim];
                input.read((char*)data, dim * sizeof(float));
                
                size_t id;
                input.read((char*)&id, sizeof(size_t));
                
                // 创建节点
                nodes[i] = new Node(id, data, dim, 0);
                delete[] data;
                
                // 加载节点连接
                size_t numLevels;
                input.read((char*)&numLevels, sizeof(size_t));
                
                nodes[i]->connections.resize(numLevels);
                
                for (size_t level = 0; level < numLevels; level++) {
                    size_t numConnections;
                    input.read((char*)&numConnections, sizeof(size_t));
                    
                    nodes[i]->connections[level].resize(numConnections);
                    
                    if (numConnections > 0) {
                        input.read((char*)&nodes[i]->connections[level][0], numConnections * sizeof(size_t));
                    }
                }
            }
        }
        
        input.close();
    }
};
