#pragma once
#include <arm_neon.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

constexpr int  PQ_M = 16;   // 子空间数
constexpr int  PQ_K = 16;   // 每子空间中心数 (4‑bit)
constexpr int  BLOCK = 16;  // Fast‑Scan 并行向量数 (一个 NEON 128bit 寄存器)

static inline size_t roundup(size_t x, size_t to) { return (x + to - 1) / to * to; }

/********************************************************************
 *                          PQIndex
 *******************************************************************/
struct PQIndex {
    /************* 训练得到的静态成员 *************/
    int    M  = PQ_M;
    int    K  = PQ_K;
    int    dim{};          // 原始维度
    int    sub_dim{};      // dim / M
    std::vector<std::vector<std::vector<float>>> codebooks; 
    std::vector<std::vector<uint8_t>>            codes;    

    /*************   Fast‑Scan runtime 缓存  *************/
    bool              fastscan_ok = false;
    uint8_t*          codes_fs    = nullptr;   
    size_t            ntotal{};                
    size_t            ntotal2{};              
    float             last_scale{};            
    uint8_t           LUT_q8[PQ_M * PQ_K]{};   

    /*************   rerank 用到的原始数据  *************/
    const float*      base_data = nullptr;
    size_t            base_num  = 0;

    /****************************************************************
     *  1) 把 codes 转置为 Fast‑Scan 布局：同一 sub‑quantizer 的
     *     连续 16 个向量的码字放在一起，便于 vld1q_u8 批量取出。
     ****************************************************************/
    void build_fastscan_layout() {
        if (codes.empty()) return;
        ntotal  = codes.size();
        ntotal2 = roundup(ntotal, BLOCK);
        if (codes_fs) delete[] codes_fs;
        codes_fs = new uint8_t[ntotal2 * M]();        
        for (size_t i = 0; i < ntotal; ++i) {
            size_t blk  = i / BLOCK;
            size_t pos  = i % BLOCK;
            for (int m = 0; m < M; ++m) {
                codes_fs[ blk * (M * BLOCK) + m * BLOCK + pos ] = codes[i][m];
            }
        }
        fastscan_ok = true;
        std::cerr << "[FastScan] layout ready, ntotal=" << ntotal
                  << ", roundup=" << ntotal2 << ", bytes=" << ntotal2 * M << "\n";
    }

    /****************************************************************
     *  2)  构建本 query 的 LUT 并量化到 uint8  
     ****************************************************************/
    void build_quantized_LUT(const float* query) {
        assert(codebooks.size() == (size_t)M);
        sub_dim = dim / M;
        float  maxv = 0.f;
        for (int m = 0; m < M; ++m) {
            const float* qsub = query + m * sub_dim;
            for (int k = 0; k < K; ++k) {
                const float* c = codebooks[m][k].data();
                float dist = 0.f;
                for (int d = 0; d < sub_dim; ++d) {
                    float diff = qsub[d] - c[d];
                    dist += diff * diff;
                }
                LUT_q8[m * K + k] = 0;            // 先占位
                if (dist > maxv) maxv = dist;
            }
        }
        // 重新来一遍，做量化
        last_scale = maxv > 0 ? maxv / 255.f : 1.f;
        for (int m = 0; m < M; ++m) {
            const float* qsub = query + m * sub_dim;
            for (int k = 0; k < K; ++k) {
                const float* c = codebooks[m][k].data();
                float dist = 0.f;
                for (int d = 0; d < sub_dim; ++d) {
                    float diff = qsub[d] - c[d];
                    dist += diff * diff;
                }
                int q = int(dist / last_scale + 0.5f);
                if (q > 255) q = 255;
                LUT_q8[m * K + k] = (uint8_t)q;
            }
        }
    }

    /****************************************************************
     *  3)  Fast‑Scan 核：一次处理 BLOCK=16 向量
     *      输入:
     *           blk_ptr -> codes_fs + blk*(M*BLOCK)
     *           LUT_q8  -> 256B   (已 ready)
     *      输出:
     *           acc16[16] : 累加后的 16‑bit 距离
     ****************************************************************/
    static inline void scan_block_16(const uint8_t* blk_ptr,
                                     const uint8_t* LUT,
                                     uint16_t*      acc16,
                                     int            M)
    {
        uint16x8_t acc0 = vdupq_n_u16(0);   // vectors 0..7
        uint16x8_t acc1 = vdupq_n_u16(0);   // vectors 8..15

        for (int m = 0; m < M; ++m)
        {
            uint8x16_t lut = vld1q_u8(LUT + m * 16);          // 16‑entry LUT_k

            // load 16 codes (1B each, already 0‑15)
            uint8x16_t codes = vld1q_u8(blk_ptr + m * BLOCK);

            // 查表
            uint8x16_t dists = vqtbl1q_u8(lut, codes);        // 16 × u8

            // widen & accum
            uint16x8_t lo = vmovl_u8(vget_low_u8 (dists));
            uint16x8_t hi = vmovl_u8(vget_high_u8(dists));
            acc0 = vaddq_u16(acc0, lo);
            acc1 = vaddq_u16(acc1, hi);
        }
        vst1q_u16(acc16,     acc0);
        vst1q_u16(acc16 + 8, acc1);
    }

    /****************************************************************
     *  4)  对外查询接口：返回 top‑k id 
     ****************************************************************/
    std::priority_queue<std::pair<float, uint32_t>>
    query(const float* q, int k, int rerank_k)
    {
        if (!fastscan_ok) build_fastscan_layout();
        build_quantized_LUT(q);

        typedef std::pair<uint32_t,uint32_t> PairU32;
std::priority_queue<
        PairU32,
        std::vector<PairU32>,
        std::less<PairU32> > cand_max;   

        size_t nblk = ntotal2 / BLOCK;
        for (size_t b = 0; b < nblk; ++b) {
            uint16_t acc[BLOCK];
            scan_block_16(codes_fs + b * (M * BLOCK), LUT_q8, acc, M);

            for (int i = 0; i < BLOCK; ++i) {
                size_t vid = b * BLOCK + i;
                if (vid >= ntotal) break;
                uint32_t dist16 = acc[i];
                if (cand_max.size() < (size_t)rerank_k) {
                    cand_max.emplace(dist16, (uint32_t)vid);
                } else if (dist16 < cand_max.top().first) {
                    cand_max.pop();
                    cand_max.emplace(dist16, (uint32_t)vid);
                }
            }
        }

        /************ 5)  精确距离重排 (使用内积距离) ************/
        auto exact_dist = [&](const float* a, const float* b)->float {
            float32x4_t acc = vdupq_n_f32(0);
            int d=0;
            for (; d+3 < dim; d+=4){
                float32x4_t va=vld1q_f32(a+d);
                float32x4_t vb=vld1q_f32(b+d);
                // 计算内积
                acc = vfmaq_f32(acc, va, vb);
            }
            float tmp[4]; vst1q_f32(tmp, acc);
            float res = tmp[0]+tmp[1]+tmp[2]+tmp[3];
            for (; d<dim; ++d){
                // 剩余元素的内积
                res += a[d] * b[d];
            }
            // 返回负内积作为距离（内积越大越相似，距离应该越小）
            return -res;
        };

        std::priority_queue<std::pair<float,uint32_t>> final_top;
        while (!cand_max.empty()) {
            uint32_t id  = cand_max.top().second;
            cand_max.pop();
            float dist = exact_dist(q, base_data + (size_t)id * dim);
            if (final_top.size() < (size_t)k) {
                final_top.emplace(dist, id);
            } else if (dist < final_top.top().first) {
                final_top.pop();  final_top.emplace(dist,id);
            }
        }
        return final_top;
    }

    /***********************  其余函数保留  **************************/
    bool load(const std::string& fname) {
        std::ifstream fin(fname, std::ios::binary);
        if (!fin) return false;
        fin.read((char*)&M,   sizeof(int));
        fin.read((char*)&K,   sizeof(int));
        fin.read((char*)&dim, sizeof(int));
        sub_dim = dim / M;
        codebooks.resize(M, std::vector<std::vector<float>>(K, std::vector<float>(sub_dim)));
        for (int m=0;m<M;++m)
            for (int k=0;k<K;++k)
                fin.read((char*)codebooks[m][k].data(), sub_dim*sizeof(float));

        int n; fin.read((char*)&n, sizeof(int));
        codes.resize(n, std::vector<uint8_t>(M));
        for (int i=0;i<n;++i)
            fin.read((char*)codes[i].data(), M);
        fin.close();
        return true;
    }
    void set_base_data(const float* b, size_t n){ base_data=b; base_num=n; }
    ~PQIndex(){ delete[] codes_fs; }
};

/************************ 全局包装 *************************/
static PQIndex g_pq_index;

inline void pq_init_fastscan(){ g_pq_index.build_fastscan_layout(); }

inline std::priority_queue<std::pair<float,uint32_t>>
pq_search(float* base, float* query, size_t base_n, size_t dim,
          size_t k, int rerank_k)
{
    g_pq_index.set_base_data(base, base_n);
    g_pq_index.dim = dim;            // 若未从磁盘加载可手动设
    return g_pq_index.query(query, (int)k, rerank_k);
}