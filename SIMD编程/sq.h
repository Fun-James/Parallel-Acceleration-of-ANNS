#pragma once
#include <vector>
#include <queue>
#include <cstdint>
#include <cmath>
#include <limits>
#include <utility>
#include <algorithm>
#include <fstream>
#include <string>
#include <iostream>
#include <arm_neon.h>

// ================= NEON 工具 =========================
namespace sq_neon {

inline float hsum_f32x4(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);          // AArch64 自带水平求和
#else
    float32x2_t t = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(t,0) + vget_lane_f32(t,1);
#endif
}

/*  dot_u8_f32 :
    输入  – vec_u8 : uint8_t*  （被量化的库向量）
           w      : float*    （预先算好的  w[d] = scale[d]*query[d]）
    输出  – Σ vec_u8[d] * w[d]  (float)
    采用 16 维一批的 NEON 并行
*/
inline float dot_u8_f32(const uint8_t* vec_u8,
                        const float*    w,
                        size_t          dim)
{
    size_t d = 0;
    float32x4_t acc = vdupq_n_f32(0.f);

    for (; d + 16 <= dim; d += 16) {
        uint8x16_t  vu8  = vld1q_u8(vec_u8 + d);

        // 扩宽到 u16
        uint16x8_t  vu16_low  = vmovl_u8(vget_low_u8 (vu8));
        uint16x8_t  vu16_high = vmovl_u8(vget_high_u8(vu8));

        // u16 → u32 → f32 (4 lanes 一组)
        float32x4_t vf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16 (vu16_low )));
        float32x4_t vf1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vu16_low )));
        float32x4_t vf2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16 (vu16_high)));
        float32x4_t vf3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vu16_high)));

        // 对应权重 w
        float32x4_t vw0 = vld1q_f32(w + d);
        float32x4_t vw1 = vld1q_f32(w + d + 4);
        float32x4_t vw2 = vld1q_f32(w + d + 8);
        float32x4_t vw3 = vld1q_f32(w + d + 12);

        acc = vmlaq_f32(acc, vf0, vw0);
        acc = vmlaq_f32(acc, vf1, vw1);
        acc = vmlaq_f32(acc, vf2, vw2);
        acc = vmlaq_f32(acc, vf3, vw3);
    }

    float sum = hsum_f32x4(acc);

    // 剩余尾巴（<16）
    for (; d < dim; ++d)
        sum += vec_u8[d] * w[d];

    return sum;
}

} // namespace sq_neon


// ================= SQIndex ===========================
class SQIndex {
public:
    SQIndex() = default;

    /* ---------- build: 量化 base ---------- */
    void build(const float* base, size_t nb_, size_t dim_) {
        nb  = nb_;
        dim = dim_;
        scale .assign(dim, 0.f);
        offset.assign(dim, 0.f);

        // 1. min/max → scale & offset
        for (size_t d = 0; d < dim; ++d) {
            float mn =  std::numeric_limits<float>::max();
            float mx = -std::numeric_limits<float>::max();
            for (size_t i = 0; i < nb; ++i) {
                float v = base[i*dim + d];
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            float rng = (mx - mn < 1e-12f) ? 1.f : (mx - mn);
            scale [d] = rng / 255.f;
            offset[d] = mn;
        }

        // 2. 量化
        base_u8.resize(nb * dim);
        for (size_t i = 0; i < nb; ++i)
            for (size_t d = 0; d < dim; ++d) {
                float q = std::round((base[i*dim + d] - offset[d]) / scale[d]);
                q = std::min(255.f, std::max(0.f, q));
                base_u8[i*dim + d] = static_cast<uint8_t>(q);
            }
    }

    /* ---------- save / load ---------- */
    bool save(const std::string& fn = "files/sq.index") const {
        std::ofstream fout(fn, std::ios::binary);
        if(!fout) return false;
        const char magic[6] = "SQIDX";
        uint8_t ver = 3;
        fout.write(magic,5); fout.write((char*)&ver,1);
        fout.write((char*)&nb , sizeof(uint64_t));
        fout.write((char*)&dim, sizeof(uint32_t));
        fout.write((char*)scale .data(), sizeof(float)*dim);
        fout.write((char*)offset.data(), sizeof(float)*dim);
        fout.write((char*)base_u8.data(), nb*dim);
        return true;
    }

    bool load(const std::string& fn = "files/sq.index") {
        std::ifstream fin(fn, std::ios::binary);
        if(!fin) return false;
        char magic[6]={0}; fin.read(magic,5);
        if(std::string(magic,5)!="SQIDX") return false;
        uint8_t ver=0; fin.read((char*)&ver,1);
        if(ver!=3) return false;               // 版本不符
        fin.read((char*)&nb , sizeof(uint64_t));
        fin.read((char*)&dim, sizeof(uint32_t));
        scale .resize(dim); offset.resize(dim);
        fin.read((char*)scale .data(), sizeof(float)*dim);
        fin.read((char*)offset.data(), sizeof(float)*dim);
        base_u8.resize(nb*dim);
        fin.read((char*)base_u8.data(), nb*dim);
        return static_cast<bool>(fin);
    }

    /* ---------- search ---------- */
    std::priority_queue<std::pair<float,uint32_t>>
    search(const float* query, size_t k) const
    {
        /* 1. w[d] = scale[d] * query[d] */
        static thread_local std::vector<float> w;
        w.resize(dim);
        for (size_t d = 0; d < dim; ++d)
            w[d] = scale[d] * query[d];

        using Pair = std::pair<float,uint32_t>;
        std::priority_queue<Pair> topk;

        /* 2. 遍历库，NEON 点乘 */
        for (size_t i = 0; i < nb; ++i) {
            const uint8_t* vec = base_u8.data() + i*dim;
            float ip = sq_neon::dot_u8_f32(vec, w.data(), dim);
            float dist = -ip;                         // 越小越好

            if (topk.size() < k) topk.emplace(dist,(uint32_t)i);
            else if (dist < topk.top().first){
                topk.pop(); topk.emplace(dist,(uint32_t)i);
            }
        }
        return topk;
    }

private:
    uint64_t             nb  = 0;
    uint32_t             dim = 0;
    std::vector<float>   scale, offset;
    std::vector<uint8_t> base_u8;
};


// ================= 懒加载包装 =========================
inline
std::priority_queue<std::pair<float,uint32_t>>
sq_search(const float* base,
            const float* query,
            size_t       base_number,
            size_t       vecdim,
            size_t       k)
{
    static SQIndex index;
    static bool ready = false;

    if (!ready) {
        if (!index.load()) {
            std::cerr<<"[SQ] load failed, building...\n";
            index.build(base, base_number, vecdim);
            index.save();
            std::cerr<<"[SQ] build & save done\n";
        } else {
            std::cerr<<"[SQ] index loaded\n";
        }
        ready = true;
    }
    return index.search(query, k);
}