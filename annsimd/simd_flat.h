#pragma once
#include <arm_neon.h>
#include <queue>
#include <cstddef>
#include <cstdint>

float InnerProductSIMDNeon(const float* b1, const float* b2, size_t vecdim) {
    
    // 初始化累加寄存器为0
    float32x4_t sum1 = vdupq_n_f32(0); // 第一个累加寄存器
    float32x4_t sum2 = vdupq_n_f32(0); // 第二个累加寄存器
    
    // 每次处理8个浮点数(使用2个NEON寄存器)
    for (size_t i = 0; i+7 < vecdim; i += 8) {
        // 加载第一个向量的4个float
        float32x4_t v1_1 = vld1q_f32(b1 + i);
        float32x4_t v1_2 = vld1q_f32(b1 + i + 4);
        
        // 加载第二个向量的4个float
        float32x4_t v2_1 = vld1q_f32(b2 + i);
        float32x4_t v2_2 = vld1q_f32(b2 + i + 4);
        
        // 执行乘法并累加到sum寄存器
        sum1 = vfmaq_f32(sum1, v1_1, v2_1);
        sum2 = vfmaq_f32(sum2, v1_2, v2_2);
    }
    
    // 处理剩余元素
size_t leftover = vecdim % 8;
size_t start = vecdim - leftover;
float tail_sum = 0;
for (size_t i = start; i < vecdim; ++i) {
    tail_sum += b1[i] * b2[i];
}
    // 将两个寄存器相加
    sum1 = vaddq_f32(sum1, sum2);
    
    // 将寄存器中的4个浮点数水平相加
    float32x2_t sum_low = vget_low_f32(sum1);
    float32x2_t sum_high = vget_high_f32(sum1);
    float32x2_t sum = vadd_f32(sum_low, sum_high);
    float32x2_t sum_pairwise = vpadd_f32(sum, sum);
    
    float result;
    vst1_lane_f32(&result, sum_pairwise, 0);
    result += tail_sum;
    // 内积距离公式: 1 - 内积值
    return 1.0f - result;
}

// 使用SIMD加速的flat_search函数，替代原有的flat_scan.h中的函数
std::priority_queue<std::pair<float, uint32_t>> simd_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;

    for(int i = 0; i < base_number; ++i) {
        // 使用SIMD加速的内积计算
        float dis = InnerProductSIMDNeon(base + i*vecdim, query, vecdim);

        if(q.size() < k) {
            q.push({dis, i});
        } else {
            if(dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }
    return q;
}