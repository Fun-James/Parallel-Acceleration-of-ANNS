# Parallel Acceleration of ANNS

本项目旨在探索和实现近似最近邻搜索（ANNS）算法的并行加速优化。项目包含基础实现以及基于不同并行计算架构（SIMD, OpenMP, MPI, GPU）的优化版本。

## 📂 项目结构

- **`annCompleteCode`**: 基础代码实现，包含 HNSW 算法的核心逻辑 (`hnswlib`) 和主程序。
- **`SIMD编程`**: 基于 SIMD 指令集（如 X86 AVX, ARM NEON）的向量化优化实现。
- **`pthreadOpenMP`**: 基于 pthread 和 OpenMP 的多线程并行优化实现。
- **`MPI`**: 基于 MPI (Message Passing Interface) 的分布式并行实现。
- **`GPU`**: 基于 CUDA 的 GPU 加速实现。
- **报告文档**: 根目录下包含各阶段的实验报告和理论调研文档。



## 📚 研究报告
项目包含详细的 PDF 报告，涵盖了从 SIMD 到 GPU 加速的详细实验数据和分析，一份较为全面的研究总结在 ANN 研究报告.pdf 中。
