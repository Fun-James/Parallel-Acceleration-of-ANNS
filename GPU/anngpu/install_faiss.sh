#!/bin/bash

# Faiss-GPU 安装脚本
# 适用于Ubuntu/Debian系统

echo "=== Faiss-GPU 安装脚本 ==="

# 1. 安装必要的依赖
echo "1. 安装系统依赖..."
sudo apt-get update
sudo apt-get install -y \
    cmake \
    build-essential \
    libblas-dev \
    liblapack-dev \
    python3-dev \
    python3-pip \
    swig \
    libopenblas-dev \
    git \
    pkg-config \
    libeigen3-dev

# 2. 安装CUDA（如果没有的话）
echo "2. 检查CUDA安装..."
if ! command -v nvcc &> /dev/null; then
    echo "CUDA未安装，请先安装CUDA Toolkit"
    echo "可以从 https://developer.nvidia.com/cuda-downloads 下载"
    echo "或使用包管理器："
    echo "sudo apt install nvidia-cuda-toolkit"
else
    echo "CUDA已安装: $(nvcc --version | grep release)"
fi

# 3. 从源代码编译安装Faiss
echo "3. 编译安装Faiss..."

# 克隆Faiss源代码
if [ ! -d "faiss" ]; then
    git clone https://github.com/facebookresearch/faiss.git
fi

cd faiss

# 创建构建目录
mkdir -p build
cd build

# 配置CMake（启用GPU支持）
/snap/bin/cmake .. \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="60;61;70;75;80;86" \
    -DFAISS_OPT_LEVEL=avx2

# 编译（使用所有可用的CPU核心）
make -j$(nproc)

# 安装
sudo make install

# 4. 更新库路径
echo "4. 更新库路径..."
sudo ldconfig

# 5. 验证安装
echo "5. 验证Faiss安装..."
if [ -f "/usr/local/lib/libfaiss.so" ]; then
    echo "✓ Faiss库已成功安装"
    echo "  - libfaiss.so 位于: /usr/local/lib/"
    ls -la /usr/local/lib/libfaiss*
else
    echo "✗ Faiss安装可能失败，请检查错误信息"
fi

# 6. 设置环境变量
echo "6. 设置环境变量..."
echo "export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:\$PKG_CONFIG_PATH" >> ~/.bashrc

echo "=== 安装完成 ==="
echo "请运行 'source ~/.bashrc' 或重新打开终端来加载环境变量"
echo ""
echo "编译命令示例："
echo "g++ faiss_gpu_ivf.cpp -O2 -o faiss_gpu_ivf -std=c++14 \\"
echo "    -I/usr/local/include \\"
echo "    -L/usr/local/lib \\"
echo "    -lfaiss -lcuda -lcublas \\"
echo "    -lopenblas -lgomp"
echo ""
echo "如果遇到链接问题，可以尝试："
echo "export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
