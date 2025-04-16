#!/bin/bash

# 编译为 ARM64 程序
echo "正在编译为 ARM64 架构..."
aarch64-linux-gnu-g++ -O2 -march=armv8-a -o main main.cc

# 如果编译成功，使用 QEMU 运行
if [ $? -eq 0 ]; then
    echo "编译成功，正在运行..."
    qemu-aarch64 -L /usr/aarch64-linux-gnu/ ./main
else
    echo "编译失败，请检查错误信息。"
fi