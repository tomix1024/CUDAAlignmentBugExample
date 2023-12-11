#!/bin/bash

CUDA_HOME=/usr/local/cuda

"${CUDA_HOME}/bin/nvcc" -isystem "${CUDA_HOME}/include" -std=c++17 -x cu -c debug.cu -o debug.cu.o
/usr/bin/g++ debug.cu.o -o debug_alignment -lcudart -L"${CUDA_HOME}/targets/x86_64-linux/lib"
./debug_alignment
