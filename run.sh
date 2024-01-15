#!/bin/bash

CUDA_HOME=/usr/local/cuda

function run_experiment {
    echo ""
    echo "Run with args: $@"
    "${CUDA_HOME}/bin/nvcc" -isystem "${CUDA_HOME}/include" -std=c++17 -x cu -c debug.cu -o debug.cu.o $@
    /usr/bin/g++ debug.cu.o -o debug_alignment -lcudart -L"${CUDA_HOME}/targets/x86_64-linux/lib"
    ./debug_alignment
}

run_experiment

run_experiment -DFIX_SWAP_MEMBERS
run_experiment -DFIX_AVOID_AT_CALL
run_experiment -DFIX_NO_VALUE_REFERENCE
run_experiment -DFIX_NO_ADDITIONAL_MEMBER
run_experiment -DFIX_ASSIGNMENT_OPERATOR
run_experiment -DFIX_STATE_64BIT
run_experiment -DFIX_STATE_16BIT

run_experiment -DHARD_MODE_NO_ALIGNAS -DFIX_SWAP_MEMBERS
run_experiment -DHARD_MODE_NO_ALIGNAS -DFIX_AVOID_AT_CALL
run_experiment -DHARD_MODE_NO_ALIGNAS -DFIX_NO_VALUE_REFERENCE
run_experiment -DHARD_MODE_NO_ALIGNAS -DFIX_NO_ADDITIONAL_MEMBER
run_experiment -DHARD_MODE_NO_ALIGNAS -DFIX_ASSIGNMENT_OPERATOR
run_experiment -DHARD_MODE_NO_ALIGNAS -DFIX_STATE_64BIT
run_experiment -DHARD_MODE_NO_ALIGNAS -DFIX_STATE_16BIT
