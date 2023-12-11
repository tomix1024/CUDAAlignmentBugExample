#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>

//#define HARD_MODE_NO_ALIGNAS

//#define FIX_SWAP_MEMBERS
//#define FIX_AVOID_AT_CALL // HARD_MODE_NO_ALIGNAS -> misaligned addresses!
//#define FIX_NO_VALUE_REFERENCE
//#define FIX_NO_ADDITIONAL_MEMBER // HARD_MODE_NO_ALIGNAS -> misaligned addresses!
//#define FIX_ASSIGNMENT_OPERATOR
//#define FIX_STATE_64BIT
//#define FIX_STATE_16BIT
#include "persistentdata.h"


struct LaunchParams
{
    uint32_t *data;
    void **out;
};

extern "C" __global__ void main_kernel(LaunchParams params)
{
    opg::PersistentData pd = opg::PersistentData(params.data);

#ifdef HARD_MODE_NO_ALIGNAS
    struct NEEData
#else
    struct alignas(16) NEEData
#endif
    {
#ifdef FIX_STATE_64BIT
        uint64_t state = 1;
#elif defined(FIX_STATE_16BIT)
        uint16_t state = 1;
#else
        uint32_t state = 1;
#endif
        void *emitter;// = (void *)123llu;
#ifndef FIX_NO_ADDITIONAL_MEMBER
        uint32_t emitter_direction = 3;
#endif
#ifdef FIX_ASSIGNMENT_OPERATOR
        __forceinline__ __device__ NEEData &operator=(const NEEData &other)
        {
            state = other.state;
            emitter = other.emitter;
#ifndef FIX_NO_ADDITIONAL_MEMBER
            emitter_direction = other.emitter_direction;
#endif
        }
#endif
    };
    auto pd_data = opg::PersistentDataEntry<NEEData>(&pd);

    auto &pd_state = pd_data.value().state;
    auto &emitter = pd_data.value().emitter;

    if (pd_state != 10)
    {
        printf("reached exit point\n");
        pd_state = 2;
        return;
    }
#ifndef FIX_NO_VALUE_REFERENCE
    *params.out = &emitter;
#endif
    //optixDirectCall<void, const float &>(0, emitter_direction);
}

int main()
{
    uint32_t data_count = 8;
    uint32_t *data_buffer;
    cudaMalloc(&data_buffer, sizeof(uint32_t) * data_count);
    cudaMemset(data_buffer, -1, sizeof(uint32_t) * data_count);

    void **out_buffer;
    cudaMalloc(&out_buffer, sizeof(void*));

    LaunchParams launch_params;
    launch_params.data = data_buffer;
    launch_params.out = out_buffer;

    main_kernel<<<1, 1>>>(launch_params);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::stringstream ss;
        ss << "CUDA error on synchronize with error '" << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }


    std::vector<uint32_t> data_cpu(data_count);
    cudaMemcpy(data_cpu.data(), data_buffer, sizeof(uint32_t) * data_count, cudaMemcpyDeviceToHost);
    std::cout << "data:";
    for (auto &v : data_cpu)
    {
        std::cout << " 0x" << std::hex << v;
    }
    std::cout << std::endl;
    return 0;
}
