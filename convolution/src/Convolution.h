#pragma once

// CUDA runtime
#include <cuda_runtime.h>

// C++ system includes
#include <cstdint>

// Own includes
#include "common/helpers.h"

static constexpr int32_t KERNEL_SIZE = 7;

// CPU functions
extern void __host__ cpu1DConvolution(const float* input, float* output, const float* kernel,
                                      const int32_t inputWidth);

extern void __host__ cpu2DConvolution(const float* input, float* output, const float* kernel,
                                      const int32_t inputWidth, const int32_t inputHeight);

extern bool __host__ compareResults(const float* resA, const float* resB, const int32_t width,
                                    const int32_t height);

// GPU functions
extern void __host__ set1DGPUConvolution(const float* hostInput, float* hostOutput,
                                         const float* hostKernel, const int32_t inputWidth);

extern void __global__ gpu1DConvolutionKernel(const float* devInput, float* devOutput,
                                              const float* kernel, const int32_t inputWidth);

// extern void gpu2DConvolutionKernel(const float* input, float* output, const float* kernel,
//                                    const int32_t inputWidth, const int32_t inputHeight);
