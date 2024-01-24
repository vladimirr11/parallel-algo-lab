#include <device_launch_parameters.h>
#ifndef __CUDACC__  // for __syncthreads()
#define __CUDACC__
#endif
#include <device_functions.h>
#include "Convolution.h"

// Storage for the convolution kernel
__constant__ float kernel[KERNEL_SIZE];

__global__ void tiled1DConvolutionKernel(const float* input, float* output,
                                         const int32_t inputWidth) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sharedData[TILE_SIZE_1D + KERNEL_SIZE - 1];

    const int32_t halfKernSize = KERNEL_SIZE / 2;

    // Load left halo cells in per block shared memory
    const int32_t leftHaloCellIdx = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x >= (blockDim.x - halfKernSize)) {
        sharedData[threadIdx.x - (blockDim.x - halfKernSize)] =
            leftHaloCellIdx < 0 ? 0 : input[leftHaloCellIdx];
    }

    // Load inner cells in shared memory
    sharedData[threadIdx.x + halfKernSize] = input[tid];

    // Load right halo cells
    const int32_t rightHaloCellIdx = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x < halfKernSize) {
        sharedData[threadIdx.x + blockDim.x + halfKernSize] =
            rightHaloCellIdx >= inputWidth ? 0 : input[rightHaloCellIdx];
    }

    __syncthreads();

    float currSum = 0.f;
#pragma unroll
    for (int32_t i = 0; i < KERNEL_SIZE; i++) {
        currSum += sharedData[threadIdx.x + i] * kernel[i];
    }

    output[tid] = currSum;
}

__global__ void gpu1DConvolutionKernel(const float* devInput, float* devOutput,
                                       const int32_t inputWidth) {
    const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < inputWidth) {
        float currSum = 0.f;
        for (int32_t k = 0; k < KERNEL_SIZE; k++) {
            int32_t pos = x - (KERNEL_SIZE / 2) + k;
            if (pos >= 0 && pos < inputWidth) {
                currSum += devInput[pos] * kernel[k];
            }
        }
        devOutput[x] = currSum;
    }
}

void set1DGPUConvolution(const float* hostInput, float* hostOutput, const float* hostKernel,
                         const int32_t inputWidth) {
    const int32_t numBytes = inputWidth * sizeof(float);

    // Allocate memory on the device
    float *devInput, *devOutput, *devKernel;
    handleCUDAError(cudaMalloc((void**)&devInput, numBytes));
    handleCUDAError(cudaMalloc((void**)&devOutput, numBytes));
    handleCUDAError(cudaMalloc((void**)&devKernel, KERNEL_SIZE * sizeof(float)));

    // Transfer data to global memory of the device
    handleCUDAError(cudaMemcpy(devInput, hostInput, numBytes, cudaMemcpyHostToDevice));

    // Transfer data to the constant memory of the device
    handleCUDAError(cudaMemcpyToSymbol(kernel, hostKernel, KERNEL_SIZE * sizeof(float)));

    // Create event handles
    cudaEvent_t start, stop;
    handleCUDAError(cudaEventCreate(&start));
    handleCUDAError(cudaEventCreate(&stop));

    // Start timer and launch kernel
    const int32_t dimx = 256;
    handleCUDAError(cudaEventRecord(start));
    tiled1DConvolutionKernel<<<ceil(inputWidth / (float)dimx), dimx>>>(devInput, devOutput,
                                                                       inputWidth);
    // gpu1DConvolutionKernel<<<ceil(inputWidth / (float)dimx), dimx>>>(devInput, devOutput,
    //                                                                  inputWidth);
    handleCUDAError(cudaEventRecord(stop));

    // Synchronize with kernel execution
    handleCUDAError(cudaEventSynchronize(stop));

    float gpuTime;
    handleCUDAError(cudaEventElapsedTime(&gpuTime, start, stop));
    fprintf(
        stdout,
        "1D convolution of array with %d elements computed for %.2fms on GPU, KERNEL_SIZE = %d\n",
        inputWidth, gpuTime, KERNEL_SIZE);

    // Transfer result back to host
    handleCUDAError(cudaMemcpy(hostOutput, devOutput, numBytes, cudaMemcpyDeviceToHost));

    // Deallocate memory
    handleCUDAError(cudaFree(devInput));
    handleCUDAError(cudaFree(devOutput));
}
