// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <device_functions.h>

// Own includes
#include "common/helpers.h"

static constexpr int32_t SCAN_ARR_SIZE = 1024 * 64;
static constexpr int32_t BLOCK_SIZE = 256;
static constexpr int32_t PARTIAL_SUMS_ARRAY_SIZE = SCAN_ARR_SIZE / (float)BLOCK_SIZE;
static constexpr int64_t SCAN_ARR_NUM_BYTES = SCAN_ARR_SIZE * sizeof(float);

/// @brief Kogge-Stone scan kernel - scans block wide sections from the input
__global__ void inclusiveScanKernel(const int32_t* input, int32_t* output, int32_t* partialSums) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int32_t sharedData[BLOCK_SIZE];

    if (tid < SCAN_ARR_SIZE) {
        sharedData[threadIdx.x] = input[tid];
    }
    for (int32_t stride = 1; stride < blockDim.x; stride <<= 1) {
        int32_t val = 0;
        __syncthreads();
        if (threadIdx.x >= stride) {
            val = sharedData[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            sharedData[threadIdx.x] += val;
        }
    }

    output[tid] = sharedData[threadIdx.x];

    if (threadIdx.x == blockDim.x - 1) {
        partialSums[blockIdx.x] = sharedData[blockDim.x - 1];
    }
}

__global__ void scanPartialSumsKernel(int32_t* partialSums, const int32_t partialSumsSize) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int32_t sharedData[BLOCK_SIZE];

    if (tid < partialSumsSize) {
        sharedData[threadIdx.x] = partialSums[tid];
    }

    for (int32_t stride = 1; stride < blockDim.x; stride <<= 1) {
        int32_t val = 0;
        __syncthreads();
        if (threadIdx.x >= stride) {
            val = sharedData[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            sharedData[threadIdx.x] += val;
        }
    }

    partialSums[tid] = sharedData[threadIdx.x];
}

__global__ void scanLastPhaseKernel(int32_t* result, int32_t* partialSums) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0) {
        result[tid] += partialSums[blockIdx.x - 1];
    }
}

void cpuInclusiveScan(const int32_t* inputArr, int32_t* outputArr, const int32_t inputSize) {
    int32_t accum = inputArr[0];
    outputArr[0] = accum;
    for (int32_t i = 1; i < inputSize; i++) {
        accum += inputArr[i];
        outputArr[i] = accum;
    }
}

int main() {
    printf("[Exploring different parallel prefix sum algorithms]\n\n");

    // Set device
    printf("Setting device...\n");
    queryAndSetDevice();

    // Allocate needed host memory
    int32_t* hostInputData = (int32_t*)malloc(SCAN_ARR_NUM_BYTES);
    int32_t* hostOutputDataRef = (int32_t*)malloc(SCAN_ARR_NUM_BYTES);
    int32_t* hostOutputData1 = (int32_t*)malloc(SCAN_ARR_NUM_BYTES);

    // Populate input data
    for (int32_t i = 0; i < SCAN_ARR_SIZE; i++) {
        hostInputData[i] = rand() % 10;
    }

    // CPU clock alliases
    using nanosec = std::chrono::nanoseconds;
    using high_res_clock = std::chrono::high_resolution_clock;

    // Allocate device memory and transfer data
    int32_t *devInputData, *devOutputData, *devPartialSum;
    handleCUDAError(cudaMalloc((void**)&devInputData, SCAN_ARR_NUM_BYTES));
    handleCUDAError(cudaMalloc((void**)&devOutputData, SCAN_ARR_NUM_BYTES));
    handleCUDAError(cudaMalloc((void**)&devPartialSum, PARTIAL_SUMS_ARRAY_SIZE * sizeof(int32_t)));

    handleCUDAError(
        cudaMemcpy(devInputData, hostInputData, SCAN_ARR_NUM_BYTES, cudaMemcpyHostToDevice));

    // Measure host time
    {
        auto cpuStartTime = high_res_clock::now();
        cpuInclusiveScan(hostInputData, hostOutputDataRef, SCAN_ARR_SIZE);
        auto cpuEndTime = high_res_clock::now();

        const float cpuTime = cpuDuration<nanosec>(cpuStartTime, cpuEndTime) / (float)1e6;
        printf("CPU prefix sum scan algorithm run for %.2fms for array with size %ld\n", cpuTime,
               SCAN_ARR_SIZE);
    }

    // Measure device time
    {
        // Create event handles and record time
        cudaEvent_t start, stop;
        handleCUDAError(cudaEventCreate(&start));
        handleCUDAError(cudaEventCreate(&stop));

        // Config partial scan kernel
        const int32_t gridDimPartSumsScan = (PARTIAL_SUMS_ARRAY_SIZE > BLOCK_SIZE)
                                                ? ceil(PARTIAL_SUMS_ARRAY_SIZE / (float)BLOCK_SIZE)
                                                : 1;
        const int32_t blockDimPartSumsScan =
            (PARTIAL_SUMS_ARRAY_SIZE >= BLOCK_SIZE) ? BLOCK_SIZE : PARTIAL_SUMS_ARRAY_SIZE;

        handleCUDAError(cudaEventRecord(start));
        // Kernel launch
        inclusiveScanKernel<<<ceil(SCAN_ARR_SIZE / (float)BLOCK_SIZE), BLOCK_SIZE>>>(
            devInputData, devOutputData, devPartialSum);
        handleCUDAError(cudaDeviceSynchronize());
        scanPartialSumsKernel<<<gridDimPartSumsScan, blockDimPartSumsScan>>>(
            devPartialSum, std::max(gridDimPartSumsScan, blockDimPartSumsScan));
        handleCUDAError(cudaDeviceSynchronize());
        scanLastPhaseKernel<<<ceil(SCAN_ARR_SIZE / (float)BLOCK_SIZE), BLOCK_SIZE>>>(devOutputData,
                                                                                     devPartialSum);
        handleCUDAError(cudaEventRecord(stop));

        // Synchronize with kernel execution
        handleCUDAError(cudaEventSynchronize(stop));

        float gpuTime;
        handleCUDAError(cudaEventElapsedTime(&gpuTime, start, stop));
        printf("GPU prefix sum scan algorithm run for %.2fms for array with size %ld\n", gpuTime,
               SCAN_ARR_SIZE);

        // Trasfer device result ot host
        handleCUDAError(
            cudaMemcpy(hostOutputData1, devOutputData, SCAN_ARR_NUM_BYTES, cudaMemcpyDeviceToHost));

        // Compare host and device results
        compare1DArraysForEquality(hostOutputDataRef, hostOutputData1, SCAN_ARR_SIZE);
    }

    // Free device memory
    handleCUDAError(cudaFree(devInputData));
    handleCUDAError(cudaFree(devOutputData));
    handleCUDAError(cudaFree(devPartialSum));

    // Free host memory
    free(hostInputData);
    free(hostOutputDataRef);
    free(hostOutputData1);

    return 0;
}
