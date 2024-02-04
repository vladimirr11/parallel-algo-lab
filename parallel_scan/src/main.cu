// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <device_functions.h>

// Own includes
#include "common/helpers.h"

static constexpr int32_t BLOCK_SIZE = 256;
static constexpr int32_t SCAN_ARR_SIZE = 1024 * BLOCK_SIZE;
static constexpr int32_t PARTIAL_SUMS_ARRAY_SIZE = SCAN_ARR_SIZE / BLOCK_SIZE;
static constexpr int64_t SCAN_ARR_NUM_BYTES = SCAN_ARR_SIZE * sizeof(int32_t);
static constexpr int64_t PARTIAL_SUMS_NUM_BYTES = PARTIAL_SUMS_ARRAY_SIZE * sizeof(int32_t);

/// @brief Scans block wide sections from the input
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
        partialSums[blockIdx.x] = sharedData[threadIdx.x];
    }
}

/// @brief Increments per-block prefix sums that will be used in the final kernel
__global__ void scanPartialSumsKernel(int32_t* partialSums, const int32_t partialSumsSize) {
    const int32_t tid = threadIdx.x;

    extern __shared__ int32_t sharedData[];

    if (tid < partialSumsSize) {
        sharedData[tid] = partialSums[tid];
    }

    for (int32_t stride = 1; stride < partialSumsSize; stride <<= 1) {
        int32_t val = 0;
        __syncthreads();
        if (tid >= stride) {
            val = sharedData[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            sharedData[tid] += val;
        }
    }

    if (tid < partialSumsSize) {
        partialSums[tid] = sharedData[tid];
    }
}

/// @brief Final phase kernel of multipass arbitrary length hierachical scan
__global__ void scanLastPhaseKernel(int32_t* output, int32_t* partialSums) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t bid = blockIdx.x;
    if (bid > 0) {
        output[tid] += partialSums[bid - 1];
    }
}

/// @brief Run hierarchical inclusive scan
inline void runMultiplePassInclusiveScan(const int32_t* devInput, int32_t* devOutput,
                                         int32_t* devPartialSums) {
    // Kernels launch
    inclusiveScanKernel<<<ceil((float)SCAN_ARR_SIZE / (float)BLOCK_SIZE), BLOCK_SIZE>>>(
        devInput, devOutput, devPartialSums);

    scanPartialSumsKernel<<<1, PARTIAL_SUMS_ARRAY_SIZE, PARTIAL_SUMS_NUM_BYTES>>>(
        devPartialSums, PARTIAL_SUMS_ARRAY_SIZE);

    scanLastPhaseKernel<<<ceil((float)SCAN_ARR_SIZE / (float)BLOCK_SIZE), BLOCK_SIZE>>>(
        devOutput, devPartialSums);
}

/// @brief CPU inclusive scan - used as banchmark
inline void cpuInclusiveScan(const int32_t* inputArr, int32_t* outputArr, const int32_t inputSize) {
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
    int32_t* hostOutputData = (int32_t*)malloc(SCAN_ARR_NUM_BYTES);

    // Populate input data
    for (int32_t i = 0; i < SCAN_ARR_SIZE; i++) {
        hostInputData[i] = rand() % 10;
    }

    // CPU clock aliases
    using nanosec = std::chrono::nanoseconds;
    using high_res_clock = std::chrono::high_resolution_clock;

    // Allocate device memory
    int32_t *devInputData, *devOutputData, *devPartialSums;
    handleCUDAError(cudaMalloc((void**)&devInputData, SCAN_ARR_NUM_BYTES));
    handleCUDAError(cudaMalloc((void**)&devOutputData, SCAN_ARR_NUM_BYTES));
    handleCUDAError(cudaMalloc((void**)&devPartialSums, PARTIAL_SUMS_NUM_BYTES));

    // Transfer input data to the device
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
        // Create event handles
        cudaEvent_t start, stop;
        handleCUDAError(cudaEventCreate(&start));
        handleCUDAError(cudaEventCreate(&stop));

        // Run multiple pass hierarchical scan
        handleCUDAError(cudaEventRecord(start));
        runMultiplePassInclusiveScan(devInputData, devOutputData, devPartialSums);
        handleCUDAError(cudaEventRecord(stop));

        // Synchronize with kernel execution
        handleCUDAError(cudaEventSynchronize(stop));

        float gpuTime;
        handleCUDAError(cudaEventElapsedTime(&gpuTime, start, stop));
        printf("GPU prefix sum scan algorithm run for %.2fms for array with size %ld\n", gpuTime,
               SCAN_ARR_SIZE);

        // Trasfer device result ot host
        handleCUDAError(
            cudaMemcpy(hostOutputData, devOutputData, SCAN_ARR_NUM_BYTES, cudaMemcpyDeviceToHost));

        // Compare host and device results
        compare1DArraysForEquality(hostOutputDataRef, hostOutputData, SCAN_ARR_SIZE);
    }

    // Free device memory
    handleCUDAError(cudaFree(devInputData));
    handleCUDAError(cudaFree(devOutputData));
    handleCUDAError(cudaFree(devPartialSums));

    // Free host memory
    free(hostInputData);
    free(hostOutputDataRef);
    free(hostOutputData);

    return 0;
}
