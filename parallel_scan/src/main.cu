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
static constexpr int64_t SCAN_ARR_NUM_BYTES = SCAN_ARR_SIZE * sizeof(int32_t);
static constexpr int64_t PARTIAL_SUMS_NUM_BYTES = PARTIAL_SUMS_ARRAY_SIZE * sizeof(int32_t);

__global__ void adjacentBlockSyncScanKernel(const int32_t* input, int32_t* output,
                                            volatile int32_t* blocksPartialSums, int32_t* flagsArr,
                                            int32_t dynCounter) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int32_t sharedData[BLOCK_SIZE];

    //__shared__ int32_t sblId;

    // if (threadIdx.x == 0) {
    //     sblId = atomicAdd(&dynCounter, 1);
    // }
    //__syncthreads();
    // const int32_t blId = sblId;

    const int32_t blId = blockIdx.x;

    // Kogge-Stone scan of thread block
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

    // Update per block local scans
    output[tid] = sharedData[threadIdx.x];

    // Here the kernele is serialized to obtain the end partial sums for each block
    __shared__ int32_t prevSum;
    if (threadIdx.x == 0) {
        // Wait for previous flag
        while (atomicAdd(&flagsArr[blId], 0) == 0)
            ;

        // Read previous partial sum
        prevSum = blocksPartialSums[blId];

        //  Propagate partial sum
        blocksPartialSums[blId + 1] = prevSum + sharedData[blockDim.x - 1];

        // Memory fence - ennsures that the partial sum is completely stored in memory
        __threadfence();

        // Update per block flag
        atomicAdd(&flagsArr[blId + 1], 1);
    }
    __syncthreads();  // Other threads in current block wait here

    // This part is run in parallel between blocks
    if (blId > 0) {
        output[tid] += blocksPartialSums[blId];
    }
}

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

/// @brief Increments per-block prefix sums that will be used in the final kernel
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

/// @brief Final phase kernel of multipass arbitrary length hierachical scan
__global__ void scanLastPhaseKernel(int32_t* result, int32_t* partialSums) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0) {
        result[tid] += partialSums[blockIdx.x - 1];
    }
}

/// @brief Run hierarchical inclusive scan for arbitrary length inputs consisting of 3 separate
/// kernels
inline void runMultiplePassInclusiveScan(const int32_t* devInput, int32_t* devOutput,
                                         int32_t* devPartialSums) {
    // Config partial scan kernel
    const int32_t gridDimPartSumsScan = (PARTIAL_SUMS_ARRAY_SIZE > BLOCK_SIZE)
                                            ? ceil(PARTIAL_SUMS_ARRAY_SIZE / (float)BLOCK_SIZE)
                                            : 1;
    const int32_t blockDimPartSumsScan =
        (PARTIAL_SUMS_ARRAY_SIZE >= BLOCK_SIZE) ? BLOCK_SIZE : PARTIAL_SUMS_ARRAY_SIZE;

    // Kernel launch
    inclusiveScanKernel<<<ceil(SCAN_ARR_SIZE / (float)BLOCK_SIZE), BLOCK_SIZE>>>(
        devInput, devOutput, devPartialSums);
    handleCUDAError(cudaDeviceSynchronize());
    scanPartialSumsKernel<<<gridDimPartSumsScan, blockDimPartSumsScan>>>(
        devPartialSums, std::max(gridDimPartSumsScan, blockDimPartSumsScan));
    handleCUDAError(cudaDeviceSynchronize());
    scanLastPhaseKernel<<<ceil(SCAN_ARR_SIZE / (float)BLOCK_SIZE), BLOCK_SIZE>>>(devOutput,
                                                                                 devPartialSums);
}

/// @brief Run single pass hierarchical inclusive scan for arbitrary length inputs
inline void runSinglePassInclusiveScan(const int32_t* devInput, int32_t* devOutput,
                                       int32_t* devPartialSums, int32_t* devFlagsArr) {
    int32_t devDynCounter = 0;
    adjacentBlockSyncScanKernel<<<ceil(SCAN_ARR_SIZE / (float)BLOCK_SIZE), BLOCK_SIZE>>>(
        devInput, devOutput, devPartialSums, devFlagsArr, devDynCounter);
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
    int32_t* hostOutputData1 = (int32_t*)malloc(SCAN_ARR_NUM_BYTES);

    // Populate input data
    for (int32_t i = 0; i < SCAN_ARR_SIZE; i++) {
        hostInputData[i] = rand() % 10;
    }

    // CPU clock alliases
    using nanosec = std::chrono::nanoseconds;
    using high_res_clock = std::chrono::high_resolution_clock;

    // Allocate device memory
    int32_t *devInputData, *devOutputData, *devPartialSums, *devFlagsArr;
    handleCUDAError(cudaMalloc((void**)&devInputData, SCAN_ARR_NUM_BYTES));
    handleCUDAError(cudaMalloc((void**)&devOutputData, SCAN_ARR_NUM_BYTES));
    handleCUDAError(cudaMalloc((void**)&devPartialSums, PARTIAL_SUMS_NUM_BYTES));
    handleCUDAError(cudaMalloc((void**)&devFlagsArr, PARTIAL_SUMS_NUM_BYTES));

    // Set only the first flag of the array to 1 - to be used in the kernel
    handleCUDAError(cudaMemset(devFlagsArr, 1, sizeof(int32_t)));

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
        // runMultiplePassInclusiveScan(devInputData, devOutputData, devPartialSums);
        runSinglePassInclusiveScan(devInputData, devOutputData, devPartialSums, devFlagsArr);
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
    handleCUDAError(cudaFree(devPartialSums));

    // Free host memory
    free(hostInputData);
    free(hostOutputDataRef);
    free(hostOutputData1);

    return 0;
}
