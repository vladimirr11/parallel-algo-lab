// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <device_functions.h>

// Own includes
#include "common/helpers.h"

constexpr unsigned INPUT_SIZE = 10'000;
constexpr unsigned NUM_BUCKETS = 4;
constexpr unsigned NUM_LETTERS_IN_BUCKET = 6;
constexpr unsigned NUM_THREADBLK = 7;
constexpr unsigned THREADBLK_SIZE = 256;

/// @brief Privatization strategy - data is placed in shared memory to reduce DRAM access latency.
/// Downside is that the shared memory data must be merged into the output histogram
__global__ void computeDeviceHistgramPrivatization(const unsigned* data, unsigned* histogram) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ unsigned sharedHist[NUM_BUCKETS];

    // Init to zero shared memory for each thread block
    for (unsigned i = threadIdx.x; i < NUM_THREADBLK; i += THREADBLK_SIZE) {
        sharedHist[i] = 0u;
    }
    __syncthreads();

    // Compute histogram in shared memory
    for (unsigned i = tid; i < INPUT_SIZE; i += blockDim.x * gridDim.x) {
        if (i < INPUT_SIZE) {
            const unsigned letter = data[i] - 'a';
            const unsigned histBucket = letter / NUM_LETTERS_IN_BUCKET;
            atomicAdd(&sharedHist[histBucket], 1);
        }
    }
    __syncthreads();

    // Transfer shared memory to global
    for (unsigned i = threadIdx.x; i < NUM_THREADBLK; i += THREADBLK_SIZE) {
        atomicAdd(&histogram[i], sharedHist[i]);
    }
}

/// @brief Interleaved partitioning strategy - each thread processes elements that are
/// separated by the elements processed by all threads during one iteration. Threads in a warp
/// access consecutive locations to enable memory coalescing
__global__ void computeDeviceHistgramInterleavedPart(const unsigned* data, unsigned* histogram) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sectionSize = (INPUT_SIZE - 1) / (blockDim.x * gridDim.x) + 1;
    int stride = blockDim.x * gridDim.x;
    for (int k = 0; k < sectionSize; k++) {
        if (tid + (stride * k) < INPUT_SIZE) {
            const unsigned letter = data[tid + (stride * k)] - 'a';
            const unsigned histBucket = letter / NUM_LETTERS_IN_BUCKET;
            atomicAdd(&histogram[histBucket], 1);
        }
    }
}

/// @brief Block partitioning strategy for procesing input data - each thread works on section size
/// consecutive input data elements
__global__ void computeDeviceHistgramBlockPart(const unsigned* data, unsigned* histogram) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sectionSize = (INPUT_SIZE - 1) / (blockDim.x * gridDim.x) + 1;
    int start = tid * sectionSize;
    for (int k = 0; k < sectionSize; k++) {
        if (start + k < INPUT_SIZE) {
            const unsigned letter = data[start + k] - 'a';
            const unsigned histBucket = letter / NUM_LETTERS_IN_BUCKET;
            atomicAdd(&histogram[histBucket], 1);
        }
    }
}

void computeHostHistogram(const unsigned* data, unsigned* histogram) {
    for (int i = 0; i < INPUT_SIZE; i++) {
        const unsigned letter = data[i] - 'a';
        if (letter >= 24) {
            fprintf(stderr, "Unknown letter encountered in host: %c\n", letter);
        }
        const unsigned histBucket = letter / NUM_LETTERS_IN_BUCKET;
        histogram[histBucket]++;
    }
}

void compareResults(const unsigned* hostHist, const unsigned* devHist) {
    bool equalRes = true;
    static int counter = 1;
    for (unsigned i = 0; i < NUM_BUCKETS; i++) {
        if (hostHist[i] != devHist[i]) {
            fprintf(stderr, "hostHist[%d] != devHist[%d]; hostHist[%d] = %d, devHist[%d] = %d\n", i,
                    i, i, hostHist[i], i, devHist[i]);
            exit(EXIT_FAILURE);
        }
    }
    if (equalRes) {
        fprintf(stdout, "Host and Device results match for strategy %d\n", counter);
        counter++;
    }
}

int main() {
    printf("[Histograms]\n\n");
    // Set device
    printf("Setting device...\n");
    queryAndSetDevice();

    // CPU clock alliases
    using nanosec = std::chrono::nanoseconds;
    using high_res_clock = std::chrono::high_resolution_clock;

    // Allocate host memory for input and output data and generate random input
    unsigned* hostData = (unsigned*)malloc(sizeof(unsigned) * INPUT_SIZE);
    unsigned* hostHist = (unsigned*)malloc(sizeof(unsigned) * NUM_BUCKETS);
    unsigned* devResult = (unsigned*)malloc(sizeof(unsigned) * NUM_BUCKETS);
    for (unsigned i = 0; i < INPUT_SIZE; i++) {
        hostData[i] = 'a' + (rand() % ('x' - 'a'));
    }

    // Zero host output storage
    memset(hostHist, 0, sizeof(unsigned) * NUM_BUCKETS);
    memset(devResult, 0, sizeof(unsigned) * NUM_BUCKETS);

    // Allocate device memory and transfer data from host to device
    unsigned *devData, *devHist;
    handleCUDAError(cudaMalloc((void**)&devData, sizeof(unsigned) * INPUT_SIZE));
    handleCUDAError(cudaMalloc((void**)&devHist, sizeof(unsigned) * NUM_BUCKETS));
    handleCUDAError(
        cudaMemcpy(devData, hostData, sizeof(unsigned) * INPUT_SIZE, cudaMemcpyHostToDevice));

    // Measure host execution time
    {
        auto cpuStartTime = high_res_clock::now();
        computeHostHistogram(hostData, hostHist);
        auto cpuEndTime = high_res_clock::now();

        const float cpuTime = cpuDuration<nanosec>(cpuStartTime, cpuEndTime) / (float)1e6;
        printf("CPU histogram run for %.2fms for input data with size %d\n", cpuTime, INPUT_SIZE);
    }

    // Create event handles
    cudaEvent_t start, stop;
    handleCUDAError(cudaEventCreate(&start));
    handleCUDAError(cudaEventCreate(&stop));

    // Measure device time - strategy I
    {
        // Zero device output storage
        handleCUDAError(cudaMemset(devHist, 0, sizeof(unsigned) * NUM_BUCKETS));
        // Run simple histogram
        handleCUDAError(cudaEventRecord(start));
        computeDeviceHistgramBlockPart<<<NUM_THREADBLK, THREADBLK_SIZE>>>(devData, devHist);
        handleCUDAError(cudaEventRecord(stop));

        // Synchronize with kernel execution
        handleCUDAError(cudaEventSynchronize(stop));

        float gpuTime;
        handleCUDAError(cudaEventElapsedTime(&gpuTime, start, stop));
        printf("Blcokc Partitioning GPU histogram run for %.2fms for input data with size %d\n",
               gpuTime, INPUT_SIZE);

        // Trasfer device result to host
        handleCUDAError(
            cudaMemcpy(devResult, devHist, sizeof(unsigned) * NUM_BUCKETS, cudaMemcpyDeviceToHost));

        // Validate results
        compareResults(hostHist, devResult);
    }

    // Measure device time - strategy II
    {
        // Zero device output storage
        handleCUDAError(cudaMemset(devHist, 0, sizeof(unsigned) * NUM_BUCKETS));
        // Run simple histogram
        handleCUDAError(cudaEventRecord(start));
        computeDeviceHistgramInterleavedPart<<<NUM_THREADBLK, THREADBLK_SIZE>>>(devData, devHist);
        handleCUDAError(cudaEventRecord(stop));

        // Synchronize with kernel execution
        handleCUDAError(cudaEventSynchronize(stop));

        float gpuTime;
        handleCUDAError(cudaEventElapsedTime(&gpuTime, start, stop));
        printf(
            "Interleaved partitioning GPU histogram run for %.2fms for input data with size %d\n",
            gpuTime, INPUT_SIZE);

        // Trasfer device result to host
        handleCUDAError(
            cudaMemcpy(devResult, devHist, sizeof(unsigned) * NUM_BUCKETS, cudaMemcpyDeviceToHost));

        // Validate results
        compareResults(hostHist, devResult);
    }

    // Measure device time - strategy III
    {
        // Zero device output storage
        handleCUDAError(cudaMemset(devHist, 0, sizeof(unsigned) * NUM_BUCKETS));
        // Run simple histogram
        handleCUDAError(cudaEventRecord(start));
        computeDeviceHistgramPrivatization<<<NUM_THREADBLK, THREADBLK_SIZE>>>(devData, devHist);
        handleCUDAError(cudaEventRecord(stop));

        // Synchronize with kernel execution
        handleCUDAError(cudaEventSynchronize(stop));

        float gpuTime;
        handleCUDAError(cudaEventElapsedTime(&gpuTime, start, stop));
        printf("Privatization strategy GPU histogram run for %.2fms for input data with size %d\n",
               gpuTime, INPUT_SIZE);

        // Trasfer device result to host
        handleCUDAError(
            cudaMemcpy(devResult, devHist, sizeof(unsigned) * NUM_BUCKETS, cudaMemcpyDeviceToHost));

        // Validate results
        compareResults(hostHist, devResult);
    }

    // Free host memory
    free(hostData);
    free(hostHist);
    free(devResult);

    // Free device memory
    cudaFree(devData);
    cudaFree(devHist);

    return 0;
}
