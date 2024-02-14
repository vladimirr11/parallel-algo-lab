// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Own includes
#include "common/helpers.h"

constexpr int numElements = 1'000'000;

inline float calcKahamSum(const float* resultVec) {
    float sum = 0.0f;
    float err = 0.0f;
    for (int i = 0; i < numElements; i++) {
        float currVal = resultVec[i] - err;
        float currSum = sum + currVal;
        err = (currSum - sum) - currVal;
        sum = currSum;
    }
    return sum;
}

inline void cpuVectorAdd(float* hostA, float* hostB, float* hostC) {
    for (int i = 0; i < numElements; i++) {
        hostC[i] = hostA[i] + hostB[i];
    }
}

// Compute vector sum C = A + B
__global__ void vectorAddKernel(const float* A, const float* B, float* C, const int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        C[tid] = A[tid] + B[tid];
    }
}

void launchGPUVectorAdd(const float* hostA, const float* hostB) {
    // alloc peagable host memory for the result
    const int numBytes = numElements * sizeof(float);
    float* hostC = (float*)malloc(numBytes);

    // create cuda event handles to measure end-to-end and kernel time
    cudaEvent_t start, stop, kernelStart, kernelStop;
    handleCUDAError(cudaEventCreate(&start));
    handleCUDAError(cudaEventCreate(&stop));
    handleCUDAError(cudaEventCreate(&kernelStart));
    handleCUDAError(cudaEventCreate(&kernelStop));

    // start to measure end-to-end time execution
    handleCUDAError(cudaEventRecord(start, 0));

    // allocate memory on the device
    float *devA, *devB, *devC;
    handleCUDAError(cudaMalloc((void**)&devA, numBytes));
    handleCUDAError(cudaMalloc((void**)&devB, numBytes));
    handleCUDAError(cudaMalloc((void**)&devC, numBytes));

    // transfer host data to the device
    handleCUDAError(cudaMemcpy(devA, hostA, numBytes, cudaMemcpyHostToDevice));
    handleCUDAError(cudaMemcpy(devB, hostB, numBytes, cudaMemcpyHostToDevice));

    // kernel execution config
    const int blockSize = 256;
    const int gridSize = (numElements + blockSize - 1) / blockSize;

    // run the kernele and measure execution time
    handleCUDAError(cudaEventRecord(kernelStart, 0));
    vectorAddKernel<<<gridSize, blockSize>>>(devA, devB, devC, numElements);
    handleCUDAError(cudaEventRecord(kernelStop, 0));

    // synchronize with kernel execution
    handleCUDAError(cudaEventSynchronize(kernelStop));

    // thansfer data back to the host
    handleCUDAError(cudaMemcpy(hostC, devC, numBytes, cudaMemcpyDeviceToHost));

    // stop to measure end-to-end time
    handleCUDAError(cudaEventRecord(stop, 0));

    // synchronize with data stransfer
    handleCUDAError(cudaEventSynchronize(stop));

    float kernelTime, endToEndTime;
    handleCUDAError(cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop));
    handleCUDAError(cudaEventElapsedTime(&endToEndTime, start, stop));

    const float devResult = calcKahamSum(hostC);
    fprintf(stdout,
            "GPU time for vector addition of %d elements: [%.4f] - result %f, end-to-end time: "
            "[%.4f]; executed on default stream\n",
            numElements, kernelTime, devResult, endToEndTime);

    // destroy events
    handleCUDAError(cudaEventDestroy(kernelStart));
    handleCUDAError(cudaEventDestroy(kernelStop));
    handleCUDAError(cudaEventDestroy(start));
    handleCUDAError(cudaEventDestroy(stop));

    // free device memory
    handleCUDAError(cudaFree(devA));
    handleCUDAError(cudaFree(devB));
    handleCUDAError(cudaFree(devC));
}

void launchGPUVectorAddAsync(const float* hostA, const float* hostB) {
    // alloc pinned host memory for the result
    const int numBytes = numElements * sizeof(float);
    float* hostC;
    handleCUDAError(cudaMallocHost((void**)&hostC, numBytes));

    // allocate memory on the device
    float *devA, *devB, *devC;
    handleCUDAError(cudaMalloc((void**)&devA, numBytes));
    handleCUDAError(cudaMalloc((void**)&devB, numBytes));
    handleCUDAError(cudaMalloc((void**)&devC, numBytes));

    // create streams
    constexpr int numStreams = 10;
    constexpr int streamSize = numElements / numStreams;
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; i++) {
        handleCUDAError(cudaStreamCreate(&streams[i]));
    }

    // create cuda event handles to measure end-to-end and kernel time
    cudaEvent_t start, stop, kernelStart, kernelStop;
    handleCUDAError(cudaEventCreate(&start));
    handleCUDAError(cudaEventCreate(&stop));
    handleCUDAError(cudaEventCreate(&kernelStart));
    handleCUDAError(cudaEventCreate(&kernelStop));

    // kernel execution config
    constexpr int blockSize = 256;
    constexpr int gridSize = (numElements + blockSize - 1) / blockSize;

    // start to measure end-to-end time execution
    handleCUDAError(cudaEventRecord(start, 0));

    // thansfer data to the device asynchronously
    for (int i = 0; i < numStreams; i++) {
        int offset = streamSize * i;
        handleCUDAError(cudaMemcpyAsync(&devA[offset], &hostA[offset], streamSize * sizeof(float),
                                        cudaMemcpyHostToDevice, streams[i]));
        handleCUDAError(cudaMemcpyAsync(&devB[offset], &hostB[offset], streamSize * sizeof(float),
                                        cudaMemcpyHostToDevice, streams[i]));
    }

    // start to measure kernel time
    handleCUDAError(cudaEventRecord(kernelStart, 0));
    for (int i = 0; i < numStreams; i++) {
        int offset = streamSize * i;
        vectorAddKernel<<<(gridSize + numStreams - 1) / numStreams, blockSize, 0, streams[i]>>>(
            &devA[offset], &devB[offset], &devC[offset], streamSize);
    }
    // stop to measure kernel time
    handleCUDAError(cudaEventRecord(kernelStop, 0));

    // thansfer data back to the host
    for (int i = 0; i < numStreams; i++) {
        int offset = streamSize * i;
        handleCUDAError(cudaMemcpyAsync(&hostC[offset], &devC[offset], streamSize * sizeof(float),
                                        cudaMemcpyDeviceToHost, streams[i]));
    }

    // stop to measure end-to-end time
    handleCUDAError(cudaEventRecord(stop, 0));

    // synchronize with data transfer
    handleCUDAError(cudaEventSynchronize(stop));

    float kernelTime, endToEndTime;
    handleCUDAError(cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop));
    handleCUDAError(cudaEventElapsedTime(&endToEndTime, start, stop));

    const float devResult = calcKahamSum(hostC);
    fprintf(stdout,
            "GPU time for vector addition of %d elements: [%.4f] - result %f, end-to-end time: "
            "[%.4f]; executed on %d streams\n",
            numElements, kernelTime, devResult, endToEndTime, numStreams);

    // destroy events
    handleCUDAError(cudaEventDestroy(kernelStart));
    handleCUDAError(cudaEventDestroy(kernelStop));
    handleCUDAError(cudaEventDestroy(start));
    handleCUDAError(cudaEventDestroy(stop));

    // destroy streams
    for (int i = 0; i < numStreams; i++) {
        handleCUDAError(cudaStreamDestroy(streams[i]));
    }

    // free device memory
    handleCUDAError(cudaFree(devA));
    handleCUDAError(cudaFree(devB));
    handleCUDAError(cudaFree(devC));

    // free pinned host memory
    handleCUDAError(cudaFreeHost(hostC));
}

int main() {
    using nanosec = std::chrono::nanoseconds;
    using high_res_clock = std::chrono::high_resolution_clock;

    // allocate memmory on the host
    float* hostA = (float*)malloc(numElements * sizeof(float));
    float* hostB = (float*)malloc(numElements * sizeof(float));
    float* hostC = (float*)malloc(numElements * sizeof(float));

    // generate random numbers
    for (int i = 0; i < numElements; i++) {
        hostA[i] = rand() % numElements;
        hostB[i] = rand() % numElements;
    }

    // measure host time
    {
        auto hostStartTime = high_res_clock::now();
        cpuVectorAdd(hostA, hostB, hostC);
        auto hostEndTime = high_res_clock::now();

        const float hostResult = calcKahamSum(hostC);
        const float cpuTime = cpuDuration<nanosec>(hostStartTime, hostEndTime) / 1e6;
        fprintf(stdout, "CPU time for vector addition of %d elements: [%.4f] - result %f\n",
                numElements, cpuTime, hostResult);
    }

    // run on defalut stream and measure time
    launchGPUVectorAdd(hostA, hostB);

    // run async on non-default streams and measure time
    launchGPUVectorAddAsync(hostA, hostB);

    // free host memory
    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
