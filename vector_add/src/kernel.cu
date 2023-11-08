// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// C++ system includes
#include <chrono>

// Own includes
#include "common/helpers.h"

float calcKahamSum(const float* resultVec, const int numElements) {
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

void cpuVectorAdd(float* hostA, float* hostB, float* hostC, const int n) {
    for (int i = 0; i < n; i++) {
        hostC[i] = hostA[i] + hostB[i];
    }
}

// Compute vector sum C = A + B
__global__ void vectorAddKernel(const float* A, const float* B, float* C, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void gpuVectorAdd(const float* hostA, const float* hostB, float* hostC, const int n) {
    const int numBytes = n * sizeof(float);
    // allocate memory on the device
    float *devA, *devB, *devC;
    handleCUDAError(cudaMalloc((void**)&devA, numBytes));
    handleCUDAError(cudaMalloc((void**)&devB, numBytes));
    handleCUDAError(cudaMalloc((void**)&devC, numBytes));

    // transfer host data to the device
    handleCUDAError(cudaMemcpy(devA, hostA, numBytes, cudaMemcpyHostToDevice));
    handleCUDAError(cudaMemcpy(devB, hostB, numBytes, cudaMemcpyHostToDevice));

    // create cuda event handles
    cudaEvent_t start, stop;
    handleCUDAError(cudaEventCreate(&start));
    handleCUDAError(cudaEventCreate(&stop));

    // kernel execution config
    const int threadsPerBlock = 256;
    const int numThreadBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    handleCUDAError(cudaEventRecord(start, 0));
    vectorAddKernel<<<numThreadBlocks, threadsPerBlock>>>(devA, devB, devC, n);
    handleCUDAError(cudaEventRecord(stop, 0));

    // thansfer data back to the host
    handleCUDAError(cudaMemcpy(hostC, devC, numBytes, cudaMemcpyDeviceToHost));

    const float devResult = calcKahamSum(hostC, n);
    float gpuTime;
    handleCUDAError(cudaEventElapsedTime(&gpuTime, start, stop));
    fprintf(stdout, "GPU time for vector addition of %d elements: [%f] - result %f\n", n, gpuTime,
            devResult);

    // free device memory
    handleCUDAError(cudaFree(devA));
    handleCUDAError(cudaFree(devB));
    handleCUDAError(cudaFree(devC));
}

int main() {
    using nanosec = std::chrono::nanoseconds;
    using high_res_clock = std::chrono::high_resolution_clock;

    // vector size
    const int numElements = 1'000'000;

    // allocate memmory on the host
    float* hostA = (float*)malloc(numElements * sizeof(float));
    float* hostB = (float*)malloc(numElements * sizeof(float));
    float* hostC = (float*)malloc(numElements * sizeof(float));

    // gen rand numbers
    for (int i = 0; i < numElements; i++) {
        hostA[i] = rand() / (float)RAND_MAX;
        hostB[i] = rand() / (float)RAND_MAX;
    }

    // measure host time
    auto hostStartTime = high_res_clock::now();
    cpuVectorAdd(hostA, hostB, hostC, numElements);
    auto hostEndTime = high_res_clock::now();

    const float hostResult = calcKahamSum(hostC, numElements);
    const float cpuTime =
        std::chrono::duration_cast<nanosec>(hostEndTime - hostStartTime).count() / 1e6;
    fprintf(stdout, "CPU time for vector addition of %d elements: [%f] - result %f\n", numElements,
            cpuTime, hostResult);

    float* hostCC = (float*)malloc(numElements * sizeof(float));
    gpuVectorAdd(hostA, hostB, hostCC, numElements);

    // free host memory
    free(hostA);
    free(hostB);
    free(hostC);
    free(hostCC);

    return 0;
}
