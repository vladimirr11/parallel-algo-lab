#include <device_launch_parameters.h>
#include "Convolution.h"

__global__ void gpu1DConvolutionKernel(const float* devInput, float* devOutput, const float* kernel,
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

    // Transfer data to device
    handleCUDAError(cudaMemcpy(devInput, hostInput, numBytes, cudaMemcpyHostToDevice));
    handleCUDAError(
        cudaMemcpy(devKernel, hostKernel, KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Create event handles
    cudaEvent_t start, stop;
    handleCUDAError(cudaEventCreate(&start));
    handleCUDAError(cudaEventCreate(&stop));

    // Start timer and launch kernel
    const int32_t dimx = 16;
    handleCUDAError(cudaEventRecord(start));
    gpu1DConvolutionKernel<<<ceil(inputWidth / (float)dimx), dimx>>>(devInput, devOutput, devKernel,
                                                                     inputWidth);
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
    handleCUDAError(cudaFree(devKernel));
}
