// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Own includes
#include "common/helpers.h"

__global__ void basicMatrixMulKernel(const float* M, const float* N, float* P, const int32_t rowsM,
                                     const int32_t colsM, const int32_t colsN) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsM && col < colsN) {
        float dotProd = 0.f;
        // each thread computes one element of the block sub-matrix
        for (int32_t k = 0; k < colsM; k++) {
            dotProd += M[row * colsM + k] * N[k * colsN + col];
        }

        P[row * colsN + col] = dotProd;
    }
}

void matrixMulGPU(const float* hostM, const float* hostN, float* hostP, const dim3 dimHostM,
                  const dim3 dimHostN) {
    const int numBytesDevM = dimHostM.x * dimHostM.y * sizeof(float);  // bytes for M matrix
    const int numBytesDevN = dimHostN.x * dimHostN.y * sizeof(float);  // bytes for N matrix
    const int numBytesDevP = dimHostM.x * dimHostN.y * sizeof(float);  // bytes for P matrix

    // allocate memory on the device
    float *devM, *devN, *devP;
    handleCUDAError(cudaMalloc((void**)&devM, numBytesDevM));
    handleCUDAError(cudaMalloc((void**)&devN, numBytesDevN));
    handleCUDAError(cudaMalloc((void**)&devP, numBytesDevP));

    // transfer host data to the device
    handleCUDAError(cudaMemcpy(devM, hostM, numBytesDevM, cudaMemcpyHostToDevice));
    handleCUDAError(cudaMemcpy(devN, hostN, numBytesDevN, cudaMemcpyHostToDevice));

    // create cuda event handles
    cudaEvent_t start, stop;
    handleCUDAError(cudaEventCreate(&start));
    handleCUDAError(cudaEventCreate(&stop));

    // kernel execution config
    dim3 dimBlock(16, 16);
    dim3 dimGrid(ceil((float)dimHostM.x / 16), ceil((float)dimHostN.y / 16));

    fprintf(stdout, "Start GPU matrix multiplication...\n");

    // run and measure kernel execution time
    handleCUDAError(cudaEventRecord(start, 0));
    basicMatrixMulKernel<<<dimGrid, dimBlock>>>(devM, devN, devP, dimHostM.x, dimHostM.y,
                                                dimHostN.y);
    handleCUDAError(cudaEventRecord(stop, 0));

    // thansfer data back to the host
    handleCUDAError(cudaMemcpy(hostP, devP, numBytesDevP, cudaMemcpyDeviceToHost));

    // synchronize with kernel execution
    handleCUDAError(cudaEventSynchronize(stop));

    float gpuTime;
    handleCUDAError(cudaEventElapsedTime(&gpuTime, start, stop));
    fprintf(stdout, "M[%d, %d] x N[%d, %d] multiplied for %.2f miliseconds\n", dimHostM.x,
            dimHostM.y, dimHostN.x, dimHostN.y, gpuTime);

    // transfer result matrix to host
    handleCUDAError(cudaMemcpy(hostP, devP, numBytesDevP, cudaMemcpyDeviceToHost));

    // free device memory
    handleCUDAError(cudaFree(devM));
    handleCUDAError(cudaFree(devN));
    handleCUDAError(cudaFree(devP));
}

/// @brief Naive matrix multiplication on CPU
void matrixMulCPUBasicBenchmark(const float* M, const float* N, float* P, const int32_t rowsM,
                                const int32_t colsM, const int32_t rowsN, const int32_t colsN) {
    for (int32_t i = 0; i < rowsM; i++) {
        for (int32_t j = 0; j < colsN; j++) {
            for (int32_t k = 0; k < colsM; k++) {
                P[i * colsN + j] += M[i * colsM + k] * N[k * colsN + j];
            }
        }
    }
}

/// @brief Naive matrix multiplication on CPU with reordered indices for better
/// spacial locality of matrix N
void matrixMulCPUReorderedIndx(const float* M, const float* N, float* P, const int32_t rowsM,
                               const int32_t colsM, const int32_t rowsN, const int32_t colsN) {
    for (int32_t i = 0; i < rowsM; i++) {
        for (int32_t k = 0; k < rowsN; k++) {
            for (int32_t j = 0; j < colsN; j++) {
                P[i * colsN + j] += M[i * colsM + k] * N[k * colsN + j];
            }
        }
    }
}

bool checkMatrixEqual(const float* P1, const float* P2, const int32_t width, const int32_t height) {
    for (int32_t i = 0; i < height; i++) {
        for (int32_t j = 0; j < width; j++) {
            if (P1[i * width + j] != P2[i * width + j]) {
                fprintf(stdout, "P1[%f] != P2[%f] at row = %d, col = %d\n", P1[i * width + j],
                        P2[i * width + j], i, j);
                return false;
            }
        }
    }
    return true;
}

int main() {
    printf("[Basic and Tiled Matrix Multiply Using CUDA]\n\n");

    queryAndSetDevice();

    // setup matrix dimensions - outer dimensions must be equal
    dim3 dimMatrixM(600, 400, 1);
    dim3 dimMatrixN(400, 600, 1);

    // allocate memory on the host
    float* hostM = (float*)malloc(dimMatrixM.x * dimMatrixM.y * sizeof(float));
    float* hostN = (float*)malloc(dimMatrixN.x * dimMatrixN.y * sizeof(float));
    float* hostP1 = (float*)malloc(dimMatrixM.x * dimMatrixN.y * sizeof(float));
    float* hostP2 = (float*)malloc(dimMatrixM.x * dimMatrixN.y * sizeof(float));
    float* hostP3 = (float*)malloc(dimMatrixM.x * dimMatrixN.y * sizeof(float));

    // gen rand numbers
    const int32_t matrixSize = dimMatrixM.x * dimMatrixM.y;
    for (int32_t i = 0; i < matrixSize; i++) {
        hostM[i] = rand() / (float)RAND_MAX;
        hostN[i] = rand() / (float)RAND_MAX;
    }

    // zero initialize result storage
    memset(hostP1, 0, dimMatrixM.x * dimMatrixN.y * sizeof(float));
    memset(hostP2, 0, dimMatrixM.x * dimMatrixN.y * sizeof(float));
    memset(hostP3, 0, dimMatrixM.x * dimMatrixN.y * sizeof(float));

    using nanosec = std::chrono::nanoseconds;
    using high_res_clock = std::chrono::high_resolution_clock;

    fprintf(stdout, "Start CPU matrix multiplication...\n");

    // measure host time
    {
        auto hostStartTime = high_res_clock::now();
        matrixMulCPUBasicBenchmark(hostM, hostN, hostP1, dimMatrixM.x, dimMatrixM.y, dimMatrixN.x,
                                   dimMatrixN.y);
        auto hostEndTime = high_res_clock::now();

        const float cpuTime = cpuDuration<nanosec>(hostStartTime, hostEndTime) / 1e6;
        fprintf(stdout, "M[%d, %d] x N[%d, %d] multiplied for %.2f miliseconds...", dimMatrixM.x,
                dimMatrixM.y, dimMatrixN.x, dimMatrixN.y, cpuTime);

        auto hostStartTime2 = high_res_clock::now();
        matrixMulCPUReorderedIndx(hostM, hostN, hostP2, dimMatrixM.x, dimMatrixM.y, dimMatrixN.x,
                                  dimMatrixN.y);
        auto hostEndTime2 = high_res_clock::now();

        const float cpuTime2 = cpuDuration<nanosec>(hostStartTime2, hostEndTime2) / 1e6;
        fprintf(stdout, " with reordered indices for %.2f miliseconds\n", cpuTime2);

        // check if result matrices are equal
        if (checkMatrixEqual(hostP1, hostP2, dimMatrixM.x, dimMatrixN.y)) {
            printf("Check for CPU-CPU Matrix equality - PASSED\n\n");
        } else {
            printf("Check for CPU-CPU Matrix equality - NOT PASSED\n\n");
        }
    }

    // execute on the device
    {
        matrixMulGPU(hostM, hostN, hostP3, dimMatrixM, dimMatrixN);

        // check if CPU and GPU result matrices are equal
        if (checkMatrixEqual(hostP1, hostP3, dimMatrixM.x, dimMatrixN.y)) {
            printf("Check for CPU-GPU Matrix equality - PASSED\n\n");
        } else {
            printf("Check for CPU-GPU Matrix equality - NOT PASSED\n\n");
        }
    }

    // deallocate host memory
    free(hostM);
    free(hostN);
    free(hostP1);
    free(hostP2);
    free(hostP3);

    return 0;
}
