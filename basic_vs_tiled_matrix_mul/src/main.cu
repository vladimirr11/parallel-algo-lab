// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifndef __CUDACC__  // for __syncthreads()
#define __CUDACC__
#endif
#include <device_functions.h>

// C++ system includes
#include <omp.h>
#include <thread>
#include <algorithm>

// Own includes
#include "common/helpers.h"

static constexpr int32_t BLOCK_SIZE = 16;  // Thread block size in each dim

/// @brief Tiled matrix multiplivcation P = M * N
__global__ void tiledMatrixMulKernel(const float* M, const float* N, float* P, const int32_t rowsM,
                                     const int32_t colsM, const int32_t rowsN,
                                     const int32_t colsN) {
    // Storage for sub-matrices in the shared memory for each block in the grid
    __shared__ float sM[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sN[BLOCK_SIZE][BLOCK_SIZE];

    // Aliases for thread and block indices - stored in registers
    const int32_t bx = blockIdx.x;
    const int32_t by = blockIdx.y;
    const int32_t tx = threadIdx.x;
    const int32_t ty = threadIdx.y;

    // Global row and col - from the grid
    const int32_t row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int32_t col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float dotProd = 0.f;
    const int32_t numPhases = ceil(colsM / (float)BLOCK_SIZE);
    for (int32_t phase = 0; phase < numPhases; phase++) {
        // Load the sub-matrices from device memory to shared memory
        if (row < rowsM && (phase * BLOCK_SIZE + tx) < colsM) {
            sM[ty][tx] = M[row * colsM + (phase * BLOCK_SIZE + tx)];
        }
        if ((phase * BLOCK_SIZE + ty) < rowsN && col < colsN) {
            sN[ty][tx] = N[(phase * BLOCK_SIZE + ty) * colsN + col];
        }
        __syncthreads();
        // Multiply the two sub-matrices together
        if (phase == numPhases - 1) {
            // Handle the last phase of dot product computation in case colsM is not multiple of
            // BLOCK_SIZE
            for (int32_t k = 0; k < colsM - (phase * BLOCK_SIZE); k++) {
                dotProd += sM[ty][k] * sN[k][tx];
            }
        } else {
#pragma unroll
            for (int32_t k = 0; k < BLOCK_SIZE; k++) {
                dotProd += sM[ty][k] * sN[k][tx];
            }
        }
        __syncthreads();
    }
    // Write the block sub-matrix to device memory
    if (row < rowsM && col < colsN) {
        P[row * colsN + col] = dotProd;
    }
}

/// @brief Basic matrix multiplivcation P = M * N
__global__ void basicMatrixMulKernel(const float* M, const float* N, float* P, const int32_t rowsM,
                                     const int32_t colsM, const int32_t colsN) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsM && col < colsN) {
        float dotProd = 0.f;
        // Each thread computes one element of the block sub-matrix
        for (int32_t k = 0; k < colsM; k++) {
            dotProd += M[row * colsM + k] * N[k * colsN + col];
        }

        P[row * colsN + col] = dotProd;
    }
}

/// @brief Set and launch kernel for matrix multiplicatin on the device
static void matrixMulGPU(const float* hostM, const float* hostN, float* hostP1, float* hostP2,
                         const dim3 dimHostM, const dim3 dimHostN) {
    const int numBytesDevM = dimHostM.y * dimHostM.x * sizeof(float);  // bytes for M matrix
    const int numBytesDevN = dimHostN.y * dimHostN.x * sizeof(float);  // bytes for N matrix
    const int numBytesDevP = dimHostM.y * dimHostN.x * sizeof(float);  // bytes for P matrix

    // Allocate memory on the device
    float *devM, *devN, *devP1, *devP2;
    handleCUDAError(cudaMalloc((void**)&devM, numBytesDevM));
    handleCUDAError(cudaMalloc((void**)&devN, numBytesDevN));
    handleCUDAError(cudaMalloc((void**)&devP1, numBytesDevP));
    handleCUDAError(cudaMalloc((void**)&devP2, numBytesDevP));

    // Transfer host data to the device
    handleCUDAError(cudaMemcpy(devM, hostM, numBytesDevM, cudaMemcpyHostToDevice));
    handleCUDAError(cudaMemcpy(devN, hostN, numBytesDevN, cudaMemcpyHostToDevice));

    // Create cuda event handles
    cudaEvent_t start, stop;
    handleCUDAError(cudaEventCreate(&start));
    handleCUDAError(cudaEventCreate(&stop));

    // Kernel execution config
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil((float)dimHostN.x / BLOCK_SIZE), ceil((float)dimHostM.y / BLOCK_SIZE));

    fprintf(stdout, "GPU matrix multiplication\n\n");

    // Basic matrix multiplication
    // -------------------------------------------------------------------------------
    fprintf(stdout, "Start basic matrix multiplication ...\n");

    handleCUDAError(cudaEventRecord(start, 0));
    basicMatrixMulKernel<<<dimGrid, dimBlock>>>(devM, devN, devP1, dimHostM.y, dimHostM.x,
                                                dimHostN.x);
    handleCUDAError(cudaEventRecord(stop, 0));

    // Synchronize with kernel execution
    handleCUDAError(cudaEventSynchronize(stop));

    float gpuTime;
    handleCUDAError(cudaEventElapsedTime(&gpuTime, start, stop));
    fprintf(stdout, "M[%d, %d] x N[%d, %d] matrices multiplied for %.2f miliseconds\n", dimHostM.y,
            dimHostM.x, dimHostN.y, dimHostN.x, gpuTime);

    // Thansfer data back to the host
    handleCUDAError(cudaMemcpy(hostP1, devP1, numBytesDevP, cudaMemcpyDeviceToHost));

    // Tiled matrix mutiplication
    // ----------------------------------------------------------------------------------
    fprintf(stdout, "Start tiled matrix multiplication ...\n");

    handleCUDAError(cudaEventRecord(start, 0));
    tiledMatrixMulKernel<<<dimGrid, dimBlock>>>(devM, devN, devP2, dimHostM.y, dimHostM.x,
                                                dimHostN.y, dimHostN.x);
    handleCUDAError(cudaEventRecord(stop, 0));
    handleCUDAError(cudaEventSynchronize(stop));

    gpuTime = 0.f;
    handleCUDAError(cudaEventElapsedTime(&gpuTime, start, stop));
    fprintf(stdout, "M[%d, %d] x N[%d, %d] matrices multiplied for %.2f miliseconds\n", dimHostM.y,
            dimHostM.x, dimHostN.y, dimHostN.x, gpuTime);

    handleCUDAError(cudaMemcpy(hostP2, devP2, numBytesDevP, cudaMemcpyDeviceToHost));

    // Free device memory
    handleCUDAError(cudaFree(devM));
    handleCUDAError(cudaFree(devN));
    handleCUDAError(cudaFree(devP1));
    handleCUDAError(cudaFree(devP2));
}

/// @brief Naive matrix multiplication on CPU. Indices order has poor spatial locality
/// for access of matrix N. Benchmark for sluggishness.
static void matrixMulCPUBasicBenchmark(const float* M, const float* N, float* P,
                                       const int32_t rowsM, const int32_t colsM,
                                       const int32_t colsN) {
    for (int32_t i = 0; i < rowsM; i++) {
        for (int32_t j = 0; j < colsN; j++) {
            for (int32_t k = 0; k < colsM; k++) {
                P[i * colsN + j] += M[i * colsM + k] * N[k * colsN + j];
            }
        }
    }
}

/// @brief Matrix multiplication on CPU with reordered indices for better spacial locality.
static void matrixMulCPUReorderedIndx(const float* M, const float* N, float* P, const int32_t rowsM,
                                      const int32_t colsM, const int32_t colsN) {
    for (int32_t i = 0; i < rowsM; i++) {
        for (int32_t k = 0; k < colsM; k++) {
            for (int32_t j = 0; j < colsN; j++) {
                P[i * colsN + j] += M[i * colsM + k] * N[k * colsN + j];
            }
        }
    }
}

/// @brief Matrix multiplication on CPU executed on multiple threads with OpenMP
static void matrixMulCPUParallelFor(const float* M, const float* N, float* P, const int32_t rowsM,
                                    const int32_t colsM, const int32_t colsN,
                                    const unsigned numThreads) {
    omp_set_dynamic(0);                                       // Explicitly disable dynamic teams
#pragma omp parallel for num_threads(numThreads) collapse(1)  // Parallelize outer loop only
    for (int32_t i = 0; i < rowsM; i++) {
        for (int32_t k = 0; k < colsM; k++) {
            for (int32_t j = 0; j < colsN; j++) {
                P[i * colsN + j] += M[i * colsM + k] * N[k * colsN + j];
            }
        }
    }
}

/// @brief Check if storage P1 == P2
static bool checkMatrixEqual(const float* P1, const float* P2, const int32_t width,
                             const int32_t height) {
    for (int32_t r = 0; r < height; r++) {
        for (int32_t c = 0; c < width; c++) {
            if (P1[r * width + c] != P2[r * width + c]) {
                fprintf(stdout, "P1[%f] != P2[%f] at row = %d, col = %d\n", P1[r * width + c],
                        P2[r * width + c], r, c);
                return false;
            }
        }
    }
    return true;
}

int main() {
    printf("[Basic and Tiled Matrix Multiply on CPU and GPU with simple banchmark tests]\n\n");

    // Set device
    printf("Setting device...\n");
    queryAndSetDevice();

    // Setup matrix dimensions - outer dimensions must be equal
    dim3 dimMatrixM(900, 1000);
    dim3 dimMatrixN(1000, 1100);

    // Allocate needed memory on the host
    float* hostM = (float*)malloc(dimMatrixM.y * dimMatrixM.x * sizeof(float));
    float* hostN = (float*)malloc(dimMatrixN.y * dimMatrixN.x * sizeof(float));
    float* hostP1 = (float*)malloc(dimMatrixM.y * dimMatrixN.x * sizeof(float));
    float* hostP2 = (float*)malloc(dimMatrixM.y * dimMatrixN.x * sizeof(float));
    float* hostP3 = (float*)malloc(dimMatrixM.y * dimMatrixN.x * sizeof(float));
    float* hostP4 = (float*)malloc(dimMatrixM.y * dimMatrixN.x * sizeof(float));
    float* hostP5 = (float*)malloc(dimMatrixM.y * dimMatrixN.x * sizeof(float));

    // Generate random numbers
    const int32_t matrixSize = dimMatrixM.y * dimMatrixM.x;
    for (int32_t i = 0; i < matrixSize; i++) {
        hostM[i] = rand() / (float)RAND_MAX;
        hostN[i] = rand() / (float)RAND_MAX;
    }

    // Zero initialize result storage
    memset(hostP1, 0, dimMatrixM.y * dimMatrixN.x * sizeof(float));
    memset(hostP2, 0, dimMatrixM.y * dimMatrixN.x * sizeof(float));
    memset(hostP3, 0, dimMatrixM.y * dimMatrixN.x * sizeof(float));
    memset(hostP4, 0, dimMatrixM.y * dimMatrixN.x * sizeof(float));
    memset(hostP5, 0, dimMatrixM.y * dimMatrixN.x * sizeof(float));

    using nanosec = std::chrono::nanoseconds;
    using high_res_clock = std::chrono::high_resolution_clock;

    fprintf(stdout, "CPU matrix multiplication\n");

    // Execute on the host single-threaded
    {
        auto hostStartTime = high_res_clock::now();
        matrixMulCPUBasicBenchmark(hostM, hostN, hostP1, dimMatrixM.y, dimMatrixM.x, dimMatrixN.x);
        auto hostEndTime = high_res_clock::now();

        const float cpuTime = cpuDuration<nanosec>(hostStartTime, hostEndTime) / 1e6;
        fprintf(stdout,
                "M[%d, %d] x N[%d, %d] matrices multiplied on 1 thread for %.2f miliseconds...\n",
                dimMatrixM.y, dimMatrixM.x, dimMatrixN.y, dimMatrixN.x, cpuTime);

        auto hostStartTime2 = high_res_clock::now();
        matrixMulCPUReorderedIndx(hostM, hostN, hostP2, dimMatrixM.y, dimMatrixM.x, dimMatrixN.x);
        auto hostEndTime2 = high_res_clock::now();

        const float cpuTime2 = cpuDuration<nanosec>(hostStartTime2, hostEndTime2) / 1e6;
        fprintf(stdout, "Same matrices with reordered indices multiplied for %.2f miliseconds\n",
                cpuTime2);

        // Check if result matrices are equal
        if (checkMatrixEqual(hostP1, hostP2, dimMatrixN.x, dimMatrixM.y)) {
            printf("Check for CPU-CPU Matrix equality - PASSED\n\n");
        } else {
            printf("Check for CPU-CPU Matrix equality - NOT PASSED\n\n");
        }
    }

    // Execute on the host on multiple threads
    {
        // Basic matrix multiplication
        unsigned int numThreads = std::thread::hardware_concurrency();
        auto hostStartTime = high_res_clock::now();
        matrixMulCPUParallelFor(hostM, hostN, hostP3, dimMatrixM.y, dimMatrixM.x, dimMatrixN.x,
                                numThreads - 1);
        auto hostEndTime = high_res_clock::now();

        const float cpuTime = cpuDuration<nanosec>(hostStartTime, hostEndTime) / 1e6;
        fprintf(stdout,
                "Matrices M[%d, %d] x N[%d, %d] multiplied on %d threads for %.2f miliseconds...\n",
                dimMatrixM.y, dimMatrixM.x, dimMatrixN.y, dimMatrixN.x, numThreads - 1, cpuTime);

        if (checkMatrixEqual(hostP1, hostP3, dimMatrixN.x, dimMatrixM.y)) {
            printf("Check for CPU-CPU Matrix equality - PASSED\n\n");
        } else {
            printf("Check for CPU-CPU Matrix equality - NOT PASSED\n\n");
        }
    }

    // Execute on the device
    {
        matrixMulGPU(hostM, hostN, hostP4, hostP5, dimMatrixM, dimMatrixN);

        if (checkMatrixEqual(hostP1, hostP4, dimMatrixN.x, dimMatrixM.y)) {
            printf("Check for CPU-GPU Matrix equality - PASSED\n");
        } else {
            printf("Check for CPU-GPU Matrix equality - NOT PASSED\n");
        }

        if (checkMatrixEqual(hostP1, hostP5, dimMatrixN.x, dimMatrixM.y)) {
            printf("Check for CPU-GPU Matrix equality - PASSED\n");
        } else {
            printf("Check for CPU-GPU Matrix equality - NOT PASSED\n");
        }
    }

    // Deallocate host memory
    free(hostM);
    free(hostN);
    free(hostP1);
    free(hostP2);
    free(hostP3);
    free(hostP4);
    free(hostP5);

    return 0;
}
