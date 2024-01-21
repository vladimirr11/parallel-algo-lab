// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <device_functions.h>
// C++ system includes
#include <fstream>
#include <vector>
#include <string>

// Own includes
#include "common/helpers.h"

/// @brief Simple kernel that computes A*X = 0; It has 2 distinctive drowbacks - 1) lack of memeory
/// coalesced access and 2) potential warp divergence
__global__ void computeSpMVMulOnDevice(const int32_t* data, const int32_t* colIndices,
                                       const int32_t* indptr, const int32_t numRows,
                                       const int32_t* X, int64_t* result) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < numRows) {
        int start = indptr[row];
        int end = indptr[row + 1];
        int dot = 0;
        for (int el = start; el < end; el++) {
            dot += data[el] * X[colIndices[el]];
        }
        result[row] = dot;
    }
}

/// @brief Sparse Matrix-Vector Multiplication from the form A*X = 0 where A is in CSR format
/// @param data - keep all non-zero elements in the SpM
/// @param colIndices - store column indices of the non-zero data elements
/// @param indptr - every two consecutive elements store start and end indices of the current row
/// @param numRws - number of rows in the SpM
/// @param X - dense vector
void computeSpMVMulOnHost(const int32_t* data, const int32_t* colIndices, const int32_t* indptr,
                          const int32_t numRows, const int32_t* X, int64_t* result) {
    for (int32_t row = 0; row < numRows - 1; row++) {
        int32_t dot = 0;
        int32_t rowSatrt = indptr[row];
        int32_t rowEnd = indptr[row + 1];
        for (int32_t el = rowSatrt; el < rowEnd; el++) {
            dot += data[el] * X[colIndices[el]];
        }
        result[row] = dot;
    }
}

/// @brief Compare host and device results
void compareResults(const int64_t* hostRes, const int64_t* devRes, const int32_t matDim) {
    static uint32_t counter = 1;
    for (uint32_t i = 0; i < matDim; i++) {
        if (hostRes[i] != devRes[i]) {
            fprintf(stderr, "hostRes[%d] != devRes[%d]; hostRes[%d] = %d, devRes[%d] = %d\n", i, i,
                    i, hostRes[i], i, devRes[i]);
            // exit(EXIT_FAILURE);
        }
    }
    fprintf(stdout, "Host and Device results match for SpMV mul strategy %d\n", counter);
    counter++;
}

/// @brief Read a CSR format component input file
std::vector<int32_t> readCSRCompFromFile(const std::string& fileName) {
    std::ifstream inputFile(fileName.data(), std::ios::in | std::ios::binary);
    if (!inputFile.good()) {
        std::cout << "Couldn't read file: " << fileName.data() << std::endl;
        exit(EXIT_FAILURE);
    }
    char _;
    inputFile >> _;
    std::vector<int32_t> buffer{std::istream_iterator<int32_t>{inputFile}, {}};
    if (buffer.empty()) {
        std::cout << "Couldn't extract matrix componenets from file: " << fileName.data()
                  << std::endl;
    }
    return buffer;
}

int main() {
    printf("[Sparse Matrix computation]\n\nSetting device...\n");
    queryAndSetDevice();  // Set device

    // CPU clock alliases
    using nanosec = std::chrono::nanoseconds;
    using high_res_clock = std::chrono::high_resolution_clock;

    // Read CSR format data
    std::vector<int32_t> csrData = readCSRCompFromFile("data/csr_matrix_data.txt");
    std::vector<int32_t> csrIndices = readCSRCompFromFile("data/csr_matrix_indices.txt");
    std::vector<int32_t> csrIndptr = readCSRCompFromFile("data/csr_matrix_indptr.txt");
    const int32_t numRows = (int32_t)csrIndptr.size();
    const int32_t dataSize = (int32_t)csrData.size();

    // Allocate storage on the host for both dense and result vectors
    int32_t* hostDenseVector = (int32_t*)malloc(numRows * sizeof(int32_t));
    int64_t* hostResult = (int64_t*)malloc(numRows * sizeof(int64_t));
    int64_t* devResult = (int64_t*)malloc(numRows * sizeof(int64_t));

    // Zero memory for dense and result vectors
    memset(hostDenseVector, 0, numRows * sizeof(int32_t));
    memset(hostResult, 0, numRows * sizeof(int64_t));
    memset(devResult, 0, numRows * sizeof(int64_t));

    // Generate random dense vector
    for (int32_t i = 0; i < numRows; i++) {
        hostDenseVector[i] = rand() % 1'000;
    }

    // Measure host execution time
    {
        auto cpuStartTime = high_res_clock::now();
        computeSpMVMulOnHost(csrData.data(), csrIndices.data(), csrIndptr.data(), numRows,
                             hostDenseVector, hostResult);
        auto cpuEndTime = high_res_clock::now();

        const float cpuTime = cpuDuration<nanosec>(cpuStartTime, cpuEndTime) / (float)1e6;
        printf("CPU SpMV multiply run for [%.2f]ms for dense input data with size %d\n", cpuTime,
               (int32_t)csrData.size());
    }

    // Allocate device memory CSR format components, dense, and result vector
    int32_t *devCSRData, *devCSRIndices, *devCSRIndptr, *devDenseVector;
    int64_t* devResultVector;
    handleCUDAError(cudaMalloc((void**)&devCSRData, dataSize * sizeof(int32_t)));
    handleCUDAError(cudaMalloc((void**)&devCSRIndices, dataSize * sizeof(int32_t)));
    handleCUDAError(cudaMalloc((void**)&devCSRIndptr, numRows * sizeof(int32_t)));
    handleCUDAError(cudaMalloc((void**)&devDenseVector, numRows * sizeof(int32_t)));
    handleCUDAError(cudaMalloc((void**)&devResultVector, numRows * sizeof(int64_t)));

    // Transfer memory from host to device
    handleCUDAError(
        cudaMemcpy(devCSRData, csrData.data(), dataSize * sizeof(int32_t), cudaMemcpyHostToDevice));
    handleCUDAError(cudaMemcpy(devCSRIndices, csrIndices.data(), dataSize * sizeof(int32_t),
                               cudaMemcpyHostToDevice));
    handleCUDAError(cudaMemcpy(devCSRIndptr, csrIndptr.data(), numRows * sizeof(int32_t),
                               cudaMemcpyHostToDevice));
    handleCUDAError(cudaMemcpy(devDenseVector, hostDenseVector, numRows * sizeof(int32_t),
                               cudaMemcpyHostToDevice));

    // Zero device result vector
    handleCUDAError(cudaMemset(devResultVector, 0, numRows * sizeof(int64_t)));

    // Create start and stop cudaEvenets
    cudaEvent_t start, stop;
    handleCUDAError(cudaEventCreate(&start));
    handleCUDAError(cudaEventCreate(&stop));
    // Measure device executuin time
    {
        int32_t numThreadBlks = numRows / 256 + 1;
        // Run simple histogram
        handleCUDAError(cudaEventRecord(start));
        computeSpMVMulOnDevice<<<numThreadBlks, 256>>>(devCSRData, devCSRIndices, devCSRIndptr,
                                                       numRows, devDenseVector, devResultVector);
        handleCUDAError(cudaEventRecord(stop));

        // Synchronize with kernel execution
        handleCUDAError(cudaEventSynchronize(stop));

        float gpuTime;
        handleCUDAError(cudaEventElapsedTime(&gpuTime, start, stop));
        printf("GPU SpMV multiply run for [%.2f]ms for dense input data with size %d\n", gpuTime,
               dataSize);

        // Trasfer device result to host
        handleCUDAError(cudaMemcpy(devResult, devResultVector, sizeof(int64_t) * numRows,
                                   cudaMemcpyDeviceToHost));

        // Validate results
        compareResults(hostResult, devResult, numRows);
    }

    // Free device memeory
    handleCUDAError(cudaFree(devCSRData));
    handleCUDAError(cudaFree(devCSRIndices));
    handleCUDAError(cudaFree(devCSRIndptr));
    handleCUDAError(cudaFree(devDenseVector));
    handleCUDAError(cudaFree(devResultVector));

    // Free host memeory
    free(hostDenseVector);
    free(hostResult);

    return 0;
}
