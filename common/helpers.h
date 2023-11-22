#pragma once

// C++ system includes
#include <iostream>
#include <chrono>
#include <cassert>

template <typename T>
static inline void checkError(T errorCode, const char* funcName, const char* fileName,
                              const int line) {
    if (errorCode) {
        fprintf(stderr, "CUDA runtime error: %s returned error code %d in %s, line %d", funcName,
                (unsigned int)errorCode, fileName, line);
        exit(EXIT_FAILURE);
    }
}

#define handleCUDAError(func) checkError((func), #func, __FILE__, __LINE__)

template <typename Duration, typename TimePoint>
static inline int64_t cpuDuration(TimePoint startTime, TimePoint endTime) {
    return std::chrono::duration_cast<Duration>(endTime - startTime).count();
}

static inline int smVersion2Cores(const int majorVer, const int minorVer) {
    typedef struct {
        int smVersion;
        int numCores;
        const char* arcName;
    } sm2Cores;

    sm2Cores nvidiaArcVersion2Cores[]{
        {0x30, 192, "Kepler"},      {0x32, 192, "Kepler"},       {0x35, 192, "Kepler"},
        {0x37, 192, "Kepler"},      {0x50, 128, "Maxwell"},      {0x52, 128, "Maxwell"},
        {0x53, 128, "Maxwell"},     {0x60, 64, "Pascal"},        {0x61, 128, "Pascal"},
        {0x62, 128, "Pascal"},      {0x70, 64, "Volta"},         {0x72, 64, "Xavier"},
        {0x75, 64, "Turing"},       {0x80, 64, "Ampere"},        {0x86, 128, "Ampere"},
        {0x87, 128, "Ampere"},      {0x89, 128, "Ada Lovelace"}, {0x90, 128, "Hooper"},
        {-1, -1, "Graphics Device"}};

    int index = 0;
    while (nvidiaArcVersion2Cores[index].smVersion != -1) {
        if (nvidiaArcVersion2Cores[index].smVersion == ((majorVer << 4) + minorVer)) {
            return nvidiaArcVersion2Cores[index].numCores;
        }
        index++;
    }

    fprintf(stderr, "Unknown architecture version %d.%d", majorVer, minorVer);
    return -1;
}

static inline void queryAndSetDevice() {
    int deviceCount = 0;
    handleCUDAError(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No capable CUDA device in the system");
        exit(EXIT_FAILURE);
    }

    int32_t maxComputePerf = -1, maxPerfDevice = -1, cudaCores = -1;
    cudaDeviceProp deviceProps;
    for (int32_t i = 0; i < deviceCount; i++) {
        handleCUDAError(cudaGetDeviceProperties(&deviceProps, i));
        int32_t multiprocessorCount = deviceProps.multiProcessorCount;
        int32_t clockRate = deviceProps.clockRate;
        int32_t majorVer = deviceProps.major;
        int32_t minorVer = deviceProps.minor;
        int32_t numCoresPerSm = smVersion2Cores(majorVer, minorVer);
        assert(numCoresPerSm != -1);
        int64_t computePerf = (int64_t)multiprocessorCount * numCoresPerSm * clockRate;
        if (computePerf > maxComputePerf) {
            maxComputePerf = (int32_t)computePerf;
            maxPerfDevice = i;
            cudaCores = numCoresPerSm;
        }
    }

    assert(maxPerfDevice != -1);

    handleCUDAError(cudaGetDeviceProperties(&deviceProps, maxPerfDevice));
    printf("   --- Set device %d ---\n", maxPerfDevice);
    printf("Name: %s\n", deviceProps.name);
    printf("Compute capability: %d.%d\n", deviceProps.major, deviceProps.minor);
    printf("Clock rate: %.2f MHz\n", (float)deviceProps.clockRate / 1e6);

    printf("   --- Memory Information for device %d ---\n", maxPerfDevice);
    printf("Total global memory: %zu GB\n", deviceProps.totalGlobalMem / (size_t)1e9);
    printf("Total constant memory: %zu KB\n", deviceProps.totalConstMem / (size_t)1e3);
    printf("Max pitch memory: %zu GB\n", deviceProps.memPitch / (size_t)1e9);

    printf("   --- MP Information for device %d ---\n", maxPerfDevice);
    printf("Multiprocessor count: %d\n", deviceProps.multiProcessorCount);
    printf("Cuda cores per MP: %d\n", cudaCores);
    printf("Shared memory per MP: %zu MB\n", deviceProps.sharedMemPerBlock / (size_t)1e3);
    printf("Registers per MP: %d\n", deviceProps.regsPerBlock);
    printf("Max number of thread blocks per MP: %d\n", deviceProps.maxBlocksPerMultiProcessor);
    printf("Threads in warp: %d\n", deviceProps.warpSize);
    printf("Max threads per block: %d\n", deviceProps.maxThreadsPerBlock);
    printf("\n");

    handleCUDAError(cudaSetDevice(maxPerfDevice));
}

template <typename T>
static inline bool compare1DArraysForEquality(const T* benchmark, const T* testArr,
                                              const size_t arrSize) {
    printf("Start test for arrays equality...\n");
    for (size_t i = 0; i < arrSize; i++) {
        if (benchmark[i] != testArr[i]) {
            std::cout << "BenchmarkArray[" << benchmark[i] << "] != TestArray[" << testArr[i]
                      << "] at index " << i << "\n";
            printf("Test - NOT PASSED\n");
            return false;
        }
    }

    printf("Test - PASSED\n");
    return true;
}
