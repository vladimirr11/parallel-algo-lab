#pragma once

#include <iostream>
#include <chrono>
#include <cassert>

template <typename T>
void checkError(T errorCode, const char* funcName, const char* fileName, const int line) {
    if (errorCode) {
        fprintf(stderr, "CUDA runtime error: %s returned error code %d in %s, line %d", funcName,
                (unsigned int)errorCode, fileName, line);
        exit(EXIT_FAILURE);
    }
}

#define handleCUDAError(func) checkError((func), #func, __FILE__, __LINE__)

template <typename Duration, typename TimePoint>
int64_t cpuDuration(TimePoint startTime, TimePoint endTime) {
    return std::chrono::duration_cast<Duration>(endTime - startTime).count();
}

inline int smVersion2Cores(const int majorVer, const int minorVer) {
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

    fprintf(stderr, "Unknown architecture vesion %d.%d", majorVer, minorVer);
    return -1;
}

void queryAndSetDevice() {
    int deviceCount = 0;
    handleCUDAError(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No capable CUDA device in the system");
        exit(EXIT_FAILURE);
    }

    int maxComputePerf = -1, maxPerfDevice = -1;
    cudaDeviceProp deviceProps;
    for (int i = 0; i < deviceCount; i++) {
        handleCUDAError(cudaGetDeviceProperties(&deviceProps, i));
        int multiprocessorCount = deviceProps.multiProcessorCount;
        int clockRate = deviceProps.clockRate;
        int majorVer = deviceProps.major;
        int minorVer = deviceProps.minor;
        int numCoresPerSm = smVersion2Cores(majorVer, minorVer);
        assert(numCoresPerSm != -1);
        int64_t computePerf = (int64_t)multiprocessorCount * numCoresPerSm * clockRate;
        if (computePerf > maxComputePerf) {
            maxComputePerf = computePerf;
            maxPerfDevice = i;
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
    printf("Total constant memory: %ld KB\n", deviceProps.totalConstMem / (size_t)1e3);
    printf("Max pitch memory: %zu GB\n", deviceProps.memPitch/ (size_t)1e9);

    printf("   --- MP Information for device %d ---\n", maxPerfDevice);
    printf("Multiprocessor count: %d\n", deviceProps.multiProcessorCount);
    printf("Shared memory per SM: %ld Bytes\n", deviceProps.sharedMemPerBlock);
    printf("Registers per mp: %d\n", deviceProps.regsPerBlock);
    printf("Threads in warp: %d\n", deviceProps.warpSize);
    printf("Max threads per block: %d\n", deviceProps.maxThreadsPerBlock);
    printf("Max grid dimensions: (%d, %d, %d)\n", deviceProps.maxGridSize[0],
           deviceProps.maxGridSize[1], deviceProps.maxGridSize[2]);
    printf("\n");

    handleCUDAError(cudaSetDevice(maxPerfDevice));
}
