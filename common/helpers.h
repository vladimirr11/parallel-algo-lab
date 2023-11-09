#pragma once

#include <iostream>
#include <chrono>

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
