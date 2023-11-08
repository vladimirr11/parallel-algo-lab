#pragma once

#include <iostream>

template <typename T>
void checkError(T errorCode, const char* funcName, const char* fileName, const int line) {
    if (errorCode) {
        fprintf(stderr, "CUDA runtime error: %s returned %d error code in %s, line %d", funcName,
                (unsigned int)errorCode, funcName, line);
        exit(EXIT_FAILURE);
    }
}

#define handleCUDAError(func) checkError((func), #func, __FILE__, __LINE__)
