#include "Convolution.h"

void cpu1DConvolution(const float* input, float* output, const float* kernel,
                      const int32_t inputWidth) {
    for (int32_t x = 0; x < inputWidth; x++) {
        float currSum = 0.f;
        for (int32_t k = 0; k < KERNEL_SIZE; k++) {
            int32_t pos = x - (KERNEL_SIZE / 2) + k;
            if (pos >= 0 && pos < inputWidth) {
                currSum += input[pos] * kernel[k];
            }
        }
        output[x] = currSum;
    }
}

void cpu2DConvolution(const float* input, float* output, const float* kernel,
                      const int32_t inputWidth, const int32_t inputHeight) {
    for (int32_t y = 0; y < inputHeight; y++) {
        for (int32_t x = 0; x < inputWidth; x++) {
            float currSum = 0.f;
            for (int32_t k = 0; k < KERNEL_SIZE; k++) {
                int32_t xPos = x - (KERNEL_SIZE / 2) + k;
                int32_t yPos = y - (KERNEL_SIZE / 2) + k;
                if (xPos >= 0 && xPos < inputWidth && yPos >= 0 && yPos < inputHeight) {
                    currSum += input[y * inputWidth + x] * kernel[k];
                }
            }
            output[y * inputWidth + x] = currSum;
        }
    }
}

bool compareResults(const float* resA, const float* resB, const int32_t width,
                    const int32_t height) {
    printf("Start test for data equality...\n");
    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            if (std::abs(resA[y * width + x] - resB[y * width + x]) >= 0.001) {
                fprintf(stderr, "ResA[%f] != ResB[%f] at row = %d, col = %d\n", resA[y * width + x],
                        resB[y * width + x], y, x);
                fprintf(stderr, "Test - NOT PASSED\n");
                return false;
            }
        }
    }
    fprintf(stderr, "Test - PASSED\n");
    return true;
}