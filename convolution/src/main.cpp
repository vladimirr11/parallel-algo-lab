// Own includes
#include "Convolution.h"

static constexpr int32_t DATA_SIZE_PER_DIM = 1024 * 1024 * 100;

int main() {
    printf("[Exploring convolution algorithms]\n\n");

    // Set device
    printf("Setting device...\n");
    queryAndSetDevice();

    // Allocate memory on the host
    float* inputData = (float*)malloc(DATA_SIZE_PER_DIM * sizeof(float));
    float* outputData1 = (float*)malloc(DATA_SIZE_PER_DIM * sizeof(float));
    float* outputData2 = (float*)malloc(DATA_SIZE_PER_DIM * sizeof(float));
    float* hostKernel = (float*)malloc(KERNEL_SIZE * sizeof(float));

    // Generate random numbers
    for (int32_t i = 0; i < DATA_SIZE_PER_DIM; i++) {
        inputData[i] = rand() / (float)RAND_MAX;
    }

    for (int32_t i = 0; i < KERNEL_SIZE; i++) {
        hostKernel[i] = rand() / (float)RAND_MAX;
    }

    using nanosec = std::chrono::nanoseconds;
    using high_res_clock = std::chrono::high_resolution_clock;

    // 1D convolution on the host
    {
        auto hostStartTime = high_res_clock::now();
        cpu1DConvolution(inputData, outputData1, hostKernel, DATA_SIZE_PER_DIM);
        auto hostEndTime = high_res_clock::now();

        const float cpuTime = cpuDuration<nanosec>(hostStartTime, hostEndTime) / (float)1e6;
        fprintf(stdout,
                "1D convolution of array with %d elements computed for %.2fms on CPU, KERNEL_SIZE "
                "= %d\n",
                DATA_SIZE_PER_DIM, cpuTime, KERNEL_SIZE);
    }

    // 1D convolution on the device
    { set1DGPUConvolution(inputData, outputData2, hostKernel, DATA_SIZE_PER_DIM); }

    // Compare results 1D conv
    { compareResults(outputData1, outputData2, DATA_SIZE_PER_DIM, 1); }

    // Free host memory
    free(inputData);
    free(outputData1);
    free(outputData2);
    free(hostKernel);

    return 0;
}
