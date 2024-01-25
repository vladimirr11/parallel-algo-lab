// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Own includes
#include "common/helpers.h"

// Third party includes
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "common/stbi_image/stbi_image.h"
#include "common/stbi_image/stbi_image_write.h"

/// @brief Blurs a rgb image
__global__ void imageBlurKernel(const unsigned char* inputPixels, unsigned char* outputPixels,
                                const int width, const int height, const int numChannels,
                                const int blurKernelSize) {
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * numChannels;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // blurKernelSize must be odd number
    const int halfBlurKernel = blurKernelSize / 2;
    if (col < width * numChannels && row < height) {
        int currRedPixelVal = 0;
        int currGreenPixelVal = 0;
        int currBluePixelVal = 0;
        int blurKernelPixels = 0;
        const int startBlurRow = row - halfBlurKernel;
        const int endBlurRow = row + halfBlurKernel;
        const int startBlurCol = col - (halfBlurKernel * numChannels);
        const int endBlurCol = col + (halfBlurKernel * numChannels);
        for (int blurRow = startBlurRow; blurRow <= endBlurRow; ++blurRow) {
            for (int blurCol = startBlurCol; blurCol <= endBlurCol; blurCol += 3) {
                if (blurCol >= 0 && blurCol < width * numChannels && blurRow >= 0 &&
                    blurRow < height) {
                    currRedPixelVal += inputPixels[blurRow * width * numChannels + blurCol];
                    currGreenPixelVal += inputPixels[blurRow * width * numChannels + blurCol + 1];
                    currBluePixelVal += inputPixels[blurRow * width * numChannels + blurCol + 2];
                    blurKernelPixels++;
                }
            }
        }
        outputPixels[row * width * numChannels + col] =
            (unsigned char)(currRedPixelVal / (blurKernelPixels));
        outputPixels[row * width * numChannels + col + 1] =
            (unsigned char)(currGreenPixelVal / (blurKernelPixels));
        outputPixels[row * width * numChannels + col + 2] =
            (unsigned char)(currBluePixelVal / (blurKernelPixels));
    }
}

int main() {
    // set max performance CUDA device
    queryAndSetDevice();

    // load image
    int width, height, numChannels;
    const char* imagePath = "data/royal-bengal-tiger-1000x664.jpg";
    unsigned char* inputImage = stbi_load(imagePath, &width, &height, &numChannels, 0);
    if (!inputImage) {
        fprintf(stderr, "stbi_image failed to load image %s\n", imagePath);
        exit(EXIT_FAILURE);
    }

    // image size in bytes
    const int imageBytes = width * height * numChannels;

    // allocate device memory for the input and output image
    unsigned char *devInputImage, *devOutputImage;
    handleCUDAError(cudaMalloc((void**)&devInputImage, imageBytes));
    handleCUDAError(cudaMalloc((void**)&devOutputImage, imageBytes));

    // transfer input image data to the device
    handleCUDAError(cudaMemcpy(devInputImage, inputImage, imageBytes, cudaMemcpyHostToDevice));

    // create event handles
    cudaEvent_t start, stop;
    handleCUDAError(cudaEventCreate(&start));
    handleCUDAError(cudaEventCreate(&stop));

    // kernel execution configuration
    const int threadBlockDim = 32;
    const int gridWidth = ceil((float)(width) / threadBlockDim);
    const int gridHeight = ceil((float)(height) / threadBlockDim);
    dim3 dimGrid(gridWidth, gridHeight, 1);
    dim3 dimBlock(threadBlockDim, threadBlockDim, 1);

    const int blurKernelSize = 9;  // must be odd number
    handleCUDAError(cudaEventRecord(start, 0));
    imageBlurKernel<<<dimGrid, dimBlock>>>(devInputImage, devOutputImage, width, height,
                                           numChannels, blurKernelSize);
    handleCUDAError(cudaEventRecord(stop, 0));

    // wait for kernel to complete
    handleCUDAError(cudaEventSynchronize(stop));

    float elapsedTime;
    handleCUDAError(cudaEventElapsedTime(&elapsedTime, start, stop));
    fprintf(stdout, "Blured image generated for [%.2f] ms\n", elapsedTime);

    // transfer output device image data back to the host
    unsigned char* bluredImageHost = (unsigned char*)malloc(imageBytes);
    handleCUDAError(
        cudaMemcpy(bluredImageHost, devOutputImage, imageBytes, cudaMemcpyDeviceToHost));

    // store blured image on the disk
    stbi_write_jpg("../image_blur/data/blured-tiger.jpg", width, height, numChannels,
                   bluredImageHost, 100);

    // free host and device memory
    handleCUDAError(cudaFree(devInputImage));
    handleCUDAError(cudaFree(devOutputImage));
    free(inputImage);
    free(bluredImageHost);

    return 0;
}
