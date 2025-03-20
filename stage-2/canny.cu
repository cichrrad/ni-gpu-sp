// cuda_canny_noopencv.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

//------------------------------------------------------------------------------
// Definitions and Constants
//------------------------------------------------------------------------------

#define BLOCK_SIZE 16
#define KERNEL_SIZE 5

// Define a simple uchar3 structure (similar to CUDA's built-in type)
struct uchar3
{
    unsigned char x, y, z;
};

// Fixed 5x5 Gaussian kernel for sigma ~1.4 stored in constant memory
__constant__ float d_gaussianKernel[KERNEL_SIZE * KERNEL_SIZE] = {
    2, 4, 5, 4, 2,
    4, 9, 12, 9, 4,
    5, 12, 15, 12, 5,
    4, 9, 12, 9, 4,
    2, 4, 5, 4, 2};

//------------------------------------------------------------------------------
// Simple Image I/O Functions (PPM for input, PGM for output)
//------------------------------------------------------------------------------

// Load a binary PPM (P6) image into a vector of uchar3 pixels.
bool loadPPM(const std::string &filename, std::vector<uchar3> &image, int &width, int &height)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    std::string header;
    file >> header;
    if (header != "P6")
    {
        std::cerr << "Not a P6 PPM file." << std::endl;
        return false;
    }

    file >> width >> height;
    int maxVal;
    file >> maxVal;
    file.ignore(256, '\n'); // Skip the newline after header

    image.resize(width * height);
    file.read(reinterpret_cast<char *>(image.data()), width * height * sizeof(uchar3));
    if (!file)
    {
        std::cerr << "Error reading pixel data." << std::endl;
        return false;
    }
    return true;
}

// Save an 8-bit grayscale image (stored in a vector) to a binary PGM (P5) file.
bool savePGM(const std::string &filename, const std::vector<unsigned char> &image, int width, int height)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return false;
    }

    file << "P5\n"
         << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char *>(image.data()), width * height * sizeof(unsigned char));
    return true;
}

//------------------------------------------------------------------------------
// CUDA Kernels
//------------------------------------------------------------------------------

// Kernel 1: Grayscale Conversion
// Input is uchar3 (BGR) and output is a single-channel 8-bit grayscale image.
__global__ void kernelGrayscale(const uchar3 *input, unsigned char *output, int width, int height)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = width * height;
    if (idx < total)
    {
        uchar3 pixel = input[idx]; // Assumes BGR order.
        float gray = 0.299f * pixel.z + 0.587f * pixel.y + 0.114f * pixel.x;
        output[idx] = static_cast<unsigned char>(gray);
    }
}

// Kernel 2: Gaussian Blur
// Using a fixed 5x5 kernel (in constant memory) with border clamping.
__global__ void kernelGaussianBlur(const unsigned char *input, unsigned char *output, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height)
    {
        float sum = 0.0f;
        int half = KERNEL_SIZE / 2;
        for (int dx = -half; dx <= half; dx++)
        {
            int xi = min(max(row + dx, 0), height - 1);
            for (int dy = -half; dy <= half; dy++)
            {
                int yj = min(max(col + dy, 0), width - 1);
                float kVal = d_gaussianKernel[(dx + half) * KERNEL_SIZE + (dy + half)];
                sum += kVal * input[xi * width + yj];
            }
        }
        float kernelSum = 159.0f; // Sum of kernel elements.
        output[row * width + col] = static_cast<unsigned char>(sum / kernelSum);
    }
}

// Kernel 3: Sobel Filter
// Computes gradient magnitude and quantized direction (0, 45, 90, 135).
__global__ void kernelSobel(const unsigned char *input, float *magnitude, float *direction, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= 1 && row < height - 1 && col >= 1 && col < width - 1)
    {
        int idx = row * width + col;
        int GxKernel[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
        int GyKernel[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
        float Gx = 0, Gy = 0;
        for (int dx = -1; dx <= 1; dx++)
        {
            for (int dy = -1; dy <= 1; dy++)
            {
                int neighbor = (row + dx) * width + (col + dy);
                int pixel = static_cast<int>(input[neighbor]);
                Gx += pixel * GxKernel[dx + 1][dy + 1];
                Gy += pixel * GyKernel[dx + 1][dy + 1];
            }
        }
        magnitude[idx] = sqrtf(Gx * Gx + Gy * Gy);
        float angle = atan2f(Gy, Gx) * 180.0f / M_PI;
        if (angle < 0)
            angle += 180.0f;
        if ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f && angle <= 180.0f))
            direction[idx] = 0.0f;
        else if (angle >= 22.5f && angle < 67.5f)
            direction[idx] = 45.0f;
        else if (angle >= 67.5f && angle < 112.5f)
            direction[idx] = 90.0f;
        else
            direction[idx] = 135.0f;
    }
}

// Kernel 4: Non-Maximum Suppression
// Inputs: gradient magnitude and quantized direction (both float arrays)
// Output: thinned edges (float array, 0 for suppressed pixels).
__global__ void kernelNonMaxSuppression(const float *magnitude, const float *direction, float *output, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= 1 && row < height - 1 && col >= 1 && col < width - 1)
    {
        int idx = row * width + col;
        float current = magnitude[idx];
        float d = direction[idx];
        float n1 = 0, n2 = 0;
        if (d == 0.0f)
        { // horizontal: compare left and right.
            n1 = magnitude[row * width + (col - 1)];
            n2 = magnitude[row * width + (col + 1)];
        }
        else if (d == 90.0f)
        { // vertical: compare top and bottom.
            n1 = magnitude[(row - 1) * width + col];
            n2 = magnitude[(row + 1) * width + col];
        }
        else if (d == 45.0f)
        { // diagonal top-left & bottom-right.
            n1 = magnitude[(row - 1) * width + (col - 1)];
            n2 = magnitude[(row + 1) * width + (col + 1)];
        }
        else if (d == 135.0f)
        { // diagonal top-right & bottom-left.
            n1 = magnitude[(row - 1) * width + (col + 1)];
            n2 = magnitude[(row + 1) * width + (col - 1)];
        }
        output[idx] = (current >= n1 && current >= n2) ? current : 0.0f;
    }
}

// Kernel 5: Double Thresholding
// Input: thinned edges (float array).
// Output: 8-bit image with strong (255), weak (128), or non-edge (0) values.
__global__ void kernelDoubleThreshold(const float *input, unsigned char *output, int width, int height, float t_low, float t_high)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = width * height;
    if (idx < total)
    {
        float val = input[idx];
        if (val >= t_high)
            output[idx] = 255;
        else if (val >= t_low)
            output[idx] = 128;
        else
            output[idx] = 0;
    }
}

// Kernel 6: Hysteresis (Edge Tracking)
// For each strong edge pixel, check its 8 neighbors and promote any weak edge (128) to strong (255).
__global__ void kernelHysteresis(unsigned char *image, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= 1 && row < height - 1 && col >= 1 && col < width - 1)
    {
        int idx = row * width + col;
        if (image[idx] == 255)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    int nIdx = (row + dx) * width + (col + dy);
                    if (image[nIdx] == 128)
                        image[nIdx] = 255;
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Host Code (Main)
//------------------------------------------------------------------------------

// A simple PPM loader for binary P6 files.
bool loadPPM(const std::string &filename, std::vector<uchar3> &image, int &width, int &height)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }
    std::string header;
    file >> header;
    if (header != "P6")
    {
        std::cerr << "Not a P6 PPM file." << std::endl;
        return false;
    }
    file >> width >> height;
    int maxVal;
    file >> maxVal;
    file.ignore(256, '\n'); // skip to binary part

    image.resize(width * height);
    file.read(reinterpret_cast<char *>(image.data()), width * height * sizeof(uchar3));
    return true;
}

// A simple PGM writer for binary P5 files (grayscale images)
bool savePGM(const std::string &filename, const std::vector<unsigned char> &image, int width, int height)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return false;
    }
    file << "P5\n"
         << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char *>(image.data()), width * height * sizeof(unsigned char));
    return true;
}

//------------------------------------------------------------------------------
// Main Function (Host)
//------------------------------------------------------------------------------
int main()
{
    // Load input image from a PPM file (P6 format)
    std::string inputFilename = "input.ppm"; // ensure this file exists
    std::vector<uchar3> h_src;
    int width, height;
    if (!loadPPM(inputFilename, h_src, width, height))
    {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }
    int numPixels = width * height;
    std::cout << "Loaded image of size " << width << "x" << height << std::endl;

    // Create host buffers for intermediate stages.
    std::vector<unsigned char> h_gray(numPixels);
    std::vector<unsigned char> h_blur(numPixels);
    std::vector<float> h_gradMag(numPixels);
    std::vector<float> h_gradDir(numPixels);
    std::vector<float> h_nms(numPixels);
    std::vector<unsigned char> h_dt(numPixels);
    std::vector<unsigned char> h_result(numPixels);

    // Allocate device memory.
    uchar3 *d_src;
    unsigned char *d_gray, *d_blur;
    float *d_gradMag, *d_gradDir, *d_nms;
    unsigned char *d_dt;
    cudaMalloc(&d_src, numPixels * sizeof(uchar3));
    cudaMalloc(&d_gray, numPixels * sizeof(unsigned char));
    cudaMalloc(&d_blur, numPixels * sizeof(unsigned char));
    cudaMalloc(&d_gradMag, numPixels * sizeof(float));
    cudaMalloc(&d_gradDir, numPixels * sizeof(float));
    cudaMalloc(&d_nms, numPixels * sizeof(float));
    cudaMalloc(&d_dt, numPixels * sizeof(unsigned char));

    // Copy input image to device.
    cudaMemcpy(d_src, h_src.data(), numPixels * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Kernel 1: Grayscale Conversion (using 1D grid)
    int threadsPerBlock = 256;
    int blocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
    kernelGrayscale<<<blocks, threadsPerBlock>>>(d_src, d_gray, width, height);
    cudaDeviceSynchronize();

    // Kernel 2: Gaussian Blur (using 2D grid)
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernelGaussianBlur<<<gridDim, blockDim>>>(d_gray, d_blur, width, height);
    cudaDeviceSynchronize();

    // Kernel 3: Sobel Filter (using 2D grid)
    kernelSobel<<<gridDim, blockDim>>>(d_blur, d_gradMag, d_gradDir, width, height);
    cudaDeviceSynchronize();

    // Kernel 4: Non-Maximum Suppression (using 2D grid)
    kernelNonMaxSuppression<<<gridDim, blockDim>>>(d_gradMag, d_gradDir, d_nms, width, height);
    cudaDeviceSynchronize();

    // Kernel 5: Double Threshold (using 1D grid)
    float t_low = 20.0f;
    float t_high = 50.0f;
    blocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
    kernelDoubleThreshold<<<blocks, threadsPerBlock>>>(d_nms, d_dt, width, height, t_low, t_high);
    cudaDeviceSynchronize();

    // Kernel 6: Hysteresis (using 2D grid)
    kernelHysteresis<<<gridDim, blockDim>>>(d_dt, width, height);
    cudaDeviceSynchronize();

    // Copy final result from device back to host.
    cudaMemcpy(h_result.data(), d_dt, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save final edge-detected image as a PGM file.
    std::string outputFilename = "cuda_canny_result.pgm";
    if (!savePGM(outputFilename, h_result, width, height))
    {
        std::cerr << "Failed to save result." << std::endl;
    }
    else
    {
        std::cout << "Result saved as " << outputFilename << std::endl;
    }

    // Cleanup device memory.
    cudaFree(d_src);
    cudaFree(d_gray);
    cudaFree(d_blur);
    cudaFree(d_gradMag);
    cudaFree(d_gradDir);
    cudaFree(d_nms);
    cudaFree(d_dt);

    return 0;
}
