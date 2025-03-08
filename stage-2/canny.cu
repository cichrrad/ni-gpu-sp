// cuda_canny.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

// Block size for 2D kernels
#define BLOCK_SIZE 16
#define KERNEL_SIZE 5

// Fixed 5x5 Gaussian kernel for sigma ~1.4 stored in constant memory
__constant__ float d_gaussianKernel[KERNEL_SIZE * KERNEL_SIZE] = {
    2, 4, 5, 4, 2,
    4, 9, 12, 9, 4,
    5, 12, 15, 12, 5,
    4, 9, 12, 9, 4,
    2, 4, 5, 4, 2};

// ===========================
// Kernel 1: Grayscale Conversion
// ===========================
// Input is uchar3 (BGR) and output is a single-channel 8-bit grayscale image.
__global__ void kernelGrayscale(const uchar3 *input, unsigned char *output, int width, int height)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = width * height;
    if (idx < total)
    {
        uchar3 pixel = input[idx]; // OpenCV stores color as BGR
        // Gray = 0.299 * R + 0.587 * G + 0.114 * B
        float gray = 0.299f * pixel.z + 0.587f * pixel.y + 0.114f * pixel.x;
        output[idx] = static_cast<unsigned char>(gray);
    }
}

// ===========================
// Kernel 2: Gaussian Blur
// ===========================
// Using a fixed 5x5 kernel with border clamping.
// Input and output are 1D arrays (row-major order).
__global__ void kernelGaussianBlur(const unsigned char *input, unsigned char *output, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        float sum = 0.0f;
        int half = KERNEL_SIZE / 2;
        // Loop over kernel window
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
        float kernelSum = 159.0f; // Sum of kernel elements
        output[row * width + col] = static_cast<unsigned char>(sum / kernelSum);
    }
}

// ===========================
// Kernel 3: Sobel Filter
// ===========================
// Computes gradient magnitude and quantized direction (0, 45, 90, 135)
// Input is grayscale; outputs are float arrays (magnitude, direction)
__global__ void kernelSobel(const unsigned char *input, float *magnitude, float *direction, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Avoid border pixels
    if (row >= 1 && row < height - 1 && col >= 1 && col < width - 1)
    {
        int idx = row * width + col;
        // Sobel kernels (hardcoded)
        int GxKernel[3][3] = {{1, 0, -1},
                              {2, 0, -2},
                              {1, 0, -1}};
        int GyKernel[3][3] = {{1, 2, 1},
                              {0, 0, 0},
                              {-1, -2, -1}};
        float Gx = 0, Gy = 0;
        // Convolve over 3x3 neighborhood
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
        // Quantize the angle
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

// ===========================
// Kernel 4: Non-Maximum Suppression
// ===========================
// Input: gradient magnitude and quantized direction (both as float arrays)
// Output: thinned edges in a float array (0 for suppressed pixels)
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
        // Based on quantized direction, compare with two neighbors
        if (d == 0.0f)
        { // horizontal edge; compare left/right
            n1 = magnitude[row * width + (col - 1)];
            n2 = magnitude[row * width + (col + 1)];
        }
        else if (d == 90.0f)
        { // vertical edge; compare top/bottom
            n1 = magnitude[(row - 1) * width + col];
            n2 = magnitude[(row + 1) * width + col];
        }
        else if (d == 45.0f)
        { // diagonal top-left and bottom-right
            n1 = magnitude[(row - 1) * width + (col - 1)];
            n2 = magnitude[(row + 1) * width + (col + 1)];
        }
        else if (d == 135.0f)
        { // diagonal top-right and bottom-left
            n1 = magnitude[(row - 1) * width + (col + 1)];
            n2 = magnitude[(row + 1) * width + (col - 1)];
        }
        // Keep if local maximum
        output[idx] = (current >= n1 && current >= n2) ? current : 0.0f;
    }
}

// ===========================
// Kernel 5: Double Thresholding
// ===========================
// Input: thinned edges (float array); Output: 8-bit image with 255 (strong), 128 (weak), or 0 (non-edge)
// Thresholds: t_low, t_high (floats)
__global__ void kernelDoubleThreshold(const float *input, unsigned char *output, int width, int height, float t_low, float t_high)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = width * height;
    if (idx < total)
    {
        float val = input[idx];
        if (val >= t_high)
            output[idx] = 255; // strong edge
        else if (val >= t_low)
            output[idx] = 128; // weak edge
        else
            output[idx] = 0; // non-edge
    }
}

// ===========================
// Kernel 6: Hysteresis (Edge Tracking)
// ===========================
// A simple per-pixel kernel that, for each strong edge pixel, looks at its 8 neighbors
// and promotes any weak edge (128) neighbor to strong (255). For simplicity, assume one pass.
__global__ void kernelHysteresis(unsigned char *image, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= 1 && row < height - 1 && col >= 1 && col < width - 1)
    {
        int idx = row * width + col;
        if (image[idx] == 255)
        { // if current pixel is a strong edge, check neighbors
            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    int nIdx = (row + dx) * width + (col + dy);
                    if (image[nIdx] == 128)
                    { // if weak edge, promote it
                        image[nIdx] = 255;
                    }
                }
            }
        }
    }
}

// ===========================
// Host Main Function
// ===========================
int main()
{
    // Load image using OpenCV (assumed to be BGR)
    cv::Mat src = cv::imread("input.png", cv::IMREAD_COLOR);
    if (src.empty())
    {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    int width = src.cols;
    int height = src.rows;
    int numPixels = width * height;

    // Create host buffers for intermediate stages
    cv::Mat gray(src.size(), CV_8UC1);
    cv::Mat blur(src.size(), CV_8UC1);
    cv::Mat gradMag(src.size(), CV_32F);
    cv::Mat gradDir(src.size(), CV_32F);
    cv::Mat nms(src.size(), CV_32F);
    cv::Mat dt(src.size(), CV_8UC1); // double threshold result
    cv::Mat finalEdges(src.size(), CV_8UC1);

    // Allocate device memory
    uchar3 *d_src;
    unsigned char *d_gray;
    unsigned char *d_blur;
    float *d_gradMag;
    float *d_gradDir;
    float *d_nms;
    unsigned char *d_dt;

    cudaMalloc(&d_src, numPixels * sizeof(uchar3));
    cudaMalloc(&d_gray, numPixels * sizeof(unsigned char));
    cudaMalloc(&d_blur, numPixels * sizeof(unsigned char));
    cudaMalloc(&d_gradMag, numPixels * sizeof(float));
    cudaMalloc(&d_gradDir, numPixels * sizeof(float));
    cudaMalloc(&d_nms, numPixels * sizeof(float));
    cudaMalloc(&d_dt, numPixels * sizeof(unsigned char));

    // Copy source image data from host to device (assuming src is continuous)
    cudaMemcpy(d_src, src.ptr<uchar3>(), numPixels * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Launch Grayscale kernel using a 1D grid
    int threadsPerBlock = 256;
    int blocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
    kernelGrayscale<<<blocks, threadsPerBlock>>>(d_src, d_gray, width, height);
    cudaDeviceSynchronize();

    // Launch Gaussian Blur kernel using a 2D grid
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernelGaussianBlur<<<gridDim, blockDim>>>(d_gray, d_blur, width, height);
    cudaDeviceSynchronize();

    // Launch Sobel kernel using a 2D grid (skip borders)
    kernelSobel<<<gridDim, blockDim>>>(d_blur, d_gradMag, d_gradDir, width, height);
    cudaDeviceSynchronize();

    // Launch Non-Maximum Suppression kernel using a 2D grid (skip borders)
    kernelNonMaxSuppression<<<gridDim, blockDim>>>(d_gradMag, d_gradDir, d_nms, width, height);
    cudaDeviceSynchronize();

    // Launch Double Threshold kernel using a 1D grid
    float t_low = 20.0f;
    float t_high = 50.0f;
    blocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
    kernelDoubleThreshold<<<blocks, threadsPerBlock>>>(d_nms, d_dt, width, height, t_low, t_high);
    cudaDeviceSynchronize();

    // Launch Hysteresis kernel using a 2D grid
    kernelHysteresis<<<gridDim, blockDim>>>(d_dt, width, height);
    cudaDeviceSynchronize();

    // Copy final result from device back to host (final edge image is in d_dt)
    cv::Mat result(height, width, CV_8UC1);
    cudaMemcpy(result.ptr<unsigned char>(), d_dt, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save final edge-detected image
    cv::imwrite("cuda_canny_result.png", result);
    std::cout << "Result saved as cuda_canny_result.png" << std::endl;

    // Cleanup device memory
    cudaFree(d_src);
    cudaFree(d_gray);
    cudaFree(d_blur);
    cudaFree(d_gradMag);
    cudaFree(d_gradDir);
    cudaFree(d_nms);
    cudaFree(d_dt);

    return 0;
}
