#include "ImageTransform.h"
#include <cmath>
#include <iostream>
#include <queue>
#include <vector>

ImageTransform::ImageTransform(const cv::Mat &inputImage, CannyMethod method,
                               bool iterative_output) {
  this->src = inputImage.clone();
  this->method = method;
  this->iterative_output = iterative_output;
}

// Helper function to save images at each stage
void ImageTransform::_saveIntermediateImage(const cv::Mat &img,
                                            const std::string &name) {
  if (iterative_output) {
    std::string filename = "output_" + name + ".png";
    cv::imwrite(filename, img);
    std::cout << "Saved intermediate image: " << filename << std::endl;
  }
}

// Applies the chosen Canny algorithm
void ImageTransform::applyCanny() {
  if (method == OPENCV) {
    _applyOpenCVCanny();
  } else {
    _applyCustomCanny();
  }
}

// OpenCV's built-in Canny Edge Detector
void ImageTransform::_applyOpenCVCanny() {
  std::cout << "Applying OpenCV's Canny Edge Detector...\n";
  cv::Canny(src, dest, 100, 200);
}

// Custom Canny Edge Detection
void ImageTransform::_applyCustomCanny() {
  std::cout << "Applying Custom Canny Edge Detector...\n";
  cv::Mat magnitude, direction, temp;

  //* TODOs are for next stage

  // TODO parallel over rows or image segments
  _convertToGrayscale(src, dest);
  _saveIntermediateImage(dest, "1_grayscale");

  // TODO parallel over kernel area (5x5)
  _applyGaussianBlur(dest, temp);
  _saveIntermediateImage(temp, "2_gaussian_blur");

  // TODO parallel over kernel area (3x3)
  _applySobel(temp, magnitude, direction);
  _saveIntermediateImage(magnitude, "3_gradient_magnitude");
  _saveIntermediateImage(direction, "3_gradient_direction");

  // TODO parallel over triple row stripes or image segments
  _applyNonMaximumSuppression(magnitude, direction, temp);
  _saveIntermediateImage(temp, "4_non_max_suppression");

  // TODO parallel over rows or image segments
  _applyDoubleThreshold(temp, dest);
  _saveIntermediateImage(dest, "5_double_threshold");

  // TODO parallel queue processing
  _applyHysteresis(dest);
}

// 1️ Convert image to grayscale
void ImageTransform::_convertToGrayscale(const cv::Mat &input,
                                         cv::Mat &output) {
  CV_Assert(input.channels() == 3);

  // init output image with same dimensions but only one channel for grayscale
  output = cv::Mat(input.rows, input.cols, CV_8UC1);

  // row processing
  for (int i = 0; i < input.rows; i++) {
    // pointer to current row
    // Vec3b -- vector of 3 uchar fields per element -- 3 channels per pixel
    // (duh)
    const cv::Vec3b *inputRow = input.ptr<cv::Vec3b>(i);
    // just uchar -- 1 channel per pixel
    uchar *outputRow = output.ptr<uchar>(i);

    for (int j = 0; j < input.cols; j++) {
      outputRow[j] = static_cast<uchar>(0.299 * inputRow[j][2] + // Red
                                        0.587 * inputRow[j][1] + // Green
                                        0.114 * inputRow[j][0]   // Blue
      );
    }
  }
}

// 2 Apply Gaussian Blur
void ImageTransform::_applyGaussianBlur(const cv::Mat &input, cv::Mat &output) {
  CV_Assert(input.channels() == 1); // ensure grayscale input

  //[https://en.wikipedia.org/wiki/Gaussian_filter]
  // fixed kernel for sigma = 1.4
  const float kernel[5][5] = {{2, 4, 5, 4, 2},
                              {4, 9, 12, 9, 4},
                              {5, 12, 15, 12, 5},
                              {4, 9, 12, 9, 4},
                              {2, 4, 5, 4, 2}};
  const float kernelSum = 159.0; // Normalization factor

  int k = 2; // k = 2 for size 5x5 kernel
  output = cv::Mat(input.rows, input.cols, CV_8UC1, cv::Scalar(0));

  // apply convolution kernel
  for (int i = k; i < input.rows - k; i++) {
    for (int j = k; j < input.cols - k; j++) {
      float sum = 0.0;

      // operation
      for (int x = -k; x <= k; x++) {
        for (int y = -k; y <= k; y++) {
          sum += kernel[x + k][y + k] * input.at<uchar>(i + x, j + y);
        }
      }

      // normalize and assign result
      output.at<uchar>(i, j) = static_cast<uchar>(sum / kernelSum);
    }
  }
}

// 3️ Apply Sobel filter (Computes magnitude & direction)
void ImageTransform::_applySobel(const cv::Mat &input, cv::Mat &magnitude,
                                 cv::Mat &direction) {
  CV_Assert(input.channels() == 1); // ensure grayscale input

  // TLDR:
  // We want to calculate gradient -- 1st derivative in axis directions (x,y)
  // --> Gx and Gy
  // then gradient G = sqrt(Gx^2 + Gy^2)
  // direction of gradient theta = arctan(Gy/Gx)
  // we clamp direction to 4 axis -- 0, 90, 135, 45

  // Sobel Kernels for extracting gradients in axis directions
  // [https://en.wikipedia.org/wiki/Sobel_operator]

  int GxKernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  int GyKernel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  magnitude = cv::Mat(input.rows, input.cols, CV_32F, cv::Scalar(0));
  direction = cv::Mat(input.rows, input.cols, CV_32F, cv::Scalar(0));

  int k = 1; // k=1 for size 3x3 kernel

  // Compute gradients via convolution with sobel kernels
  for (int i = k; i < input.rows - k; i++) {
    for (int j = k; j < input.cols - k; j++) {
      float Gx = 0, Gy = 0;

      // operation
      for (int x = -k; x <= k; x++) {
        for (int y = -k; y <= k; y++) {
          uchar pixel = input.at<uchar>(i + x, j + y);
          Gx += pixel * GxKernel[x + k][y + k];
          Gy += pixel * GyKernel[x + k][y + k];
        }
      }

      // Compute magnitude
      magnitude.at<float>(i, j) = std::sqrt(Gx * Gx + Gy * Gy);

      // Compute direction & convert
      float angle = std::atan2(Gy, Gx) * 180.0 / CV_PI;

      // shift angle to always be [0, 180] (negative to positive)
      if (angle < 0)
        angle += 180;

      // clamp 0, 45, 90, or 135 degrees
      if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180))
        direction.at<float>(i, j) = 0;
      else if (angle >= 22.5 && angle < 67.5)
        direction.at<float>(i, j) = 45;
      else if (angle >= 67.5 && angle < 112.5)
        direction.at<float>(i, j) = 90;
      else
        direction.at<float>(i, j) = 135;
    }
  }
}

// 4️ Non-Maximum Suppression
void ImageTransform::_applyNonMaximumSuppression(const cv::Mat &magnitude,
                                                 const cv::Mat &direction,
                                                 cv::Mat &output) {
  CV_Assert(magnitude.type() == CV_32F &&
            direction.type() == CV_32F); // ensure proper input types

  output = cv::Mat::zeros(magnitude.size(), CV_32F);

  for (int i = 1; i < magnitude.rows - 1; i++) {
    for (int j = 1; j < magnitude.cols - 1; j++) {

      float currentMag = magnitude.at<float>(i, j);
      float angle = direction.at<float>(i, j);
      float neighbor1 = 0, neighbor2 = 0;

      // select pixels to compare based on the gradient direction
      if (angle == 0) {                            // edge is |
        neighbor1 = magnitude.at<float>(i, j - 1); // left
        neighbor2 = magnitude.at<float>(i, j + 1); // right

      } else if (angle == 90) {                    // edge is --
        neighbor1 = magnitude.at<float>(i - 1, j); // top
        neighbor2 = magnitude.at<float>(i + 1, j); // bottom

      } else if (angle == 135) {                       // edge is \

        neighbor1 = magnitude.at<float>(i - 1, j - 1); // top-left
        neighbor2 = magnitude.at<float>(i + 1, j + 1); // bottom-right

      } else if (angle == 45) {                        // edge is /
        neighbor1 = magnitude.at<float>(i - 1, j + 1); // top-right
        neighbor2 = magnitude.at<float>(i + 1, j - 1); // bottom-left
      }

      // Keep the pixel if it's the local maximum, otherwise suppress it
      if (currentMag >= neighbor1 && currentMag >= neighbor2) {
        output.at<float>(i, j) = currentMag;
      } else {
        output.at<float>(i, j) = 0;
      }
    }
  }
}

// 5 Double tresholding
void ImageTransform::_applyDoubleThreshold(cv::Mat &input, cv::Mat &output) {
  CV_Assert(input.type() == CV_32F); // ensure proper input type

  // TODO : implement less unga-bunga threshold selection
  // (ANYTHING ADAPTIVE would be better)
  // ideal? --> Otsu’s Method
  // [https://en.wikipedia.org/wiki/Otsu%27s_method]

  float highThreshold = 47;
  float lowThreshold = 127;

  output = cv::Mat::zeros(input.size(), CV_8UC1);

  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {

      float pixelVal = input.at<float>(i, j);

      if (pixelVal >= highThreshold) {
        output.at<uchar>(i, j) = 255; // Strong edge (white)
      } else if (pixelVal >= lowThreshold) {
        output.at<uchar>(i, j) = 128; // Weak edge (gray)
      } else {
        output.at<uchar>(i, j) = 0; // Suppressed
      }
    }
  }
}

// 6️ Edge Tracking by Hysteresis
void ImageTransform::_applyHysteresis(cv::Mat &input) {
  CV_Assert(input.type() == CV_8UC1); // Ensure proper input type

  //[https://en.wikipedia.org/wiki/Connected-component_labeling]
  cv::Mat visited = cv::Mat::zeros(input.size(), CV_8UC1);
  std::queue<std::pair<int, int>> strongEdges;

  // Step 1: Add all strong edges to the queue and visit them
  for (int i = 1; i < input.rows - 1; i++) {
    for (int j = 1; j < input.cols - 1; j++) {
      if (input.at<uchar>(i, j) == 255) {

        strongEdges.push({i, j});
        visited.at<uchar>(i, j) = 1;
      }
    }
  }

  // Step 2: Perform BFS to link weak edges to strong edges
  while (!strongEdges.empty()) {
    auto [x, y] = strongEdges.front();
    strongEdges.pop();

    // Check surrounding pixels
    for (int dx = -1; dx <= 1; dx++) {
      for (int dy = -1; dy <= 1; dy++) {
        int nx = x + dx, ny = y + dy;

        // promote attached weak edges to strong
        if (nx >= 0 && ny >= 0 && nx < input.rows && ny < input.cols) {
          if (input.at<uchar>(nx, ny) == 128 &&
              visited.at<uchar>(nx, ny) == 0) {

            input.at<uchar>(nx, ny) = 255;
            visited.at<uchar>(nx, ny) = 1;
            strongEdges.push({nx, ny});
          }
        }
      }
    }
  }

  // Step 3: Remove remaining weak edges
  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      if (input.at<uchar>(i, j) == 128) {
        input.at<uchar>(i, j) =
            0; // Suppress weak edges not linked to strong ones
      }
    }
  }
}

// Getter for processed image
cv::Mat ImageTransform::getImage() const { return dest; }