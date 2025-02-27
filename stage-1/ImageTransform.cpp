#include "ImageTransform.h"
#include <cmath>
#include <iostream>
#include <queue>

ImageTransform::ImageTransform(const cv::Mat &inputImage, CannyMethod method,
                               bool iterative_output) {
  this->image = inputImage.clone();
  this->method = method;
  this->iterative_output = iterative_output;
}

// Helper function to save images at each stage
void ImageTransform::saveIntermediateImage(const cv::Mat &img,
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
    applyOpenCVCanny();
  } else {
    applyCustomCanny();
  }
}

// OpenCV's built-in Canny Edge Detector
void ImageTransform::applyOpenCVCanny() {
  std::cout << "Applying OpenCV's Canny Edge Detector...\n";
  cv::Canny(image, edges, 100, 200); // Thresholds: 100 (low), 200 (high)
}

// Custom Canny Edge Detection (Sequential)
void ImageTransform::applyCustomCanny() {
  std::cout << "Applying Custom Canny Edge Detector...\n";

  cv::Mat gray, blurred, gradX, gradY, magnitude, direction, suppressed,
      thresholded;

  // 1️ Grayscale Conversion
  convertToGrayscale(image, gray);
  saveIntermediateImage(gray, "1_grayscale");

  // 2️ Gaussian Blur
  applyGaussianBlur(gray, blurred);
  saveIntermediateImage(blurred, "2_gaussian_blur");

  // 3️ Sobel Filtering
  applySobel(blurred, gradX, gradY, magnitude, direction);
  saveIntermediateImage(magnitude, "3_gradient_magnitude");

  // 4️ Non-Maximum Suppression
  applyNonMaximumSuppression(magnitude, direction, suppressed);
  saveIntermediateImage(suppressed, "4_non_max_suppression");

  // 5️ Double Thresholding
  applyDoubleThreshold(suppressed, thresholded, 30, 75);
  saveIntermediateImage(thresholded, "5_double_threshold");

  // 6️ Edge Tracking by Hysteresis
  applyHysteresis(thresholded);
  saveIntermediateImage(thresholded, "6_final_edges");

  edges = thresholded; // Store final edges
}

void ImageTransform::convertToGrayscale(const cv::Mat &input, cv::Mat &output) {
  output = cv::Mat(input.rows, input.cols, CV_8U);
  for (int y = 0; y < input.rows; y++) {
    for (int x = 0; x < input.cols; x++) {
      cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);
      uchar grayValue = static_cast<uchar>(
          0.2989 * pixel[2] + 0.5870 * pixel[1] + 0.1140 * pixel[0]);
      output.at<uchar>(y, x) = grayValue;
    }
  }
}

void ImageTransform::applyGaussianBlur(const cv::Mat &input, cv::Mat &output) {
  int kernelSize = 5;
  int halfSize = kernelSize / 2;
  output = cv::Mat::zeros(input.size(), CV_8U);

  double kernel[5][5] = {{2, 4, 5, 4, 2},
                         {4, 9, 12, 9, 4},
                         {5, 12, 15, 12, 5},
                         {4, 9, 12, 9, 4},
                         {2, 4, 5, 4, 2}};
  double kernelSum = 159.0;

  for (int y = halfSize; y < input.rows - halfSize; y++) {
    for (int x = halfSize; x < input.cols - halfSize; x++) {
      double sum = 0;
      for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
          sum += input.at<uchar>(y + i, x + j) *
                 kernel[i + halfSize][j + halfSize];
        }
      }
      output.at<uchar>(y, x) = static_cast<uchar>(sum / kernelSum);
    }
  }
}

void ImageTransform::applySobel(const cv::Mat &input, cv::Mat &gradX,
                                cv::Mat &gradY, cv::Mat &magnitude,
                                cv::Mat &direction) {
  int kernelSize = 3;
  int halfSize = kernelSize / 2;

  int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

  int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  gradX = cv::Mat::zeros(input.size(), CV_64F);
  gradY = cv::Mat::zeros(input.size(), CV_64F);
  magnitude = cv::Mat::zeros(input.size(), CV_64F);
  direction = cv::Mat::zeros(input.size(), CV_64F);

  for (int y = halfSize; y < input.rows - halfSize; y++) {
    for (int x = halfSize; x < input.cols - halfSize; x++) {
      double sumX = 0, sumY = 0;
      for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
          sumX +=
              input.at<uchar>(y + i, x + j) * Gx[i + halfSize][j + halfSize];
          sumY +=
              input.at<uchar>(y + i, x + j) * Gy[i + halfSize][j + halfSize];
        }
      }
      gradX.at<double>(y, x) = sumX;
      gradY.at<double>(y, x) = sumY;
      magnitude.at<double>(y, x) = sqrt(sumX * sumX + sumY * sumY);
      direction.at<double>(y, x) = atan2(sumY, sumX) * (180.0 / CV_PI);
    }
  }
}

void ImageTransform::applyNonMaximumSuppression(const cv::Mat &magnitude,
                                                const cv::Mat &direction,
                                                cv::Mat &output) {
  output = cv::Mat::zeros(magnitude.size(), CV_64F);

  for (int y = 1; y < magnitude.rows - 1; y++) {
    for (int x = 1; x < magnitude.cols - 1; x++) {
      double angle = direction.at<double>(y, x);
      angle = fmod((angle + 180), 180); // Normalize to [0, 180]

      double q = 0, r = 0;
      if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180)) {
        q = magnitude.at<double>(y, x + 1);
        r = magnitude.at<double>(y, x - 1);
      } else if (22.5 <= angle && angle < 67.5) {
        q = magnitude.at<double>(y - 1, x + 1);
        r = magnitude.at<double>(y + 1, x - 1);
      } else if (67.5 <= angle && angle < 112.5) {
        q = magnitude.at<double>(y - 1, x);
        r = magnitude.at<double>(y + 1, x);
      } else if (112.5 <= angle && angle < 157.5) {
        q = magnitude.at<double>(y + 1, x + 1);
        r = magnitude.at<double>(y - 1, x - 1);
      }

      if (magnitude.at<double>(y, x) >= q && magnitude.at<double>(y, x) >= r) {
        output.at<double>(y, x) = magnitude.at<double>(y, x);
      } else {
        output.at<double>(y, x) = 0;
      }
    }
  }
}

void ImageTransform::applyDoubleThreshold(const cv::Mat &input, cv::Mat &output,
                                          double lowThreshold,
                                          double highThreshold) {
  output = cv::Mat::zeros(input.size(), CV_8U);

  for (int y = 0; y < input.rows; y++) {
    for (int x = 0; x < input.cols; x++) {
      double val = input.at<double>(y, x);
      if (val >= highThreshold) {
        output.at<uchar>(y, x) = 255; // Strong edge
      } else if (val >= lowThreshold) {
        output.at<uchar>(y, x) = 50; // Weak edge
      } else {
        output.at<uchar>(y, x) = 0; // Non-edge
      }
    }
  }
}

void ImageTransform::applyHysteresis(cv::Mat &input) {
  std::queue<cv::Point> strongPixels;

  for (int y = 1; y < input.rows - 1; y++) {
    for (int x = 1; x < input.cols - 1; x++) {
      if (input.at<uchar>(y, x) == 255) {
        strongPixels.push(cv::Point(x, y));
      }
    }
  }

  while (!strongPixels.empty()) {
    cv::Point p = strongPixels.front();
    strongPixels.pop();

    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        int nx = p.x + dx, ny = p.y + dy;
        if (input.at<uchar>(ny, nx) == 50) {
          input.at<uchar>(ny, nx) = 255;
          strongPixels.push(cv::Point(nx, ny));
        }
      }
    }
  }

  // Remove remaining weak edges
  for (int y = 0; y < input.rows; y++) {
    for (int x = 0; x < input.cols; x++) {
      if (input.at<uchar>(y, x) == 50) {
        input.at<uchar>(y, x) = 0;
      }
    }
  }
}

// Getter for processed image
cv::Mat ImageTransform::getImage() const { return edges; }
