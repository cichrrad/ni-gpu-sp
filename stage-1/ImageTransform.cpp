#include "ImageTransform.h"
#include <iostream>
#include <cmath>
#include <queue>

ImageTransform::ImageTransform(const cv::Mat &inputImage, CannyMethod method, bool iterative_output) {
    this->image = inputImage.clone();
    this->method = method;
    this->iterative_output = iterative_output;
}

// Helper function to save images at each stage
void ImageTransform::saveIntermediateImage(const cv::Mat &img, const std::string &name) {
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

    cv::Mat gray, blurred, gradientX, gradientY, magnitude, direction, suppressed, thresholded;

    // Convert to Grayscale
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    saveIntermediateImage(gray, "1_grayscale");

    // Apply Gaussian Blur (5x5)
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4);
    saveIntermediateImage(blurred, "2_gaussian_blur");

    // Compute Sobel Gradients
    cv::Sobel(blurred, gradientX, CV_64F, 1, 0, 3); // Gx
    cv::Sobel(blurred, gradientY, CV_64F, 0, 1, 3); // Gy

    // Compute gradient magnitude and direction
    magnitude = cv::Mat::zeros(gradientX.size(), CV_64F);
    direction = cv::Mat::zeros(gradientX.size(), CV_64F);

    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            double gx = gradientX.at<double>(y, x);
            double gy = gradientY.at<double>(y, x);
            magnitude.at<double>(y, x) = sqrt(gx * gx + gy * gy);
            direction.at<double>(y, x) = atan2(gy, gx) * (180.0 / CV_PI);
        }
    }
    saveIntermediateImage(magnitude, "3_gradient_magnitude");

    // Non-Maximum Suppression
    suppressed = cv::Mat::zeros(magnitude.size(), CV_64F);

    for (int y = 1; y < gray.rows - 1; y++) {
        for (int x = 1; x < gray.cols - 1; x++) {
            double angle = direction.at<double>(y, x);
            angle = fmod((angle + 180), 180);

            double q = 255, r = 255;
            if ((0 <= angle < 22.5) || (157.5 <= angle <= 180)) {
                q = magnitude.at<double>(y, x + 1);
                r = magnitude.at<double>(y, x - 1);
            } else if (22.5 <= angle < 67.5) {
                q = magnitude.at<double>(y - 1, x + 1);
                r = magnitude.at<double>(y + 1, x - 1);
            } else if (67.5 <= angle < 112.5) {
                q = magnitude.at<double>(y - 1, x);
                r = magnitude.at<double>(y + 1, x);
            } else if (112.5 <= angle < 157.5) {
                q = magnitude.at<double>(y + 1, x + 1);
                r = magnitude.at<double>(y - 1, x - 1);
            }

            if (magnitude.at<double>(y, x) >= q && magnitude.at<double>(y, x) >= r) {
                suppressed.at<double>(y, x) = magnitude.at<double>(y, x);
            } else {
                suppressed.at<double>(y, x) = 0;
            }
        }
    }
    saveIntermediateImage(suppressed, "4_non_max_suppression");

    // Double Thresholding
    double highThreshold = 75, lowThreshold = 30;
    thresholded = cv::Mat::zeros(suppressed.size(), CV_8U);

    for (int y = 0; y < suppressed.rows; y++) {
        for (int x = 0; x < suppressed.cols; x++) {
            double val = suppressed.at<double>(y, x);
            if (val >= highThreshold) {
                thresholded.at<uchar>(y, x) = 255;
            } else if (val >= lowThreshold) {
                thresholded.at<uchar>(y, x) = 50;
            }
        }
    }
    saveIntermediateImage(thresholded, "5_double_threshold");

    // Edge Tracking by Hysteresis
    std::queue<cv::Point> strongPixels;
    for (int y = 1; y < thresholded.rows - 1; y++) {
        for (int x = 1; x < thresholded.cols - 1; x++) {
            if (thresholded.at<uchar>(y, x) == 255) {
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
                if (thresholded.at<uchar>(ny, nx) == 50) {
                    thresholded.at<uchar>(ny, nx) = 255;
                    strongPixels.push(cv::Point(nx, ny));
                }
            }
        }
    }

    edges = thresholded;
    //saveIntermediateImage(edges, "6_final_edges");
}

// Getter for processed image
cv::Mat ImageTransform::getImage() const {
    return edges;
}
