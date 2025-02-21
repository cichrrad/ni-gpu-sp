#include "ImageTransform.h"
#include <iostream>

ImageTransform::ImageTransform(const cv::Mat &inputImage, CannyMethod method) {
    this->image = inputImage.clone();
    this->method = method;
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

// Your own custom Canny (placeholder for now)
void ImageTransform::applyCustomCanny() {
    std::cout << "Applying Custom Canny Edge Detector (Not Implemented Yet)...\n";
    edges = image.clone(); // Temporary: Just copies the image for now
}

// Getter for the processed image
cv::Mat ImageTransform::getImage() const {
    return edges;
}
