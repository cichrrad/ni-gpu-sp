#ifndef IMAGETRANSFORM_H
#define IMAGETRANSFORM_H

#include <opencv2/opencv.hpp>
#include <string>

enum CannyMethod {
    OPENCV,  // Use OpenCV's built-in Canny detector
    CUSTOM   // Use custom implementation
};

class ImageTransform {
private:
    cv::Mat image; // Original image
    cv::Mat edges; // Edge-detected image
    CannyMethod method; // Algorithm selection

public:
    // Constructor with method selection
    ImageTransform(const cv::Mat &inputImage, CannyMethod method = OPENCV);

    // Applies the selected Canny edge detection method
    void applyCanny();

    // OpenCV-based Canny Edge Detection
    void applyOpenCVCanny();

    // Custom implementation of Canny Edge Detector
    void applyCustomCanny();

    // Getter for processed image
    cv::Mat getImage() const;
};

#endif // IMAGETRANSFORM_H
