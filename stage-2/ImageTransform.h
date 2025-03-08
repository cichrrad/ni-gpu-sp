#ifndef IMAGETRANSFORM_H
#define IMAGETRANSFORM_H

#include <opencv2/opencv.hpp>
#include <string>

enum CannyMethod {
  OPENCV, // Use OpenCV's built-in Canny detector
  CUSTOM  // Use custom implementation
};

class ImageTransform {
private:
  cv::Mat src;           // Original image
  cv::Mat dest;          // Edge-detected image
  CannyMethod method;    // Algorithm selection
  bool iterative_output; // Flag to save intermediate outputs

  void _saveIntermediateImage(const cv::Mat &image, const std::string &name);

  void _applyOpenCVCanny();
  void _applyCustomCanny();

  void _convertToGrayscale(const cv::Mat &input, cv::Mat &output);
  void _applyGaussianBlur(const cv::Mat &input, cv::Mat &output);
  void _applySobel(const cv::Mat &input, cv::Mat &magnitude,
                   cv::Mat &direction);
  void _applyNonMaximumSuppression(const cv::Mat &magnitude,
                                   const cv::Mat &direction, cv::Mat &output);
  void _applyDoubleThreshold(cv::Mat &input, cv::Mat &output);
  void _applyHysteresis(cv::Mat &input);

public:
  ImageTransform(const cv::Mat &inputImage, CannyMethod method = OPENCV,
                 bool iterative_output = false);

  void applyCanny();

  cv::Mat getImage() const;
};

#endif // IMAGETRANSFORM_H
