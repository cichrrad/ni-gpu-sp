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
  cv::Mat image;         // Original image
  cv::Mat edges;         // Edge-detected image
  CannyMethod method;    // Algorithm selection
  bool iterative_output; // Flag to save intermediate outputs

  // Helper function to save intermediate images
  void saveIntermediateImage(const cv::Mat &image, const std::string &name);

  // Custom processing steps (private since they are not user-facing)
  void convertToGrayscale(const cv::Mat &input, cv::Mat &output);
  void applyGaussianBlur(const cv::Mat &input, cv::Mat &output);
  void applySobel(const cv::Mat &input, cv::Mat &gradX, cv::Mat &gradY,
                  cv::Mat &magnitude, cv::Mat &direction);
  void applyNonMaximumSuppression(const cv::Mat &magnitude,
                                  const cv::Mat &direction, cv::Mat &output);
  void applyDoubleThreshold(const cv::Mat &input, cv::Mat &output,
                            double lowThreshold, double highThreshold);
  void applyHysteresis(cv::Mat &input);

  void applyOpenCVCanny();
  void applyCustomCanny();

public:
  // Constructor with method selection and iterative output flag
  ImageTransform(const cv::Mat &inputImage, CannyMethod method = OPENCV,
                 bool iterative_output = false);

  // Applies the selected Canny edge detection method
  void applyCanny();

  // Getter for processed image
  cv::Mat getImage() const;
};

#endif // IMAGETRANSFORM_H
