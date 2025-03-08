#include "ImageLoader.h"
#include <iostream>

cv::Mat ImageLoader::loadImage(const std::string &filePath) {
    cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR); // Load in color mode
    if (image.empty()) {
        std::cerr << "Error: Could not load image from " << filePath << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Image loaded successfully: " << filePath << std::endl;
    return image;
}

void ImageLoader::saveImage(const std::string &filePath, const cv::Mat &image) {
    if (!cv::imwrite(filePath, image)) {
        std::cerr << "Error: Could not save image to " << filePath << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Image saved successfully: " << filePath << std::endl;
}
