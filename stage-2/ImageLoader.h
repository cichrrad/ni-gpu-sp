#ifndef IMAGELOADER_H
#define IMAGELOADER_H

#include <opencv2/opencv.hpp>
#include <string>

class ImageLoader {
public:
    // Loads an image from a given file path
    static cv::Mat loadImage(const std::string &filePath);

    // Saves an image to a given file path
    static void saveImage(const std::string &filePath, const cv::Mat &image);
};

#endif // IMAGELOADER_H
