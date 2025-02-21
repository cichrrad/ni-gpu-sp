#include "ImageLoader.h"
#include "ImageTransform.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image> <method>\n";
        std::cerr << "Methods: opencv | custom\n";
        return EXIT_FAILURE;
    }

    std::string inputFilePath = argv[1];
    std::string outputFilePath = argv[2];
    std::string methodChoice = argv[3];

    // Determine which method to use
    CannyMethod method = (methodChoice == "custom") ? CUSTOM : OPENCV;

    // Load the image
    cv::Mat image = ImageLoader::loadImage(inputFilePath);

    // Create ImageTransform object with selected method
    ImageTransform transform(image, method);

    // Apply Canny edge detection
    transform.applyCanny();

    // Save the processed image
    ImageLoader::saveImage(outputFilePath, transform.getImage());

    return EXIT_SUCCESS;
}
