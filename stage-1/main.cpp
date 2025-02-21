#include "ImageLoader.h"
#include "ImageTransform.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image> <method> [iterative]\n";
        std::cerr << "Methods: opencv | custom\n";
        std::cerr << "Optional: Add 'iterative' to save intermediate outputs.\n";
        return EXIT_FAILURE;
    }

    std::string inputFilePath = argv[1];
    std::string outputFilePath = argv[2];
    std::string methodChoice = argv[3];

    // Determine which method to use (opencv or custom)
    CannyMethod method = (methodChoice == "custom") ? CUSTOM : OPENCV;

    // Check if 'iterative' flag is provided
    bool iterative_output = (argc > 4 && std::string(argv[4]) == "iterative");

    if (iterative_output) {
        std::cout << "Iterative output enabled: Saving intermediate images...\n";
    }

    // Load the image
    cv::Mat image = ImageLoader::loadImage(inputFilePath);

    // Create ImageTransform object with selected method and iterative flag
    ImageTransform transform(image, method, iterative_output);

    // Apply Canny edge detection
    transform.applyCanny();

    // Save the final processed image
    ImageLoader::saveImage(outputFilePath, transform.getImage());

    return EXIT_SUCCESS;
}
