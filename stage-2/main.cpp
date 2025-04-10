#include "ImageLoader.h"
#include "ImageTransform.h"
#include <chrono>
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <input_image> <output_image> <method> [iterative]\n";
    std::cerr << "Methods: opencv | custom\n";
    std::cerr << "Optional: Add 'iterative' (only with custom) to save "
                 "intermediate outputs .\n";
    return EXIT_FAILURE;
  }

  std::string inputFilePath = argv[1];
  std::string outputFilePath = argv[2];
  std::string methodChoice = argv[3];

  // opencv ... uses opencv (duh)
  // custom ... uses custom implementation (opencv only for storage purposes)
  CannyMethod method = (methodChoice == "custom") ? CUSTOM : OPENCV;

  // for saving after each stage
  bool iterative_output = (argc > 4 && std::string(argv[4]) == "iterative");

  if (iterative_output) {
    std::cout << "Iterative output enabled: Saving intermediate images...\n";
  }

  cv::Mat image = ImageLoader::loadImage(inputFilePath);
  ImageTransform transform(image, method, iterative_output);
  transform.applyCanny();
  ImageLoader::saveImage(outputFilePath, transform.getImage());

  return EXIT_SUCCESS;
}
