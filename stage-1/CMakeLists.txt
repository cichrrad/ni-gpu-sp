cmake_minimum_required(VERSION 3.10)
project(stage-1)

find_package(OpenCV REQUIRED)

add_executable(canny main.cpp ImageLoader.cpp ImageTransform.cpp)
target_link_libraries(canny ${OpenCV_LIBS})
