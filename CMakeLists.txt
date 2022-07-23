cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(HandsCVProject VERSION 1)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(HandsSegmentationMain src/segmentation/handsSegmentationMain.cpp src/segmentation/handsSegmentationFunctions.cpp src/metrics/metricsCalculation.cpp)

target_link_libraries(HandsSegmentationMain ${OpenCV_LIBS})

