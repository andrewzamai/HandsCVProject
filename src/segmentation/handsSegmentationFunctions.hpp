// Author: Andrew Zamai 2038522

#ifndef HANDS_SEGMENTATION_FUNCTIONS_H
#define HANDS_SEGMENTATION_FUNCTIONS_H

#include <opencv2/core.hpp>

/**
 Since I coldn't find any solution to convert the keras model trained in python to something that could be embedded entirely in the C++ framework
 (most of the already existing libraries do not support the conversion of the Conv2Dtranspose layer, which is part of the decoder block of my Unet model),
 I decided to deploy the model by writing a small script in python that just makes a forward propagation step of an image to be segmented.
 All preprocessing and postprocessing steps are done here before running the python script through a system call.
 
 Since the model will be loaded each time the script is called, in order to not waste time, I have decided to define a function that segments a
 vector of an arbitrary number of images. To process a single image just pass a vector with only 1 image.
 
 @param inImages a vector of images to be segmented
 @param outImages a vector of segmented images by the model (without any post processing)
 */
void segmentImages(const std::vector<cv::Mat>& inImages, std::vector<cv::Mat>& outImages);


#endif
