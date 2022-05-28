// Author: Andrew Zamai
#ifndef METRICS_CALCULATION_H
#define METRICS_CALCULATION_H

#include <opencv2/core.hpp>

/**
 Compute Intersection Over Union metric given the predictedBoundingBox coordinates and trueBoundingBox coordinates.
 Image passed to check that rectangle coordinates don't go outside image? Q and to eventually draw rectangles? (should not be done in this function though).
 Coordinates are assumed to be a vector of 4 int elements: (x, y, witdth, height).
 
 @param image on which true bounding box and predicted bounding box are compared through IOU metric
 @param trueBoundingBox coordinates as vector of 4 int elements (x, y, witdth, height)
 @param predictedBoundingBox coordinates as vector of 4 int elements (x, y, witdth, height)
 @return intersection over union score
*/
double intersectionOverUnionSingleBB(const cv::Mat& image, std::vector<int> trueBoundingBox, std::vector<int> predictedBoundingBox);

//TODO: if more than 1 bounding box
// passed a vector<pair<vector<int>>> just iterate over it, call above function and sum
double intersectionOverUnion(const cv::Mat& image,
                             std::vector<std::pair<std::vector<int>,std::vector<int>>> trueAndPredictedBoundingBoxes);


/**
 Compute percent  of pixel of class Hand that are correctly classified.
 
 @param groundTruthImage ground truth segmented image
 @param segmentedImage segmented image to test
 @return percent of correctly classified pixels
*/
double pixelAccuracyHand(const cv::Mat& groundTruthImage, const cv::Mat& segmentedImage);

/**
 Compute percent  of pixel of class Non-hand that are correctly classified.
 
 @param groundTruthImage ground truth segmented image
 @param segmentedImage segmented image to test
 @return percent of correctly classified pixels
*/
double pixelAccuracyNonHand(const cv::Mat& groundTruthImage, const cv::Mat& segmentedImage);


#endif

