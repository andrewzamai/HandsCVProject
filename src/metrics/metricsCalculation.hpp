// Author: Andrew Zamai 2038522

#ifndef METRICS_CALCULATION_H
#define METRICS_CALCULATION_H

#include <opencv2/core.hpp>

/**
 Computes Intersection Over Union metric given the predictedBoundingBox coordinates and trueBoundingBox coordinates.
 Coordinates are assumed to be a vector of 4 int elements: (x, y, witdth, height). Exception is thrown if bounding boxes fall outside the image.
 
 @param image on which true bounding box and predicted bounding box are compared through IOU metric
 @param trueBoundingBox coordinates as vector of 4 int elements (x, y, witdth, height)
 @param predictedBoundingBox coordinates as vector of 4 int elements (x, y, witdth, height)
 @return intersection over union score
*/
double intersectionOverUnionSingleBB(const cv::Mat& image, std::vector<int> trueBoundingBox, std::vector<int> predictedBoundingBox);

/**
 Computes Intersection Over Union when provided an image with more than 1 bounding boxes over it.
 
 @param image on which true bounding boxes and predicted bounding boxes are compared through IOU metric
 @param trueBoundingBoxes a vector of ground truth bounding boxes
 @param predictedBoundingBoxes a vector of predicted bounding boxes
 @return intersection over union score
 */
double intersectionOverUnion(const cv::Mat& image, std::vector<std::vector<int>> trueBoundingBoxes, std::vector<std::vector<int>> predictedBoundingBoxes);


/**
 Accuracy metric for hands segmentation task computed as
 (n° of hand pixels classified as hand + n° of non-hand pixels classified as non-hand)/(total n° of pixels)
 
 @param groundTruthImage ground truth segmented image of CV_8UC1 type
 @param segmentedImage segmented image to test of CV_8UC3 type
 @return accuracy score
*/
double accuracyMetricForSegmentation(const cv::Mat& groundTruthImage, const cv::Mat& segmentedImage);


/**
 Computes percent of pixels of class Hand that are correctly classified.
 
 @param groundTruthImage ground truth segmented image of CV_8UC1 type
 @param segmentedImage segmented image to test of CV_8UC3 type
 @return percent of correctly classified pixels
*/
double pixelPrecisionHand(const cv::Mat& groundTruthImage, const cv::Mat& segmentedImage);

/**
 Computes percent of pixels of class Non-hand that are correctly classified.
 
 @param groundTruthImage ground truth segmented image CV_8UC1 type
 @param segmentedImage segmented image to test CV_8UC3 type
 @return percent of correctly classified pixels an non hand
*/
double pixelPrecisionNonHand(const cv::Mat& groundTruthImage, const cv::Mat& segmentedImage);


#endif

