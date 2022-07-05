// Author: Andrew Zamai 2038522

#include "metricsCalculation.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <exception>

using namespace std;
using namespace cv;

// Exception to be thrown when groundTruth and image to test  have different sizes
class MetricsInvalidInputDimension : public exception
{
    virtual const char* what() const throw()
    {
      return "Inputs of different sizes";
    }
}metricsInvInpDim;

// Exception to be thrown when bounding box falls outside the image
class MetricsOutOfImageBoundingBox : public exception
{
    virtual const char* what() const throw()
    {
      return "The bounding box falls outside the image";
    }
}metricsOutOfImg;


/* --------------------------------------------------------------------------------- HAND DETECTION METRICS ---------------------------------------------------------------------------------  */


double intersectionOverUnionSingleBB(const Mat& image, vector<int> trueBoundingBox, vector<int> predictedBoundingBox)
{
    // True Bounding Box (x, y, width, height) retrieval
    int xTB = trueBoundingBox[0];
    int yTB = trueBoundingBox[1];
    int widthTB = trueBoundingBox[2];
    int heightTB = trueBoundingBox[3];
    
    // Predicted Bounding Box (x, y, width, height) retrieval
    int xPB = predictedBoundingBox[0];
    int yPB = predictedBoundingBox[1];
    int widthPB = predictedBoundingBox[2];
    int heightPB = predictedBoundingBox[3];
    
    // Checking that bounding boxes are not outside image
    if(xTB < 0 || yTB < 0 || xTB+widthTB >= image.cols || yTB+heightTB >= image.rows)
    {
        throw MetricsOutOfImageBoundingBox();
    }
    if(xPB < 0 || yPB < 0 || xPB+widthPB >= image.cols || yPB+heightPB >= image.rows)
    {
        throw MetricsOutOfImageBoundingBox();
    }
    
    // compute Intersection rectangle top left corner coordinates (xA, yA) and top right bottom coordinates (xB, yB)
    int xA = ((xTB - xPB) >= 0) ? xTB : xPB; // max of the two since measured from top left corner
    int yA = ((yTB - yPB) >= 0) ? yTB : yPB; // max
    int xB = ((xTB + widthTB) - (xPB + widthPB) >= 0) ? (xPB + widthPB) : (xTB + widthTB); // min
    int yB = ((yTB + heightTB) - (yPB + heightPB) >= 0) ? (yPB + heightPB) : (yTB + heightTB); //  min
              
    // compute the area of the intersection rectangle
    int widthIntersectionRect = ((xB - xA + 1) >= 0) ? (xB - xA + 1) : 0;
    int heightIntersectionRect = ((yB - yA + 1) >= 0) ? (yB - yA + 1) : 0;
    int intersectionArea =  widthIntersectionRect * heightIntersectionRect;
              
    // compute area of ground truth bounding box
    int trueBBArea =  widthTB * heightTB;
         
    // compute area of predicted bounding box
    int predictedBBArea =  widthPB * heightPB;
              
    // compute now IOY dividing intersection rectangle area by area of union of bounding boxes (note to subtract intersection)
    double iou =  static_cast<double>(intersectionArea)/(trueBBArea + predictedBBArea - intersectionArea);
    
    return iou;
}

/*
 To determine which Predicted Bounding Box corresponds to which Ground truth bounding box:
    1) it is computed the IOU score of 1 predicted BB with respect to all possible GT-BBs;
    2) the IOU for this Predicted BB is the maximum of the computed IOU scores;
    3) this is done for all predicted BBs and the total IOU score is the sum of all maximum scores
*/
double intersectionOverUnion(const Mat& image, vector<vector<int>> trueBoundingBoxes, vector<vector<int>> predictedBoundingBoxes)
{
    double iouTotal = 0;
    
    for(int i=0; i<predictedBoundingBoxes.size(); i++)
    {
        double maxIOU = 0;
        for(int j=0; j<trueBoundingBoxes.size(); j++)
        {
            double score = intersectionOverUnionSingleBB(image, trueBoundingBoxes[j], predictedBoundingBoxes[i]);
            if(score > maxIOU)
            {
                maxIOU = score;
            }
        }
        
        iouTotal += maxIOU;
    }
    
    return iouTotal;
}


/* --------------------------------------------------------------------------------- HAND SEGMENTATION METRICS ---------------------------------------------------------------------------------  */


double pixelAccuracyHand(const Mat& groundTruthImage, const Mat& segmentedImage)
{
    // if different sizes throw Exception
    if(groundTruthImage.rows != segmentedImage.rows || groundTruthImage.cols != segmentedImage.cols)
    {
        throw MetricsInvalidInputDimension();
    }
    
    // since segmented images have 1 color label for each hand we convert them from BGR to GRAYSCALE and consider non-zero pixels as hand-pixels
    // if not CV_8UC3 cvtColor throws invalid type exception (to be handled in main file)
    Mat groundTruthImageGS;
    cvtColor(groundTruthImage, groundTruthImageGS, COLOR_BGR2GRAY);
    Mat segmentedImageGS;
    cvtColor(segmentedImage, segmentedImageGS, COLOR_BGR2GRAY);
    
    // computing number of hand pixels in the segmentedImage that are true hand pixels
    // means: number of non black pixels that are 255 in the groundTruthImage image
    int numberTrueHandPixels = 0;
    for(int i=0; i<segmentedImageGS.rows; i++)
    {
        for(int j=0; j<segmentedImageGS.cols; j++)
        {
            if(segmentedImageGS.at<uchar>(i,j) != 0 && groundTruthImageGS.at<uchar>(i,j) == 255)
            {
                numberTrueHandPixels++;
            }
        }
    }
    
    // number of total hand pixels in the segmented image (non black pixels)
    int numberHandPixelsSI = 0;
    for(int i=0; i<segmentedImageGS.rows; i++)
    {
        for(int j=0; j<segmentedImageGS.cols; j++)
        {
            if(segmentedImageGS.at<uchar>(i,j) != 0)
            {
                numberHandPixelsSI++;
            }
        }
    }
    
    return (static_cast<double>(numberTrueHandPixels)/numberHandPixelsSI) * 100;
    
}


double pixelAccuracyNonHand(const Mat& groundTruthImage, const Mat& segmentedImage)
{
    // if different sizes throw Exception
    if(groundTruthImage.rows != segmentedImage.rows || groundTruthImage.cols != segmentedImage.cols)
    {
        throw MetricsInvalidInputDimension();
    }
    
    // since segmented images have 1 color label for each hand we convert them from BGR to GRAYSCALE and consider non-zero pixels as hand-pixels
    // if not CV_8UC3 cvtColor throws invalid type exception (to be handled in main file)
    Mat groundTruthImageGS;
    cvtColor(groundTruthImage, groundTruthImageGS, COLOR_BGR2GRAY);
    Mat segmentedImageGS;
    cvtColor(segmentedImage, segmentedImageGS, COLOR_BGR2GRAY);
    
    // number of black pixels in the segmented image that are black also in the ground truth image (true non hand pixels)
    int numberTrueNonHandPixels = 0;
    for(int i=0; i<segmentedImageGS.rows; i++)
    {
        for(int j=0; j<segmentedImageGS.cols; j++)
        {
            if(segmentedImageGS.at<uchar>(i,j) == 0 && groundTruthImageGS.at<uchar>(i,j) == 0)
            {
                numberTrueNonHandPixels++;
            }
        }
    }
    
    // total number of NON hand pixel in the segmented image (black pixels)
    int numberNonHandPixelsSI = 0;
    for(int i=0; i<segmentedImageGS.rows; i++)
    {
        for(int j=0; j<segmentedImageGS.cols; j++)
        {
            if(segmentedImageGS.at<uchar>(i,j) == 0)
            {
                numberNonHandPixelsSI++;
            }
        }
    }
    
    return (static_cast<double>(numberTrueNonHandPixels)/numberNonHandPixelsSI) * 100;
   
}
