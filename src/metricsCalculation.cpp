// Author: Andrew Zamai
#include "metricsCalculation.hpp"

#include <iostream>
#include <math.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;


double intersectionOverUnionSingleBB(const Mat& image, vector<int> trueBoundingBox, vector<int> predictedBoundingBox)
{
    int xTB = trueBoundingBox[0];
    int yTB = trueBoundingBox[1];
    int widthTB = trueBoundingBox[2];
    int heightTB = trueBoundingBox[3];
    
    int xPB = predictedBoundingBox[0];
    int yPB = predictedBoundingBox[1];
    int widthPB = predictedBoundingBox[2];
    int heightPB = predictedBoundingBox[3];
    
    // TODO: check not out of image?
    
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


double intersectionOverUnion(const Mat& image, vector<pair<vector<int>,vector<int>>> trueAndPredictedBoundingBoxes)
{
    double iouTotal = 0;
    for(int i=0; i<trueAndPredictedBoundingBoxes.size(); i++)
    {
        double iou = intersectionOverUnionSingleBB(image, trueAndPredictedBoundingBoxes[i].first, trueAndPredictedBoundingBoxes[i].second);
        iouTotal += iou;
    }
    
    return iouTotal;
}

              
double pixelAccuracyHand(const Mat& groundTruthImage, const Mat& segmentedImage)
{
    // convert them to GRAYSCALE if not already
    // TODO: check if already grayscale, check if all went correctly
    Mat groundTruthImageGS;
    cvtColor(groundTruthImage, groundTruthImageGS, COLOR_BGR2GRAY);
    
    Mat segmentedImageGS;
    cvtColor(segmentedImage, segmentedImageGS, COLOR_BGR2GRAY);
    
    // we can now assume images are in  Grayscale (uchar type)
    
    // number of hand pixels in the segmentedImage that are true hand pixels
    // means number of non black pixels (since multicolor) that are white in the groundTruthImage image
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
    
    // number of hand pixel in the segmented image (non black pixels)
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
    
    double pixelAccuracyPercent = (static_cast<double>(numberTrueHandPixels)/numberHandPixelsSI) * 100;
    
    return pixelAccuracyPercent;
}


double pixelAccuracyNonHand(const Mat& groundTruthImage, const Mat& segmentedImage)
{
    // convert them to GRAYSCALE if not already
    //TODO: check if already grayscale, check if all went correctly
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
    
    double pixelAccuracyPercent = (static_cast<double>(numberTrueNonHandPixels)/numberNonHandPixelsSI) * 100;
    
    return pixelAccuracyPercent;

}
