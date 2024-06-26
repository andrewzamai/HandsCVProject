// Author: Andrew Zamai 2038522

#include "metricsCalculation.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

// Some random tests

int main()
{
    // IOU TESTING
    /*
    // Single pair GroundTruth BB - Predicted BB
    vector<int> TBBCoordinates{631, 318, 217, 122};
    vector<int> PBBCoordinates{651, 210, 200, 150};
    
    double iou = intersectionOverUnionSingleBB(Mat(2000, 2000, CV_8U), TBBCoordinates, PBBCoordinates);
    cout << iou << '\n';
    
    // Multiple Bounding boxes on image test
    vector<int> TBBCoordinates2{651, 338, 237, 142};
    vector<int> PBBCoordinates2{671, 230, 220, 170};
    
    vector<vector<int>> trueBoundingBoxes;
    trueBoundingBoxes.push_back(TBBCoordinates);
    trueBoundingBoxes.push_back(TBBCoordinates2);
    
    vector<vector<int>> predictedBoundingBoxes;
    predictedBoundingBoxes.push_back(PBBCoordinates);
    predictedBoundingBoxes.push_back(PBBCoordinates2);
    
    double iouMultiple = intersectionOverUnion(Mat(2000, 2000, CV_8U), trueBoundingBoxes, predictedBoundingBoxes);
    cout << iouMultiple << '\n';
    */
    // Pixel Accuracy tests for hand segmentation task
    
    String groundTruthImagePath = "../../Miscellanea/MetricsTestImages/07.png";
    Mat gtImage = imread(groundTruthImagePath, IMREAD_GRAYSCALE);
    namedWindow("GroundTruth Image");
    imshow("GroundTruth Image", gtImage);
    
    String segmentedImagePath = "../../Miscellanea/MetricsTestImages/07Segmented.jpg";
    Mat segmentedImage = imread(segmentedImagePath, IMREAD_GRAYSCALE);
    namedWindow("Segmented Image");
    imshow("Segmented Image", segmentedImage);
    
    Mat segmentedImageBGR;
    cvtColor(segmentedImage, segmentedImageBGR, COLOR_GRAY2BGR);
    
    
    double pixelAccuracy = accuracyMetricForSegmentation(gtImage, segmentedImageBGR);
    double pixelHandPrecision = pixelPrecisionHand(gtImage, segmentedImageBGR);
    double pixelNonHandPrecision = pixelPrecisionNonHand(gtImage, segmentedImageBGR);
    
    cout << "Pixel accuracy: " << pixelAccuracy << endl;
    cout << "Hand pixels precision: " << pixelHandPrecision << endl;
    cout << "Non Hand pixels precision: " << pixelNonHandPrecision << endl;
    
    waitKey(0);

    return 0;
}
