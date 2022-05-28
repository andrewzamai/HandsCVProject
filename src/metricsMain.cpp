// Author: Andrew Zamai

#include "metricsCalculation.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <iostream>


using namespace cv;
using namespace std;
 
// TODO: change requiring paths to folder?
int main()
{
    // testing IOU
    vector<int> TBBCoordinates{631, 318, 217, 122};
    vector<int> PBBCoordinates{651, 210, 200, 150};
    
    double iou = intersectionOverUnionSingleBB(Mat(), TBBCoordinates, PBBCoordinates);
    cout << iou << '\n';
    
    vector<int> TBBCoordinates2{631, 318, 217, 122};
    vector<int> PBBCoordinates2{651, 210, 200, 150};
    
    vector<pair<vector<int>,vector<int>>> allBB;
    allBB.push_back(make_pair(TBBCoordinates, PBBCoordinates));
    allBB.push_back(make_pair(TBBCoordinates2, PBBCoordinates2));
    
    double iouMultiple = intersectionOverUnion(Mat(), allBB);
    cout << iouMultiple << '\n';
    
    
    
    // Ok correct!
    
    
    /*
    String groundTruthImagePath = "/Users/andrew/CVProjects/FinalProject/BenchmarkDataset/mask/07.png";
    
    Mat readedImg = imread(groundTruthImagePath, IMREAD_GRAYSCALE);
    namedWindow("GroundTruth Image");
    imshow("GroundTruth Image", readedImg);
    
    waitKey(0);
    */
    
    
    /*
    bool allLoadedCorrectly = true; // compute metrics only if all images are loaded correctly
    
    String groudTruthImagesPath = "../BenchmarkDataset/mask/*.png";
    String segmentedImagesPath = "../BenchmarkDataset/segmentedHands/*.png";  // are they saved png? same name as original image!
    
    vector<String> groundTruthImagesFilenames;
    glob(groudTruthImagesPath, groundTruthImagesFilenames, false);
    size_t gtiCount = groundTruthImagesFilenames.size(); //number of png files in mask folder
    
    vector<String> segmentedImagesFilenames;
    glob(segmentedImagesPath, segmentedImagesFilenames, false);
    int siCount = segmentedImagesFilenames.size(); //number of png files in segmentedHands folder
    
    //TODO: check the number to be equal!
    
    // Loading all images in 2 vectors
    vector<Mat> groudTruthImages;
    vector<Mat> segmentedImages;
    
    for(int i=0; i<gtiCount; i++)
    {
        // are they loaded in same order when calling glob?

        Mat readedImgMask = imread(groundTruthImagesFilenames[i]);
        Mat readedImgSegmented = imread(segmentedImagesFilenames[i]);
        
        if(readedImgMask.empty() == true || readedImgSegmented.empty() == true)
        {
            allLoadedCorrectly = false;
            break;
        }
        groudTruthImages.push_back(readedImgMask);
        segmentedImages.push_back(readedImgSegmented);
    }
     */
    
    /*
    // proceed only if all loaded correctly
    if(allLoadedCorrectly == true)
    {
    
    }
        */
        
    return 0;
}
