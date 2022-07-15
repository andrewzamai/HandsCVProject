// Author: Andrew Zamai 2038522

#include "handsSegmentationFunctions.hpp" // my defined functions
#include "../metrics/metricsCalculation.hpp" // my defined metrics

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;
 

int main()
{
    // retrieving all filepaths of the benchmark images to be segmented
    string benchmarkRGBImagesFolderPath = "../../BenchmarkDataset/rgb/*.jpg";
    vector<String> filenames;
    glob(benchmarkRGBImagesFolderPath, filenames, false);
    
    vector<Mat> inImages; // reading all images into a vector
    vector<string> originalNames; // storing the original names
    for(int i=0; i<filenames.size(); i++)
    {
        Mat inImage = imread(filenames[i]);
        // if(inImage.empty() == true) throw error
        originalNames.push_back(filenames[i]);
        inImages.push_back(inImage);
    }
    
    // segmenting all images, storing them into a vector
    vector<Mat> segmentedImages;
    try {
        segmentImages(inImages, segmentedImages);
    }
    catch(const exception& se){
        cerr << se.what() << endl;
    }
    
    // saving segmented images with original name into segmentedByUs directory
    string segmentedImagesPath = "../../BenchmarkDataset/segmentedByUs/";
    for(int i=0; i<segmentedImages.size(); i++)
    {
        string originalName = originalNames[i];
        size_t found = originalName.rfind(string("/"));
        string path = segmentedImagesPath + originalName.substr(found+1);
        
        imwrite(path, segmentedImages[i]);
    }
    
    // TODO: some POST PROCESSING + coloring, by Filippo
    
    // eg. thresholding
    vector<Mat> thresholdedSegmentations;
    for(int i=0; i<segmentedImages.size(); i++)
    {
        Mat thresholded;
        threshold(segmentedImages[i], thresholded, 215, 255, THRESH_BINARY); // 215 is like 0.85 probability that pixel belongs to class hand
        thresholdedSegmentations.push_back(thresholded);
        
        string originalName = originalNames[i];
        size_t found = originalName.rfind(string("/"));
        string path = segmentedImagesPath + originalName.substr(found+1);
        imwrite(path, thresholded);
    }
    
    
    // computing metrics of segmentation task
    
    string maskImagesPath = "../../BenchmarkDataset/mask/";
    double totalHand = 0;
    double totalNonHand = 0;
    
    for(int i=0; i<thresholdedSegmentations.size(); i++)
    {
        // load mask ground truth
        string originalName = originalNames[i];
        size_t found = originalName.rfind(string("/"));
        string path = maskImagesPath + originalName.substr(found+1, 2) + string(".png"); // if file name >2 doesn't work
        
        // convert to 3 channels
        Mat mask = imread(path, IMREAD_GRAYSCALE);
        Mat groundTruthImageBGR;
        cvtColor(mask, groundTruthImageBGR, COLOR_GRAY2BGR);
        
        // convert to 3 channels, when hands will be colored they will be already CV_8UC3
        Mat threshSegmentBGR;
        cvtColor(thresholdedSegmentations[i], threshSegmentBGR, COLOR_GRAY2BGR);
        
        totalHand += pixelAccuracyHand(groundTruthImageBGR, threshSegmentBGR);
        totalNonHand += pixelAccuracyNonHand(groundTruthImageBGR, threshSegmentBGR);
    }
    
    double IOUHandMean = totalHand/thresholdedSegmentations.size();
    cout.precision(2);
    cout << "Pixel accuracy class Hand: " << IOUHandMean << endl;
    double IOUNonHandMean = totalNonHand/thresholdedSegmentations.size();
    cout << "Pixel accuracy class Non Hand: " << IOUNonHandMean << endl;
    
    
    return 0;
}
