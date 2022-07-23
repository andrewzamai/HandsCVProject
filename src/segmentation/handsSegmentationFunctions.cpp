// Author: Andrew Zamai 2038522

#include "handsSegmentationFunctions.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include <exception>

using namespace std;
using namespace cv;


// Exception to be thrown when some image has not been correctly processed during segmentation
class SegmentationException : public exception
{
    virtual const char* what() const throw()
    {
      return "Something went wrong when segmenting the images.";
    }
}segmentationException;


void segmentImages(const vector<Mat>& inImages, vector<Mat>& outImages)
{
    // for each image we save the original size before resizing it to 256 x 256, in order to be able to resize them back to original size
    vector<Size> originalSizes;
    for(int i=0; i<inImages.size(); i++)
    {
        originalSizes.push_back(inImages[i].size());
    }
    
    // resizing them
    vector<Mat> resizedInImages;
    for(int i=0; i<inImages.size(); i++)
    {
        Mat resizedInImage;
        resize(inImages[i], resizedInImage, Size(256,256), INTER_AREA);
        resizedInImages.push_back(resizedInImage);
    }
    
    // checking if some image was lost
    if (resizedInImages.size() != inImages.size())
        throw SegmentationException();
    
    // saving them in order to pass them easily to the python script containing the model
    for(int i=0; i<resizedInImages.size(); i++)
    {
        // temporary file names
        string savingPathName = string("./") + to_string(i) + string(".jpg");
        imwrite(savingPathName, resizedInImages[i]);
    }
    
    // calling the python script passing the number of images to be segmented
    // by convention the names of the passsed images are the number in the range (0, numberOfImages).jpg
    int numberOfImages = inImages.size();
    string command = string("python3 ../src/segmentation/segmentImageUnetModel.py segmentAllImages ") + to_string(numberOfImages);
    
    try {
        system(command.c_str());
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;
        throw SegmentationException();
    }
    
    
    // resize predictions back to original sizes
    vector<Mat> resizedPredictions;
    for(int i=0; i<numberOfImages; i++)
    {
        string predictionPath = string("./") + to_string(i) + string(".jpg");
        Mat prediction = imread(predictionPath, IMREAD_GRAYSCALE);
        // delete temporary image
        remove(predictionPath.c_str());
        
        Mat resizedPrediction;
        resize(prediction, resizedPrediction, originalSizes[i], INTER_CUBIC);
        resizedPredictions.push_back(resizedPrediction);
    }
    
    outImages = resizedPredictions;

}
