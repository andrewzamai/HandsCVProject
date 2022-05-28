//Author: FILIPPO STEFFENEL
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;
 

 
int main(int argc, char** argv){
    
    String set = argv[1];
    // Path of the folder containing images
    String pathInput = "../archive/train/train/"+set+"/segmentation/*.png";
    vector<String> fn;
	glob(pathInput, fn, false);
	size_t count = fn.size(); //number of png files in images folder


	// Defining the world coordinates for 3D points
    vector<Mat> maskWithColor;

	for (size_t i = 0; i < count; i++){
    	maskWithColor.push_back(imread(fn[i]));
	}

    for(int i = 0; i < maskWithColor.size(); i++){
    	cvtColor(maskWithColor[i], maskWithColor[i], COLOR_BGR2GRAY);
    	cout << i << "/" << maskWithColor.size()-1 << "\n";
        for(int j = 0; j < maskWithColor[i].cols; j++){
        	for(int y = 0; y < maskWithColor[i].rows; y++){
        		if(maskWithColor[i].at<uchar>(y, j) != 0){
        			maskWithColor[i].at<uchar>(y, j) = 255; //all pixel that are not black become white
        		}
        	}
        }
    }
   
    //the number of char in the name of the image
    int charInName = 10;
    // Path of the folder where we put the images
    String pathOutput = "../archive/train/train/"+set+"/segmentationBW/";
   	for(int i = 0; i < maskWithColor.size(); i++){

   		String name = pathOutput + fn[i].substr(fn[i].size()-charInName);
   		imwrite(name, maskWithColor[i]);
   	}

    return 0;
}

