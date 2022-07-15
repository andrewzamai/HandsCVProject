from tensorflow import keras
import numpy as np
import sys
import cv2

# loading the model
unetVGG19Model = keras.models.load_model('./models/handSegmentationUnetVGG19_87.h5', compile=False)
    
def segmentImage(inImagePath):
    ''' load the pre-resized image '''
    inImage = cv2.imread(inImagePath)
    
    '''
    preprocess the input using preprocess_input function provided by keras when using VGG19 pretrained model as encoder backbone;
    since the preprocessing functions converts the image to RGB to BGR and each color channel is zero-centered with respect to the ImageNet dataset, without scaling,
    we can emulate this function also without having to convert to BGR, since opencv already reads images in this format.
    Looking at keras github we find the mean values are [103.939, 116.779, 123.68] for the BGR channels respectively, std = None.
    Scaling done here and not in c++ since in strictly model dependent.
    '''
    inImage = inImage.astype('float32', copy=False)
    inImage[:,:,0] -= 103.939
    inImage[:,:,1] -= 116.779
    inImage[:,:,2] -= 123.68
    
    prediction = unetVGG19Model.predict(inImage.reshape(1,256,256,3))
    cv2.imwrite(inImagePath, prediction[0]*255)
    
    
def segmentAllImages(numberOfImages):
    numberOfImages = int(numberOfImages)
    for i in range(0, numberOfImages):
        segmentImage("./" + str(i) + ".jpg")


if __name__ == '__main__':
    globals()[sys.argv[1]](sys.argv[2])
