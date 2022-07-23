# HandsCVProject
CV system for human hands detection and segmentation by Niko Picello, Filippo Steffenel & Andrew Zamai

Create a build directory and use the CMakeLists file to build the project. Please find the executables then in the build directory created. 


NB: to run on Taliercio 2020 pip install tensorflow and pip install opencv-python. 
Opencv for python is used to load the preprocessed images and run the inference steps to make the segmentations, as the model could not be embedded directly in the c++ pipeline (more in the project report). 
