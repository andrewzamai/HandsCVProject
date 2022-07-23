The subdirectory named "models" contains the Jupyter notebooks that were used to define and train the model for the task of hands segmentation (more in the project report). The trained model is saved in this directory as well in HDF5 format. 

As explained in the report the model could not be converted to be embedded directly in the C++ pipeline. The file "segmentImageUnetModel.py" is used thus to load the model and run a forward prediction step on a provided (vector of) image(s). 

The handsSegmentationFunctions contains the functions that load, preprocess and produce the segmentation predictions by the model by calling the above python script.

More post-processing operations in the other file.

Finally handsSegmentationMain defines the pipeline to load, segment and save all files from a given directory of images.

Metrics are computed by a separated main.


