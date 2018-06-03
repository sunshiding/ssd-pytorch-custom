# Test and Train Data

Under `image_data` create folders called `test` and `train` and place appropriate image data there.

## VGG Image Annotator Instructions

In this initial set of code and instructions we are only considering _one class_,. The [VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/) tool was used to create the bounding boxes as json for the images in the folder (Note:  running the annotator locally is usually a bit faster than the online version).

1. Load all images from the appropriate folder into the VGG Image Annotator
2. Use tool to draw bounding boxes around all object of interest
3. Export as json
3. Place the `via_region_data.json` in a folder called `annot` under either `train` or `test` folder - this will be read by the `custom.py` script.  (In other words, each folder, `test` and `train` should have their own `annot` folder with a `via_region_data.json`)


Other annotator instructions coming soon.