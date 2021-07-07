### This is the PyTorch implementation of paper Real-time Pupil Tracking from Monocular Video for Digital Puppetry (https://arxiv.org/pdf/2006.11341)
![](https://google.github.io/mediapipe/images/mobile/iris_tracking_example.gif)

This version doesn't have BatchNorm layers for fine-tuning. If you want to use such model for training, you should add these layers manually.

I've made the conversion semi-manually using a similar method as available for both the BlazeFace PyTorch implementation and the FaceMesh, seen here:
- https://github.com/hollance/BlazeFace-PyTorch
- https://github.com/thepowerfuldeez/facemesh.pytorch


#### Input for the model is expected to be cropped iris image normalized to -1.0 to 1.0. The cropped image should be 60x60 and centered at the center of the eye contour points as given by FaceMesh.
To get the right scaling, simply use the 192x192 cropped face image used as input for the FaceMesh model, get the average eye contour position for each eyes, use a rect of 64x64 centered at the average position of one eye and use it to crop from the 192x192 image.

However, `predict_on_image` function normalizes your image itself, so you can even treat resized image as np.array as input

See Inference-IrisLandmarks.ipynb notebook for usage example.
See Convert-Iris.ipynb notebook for conversion example.

All other files were just me figuring stuff out. You can take a look at the (very rough) code I used if you're trying something similar.