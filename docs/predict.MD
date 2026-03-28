# Predict

The prediction process is detailed in README.md

## Image prediction

Prerequisites : having the shape detector & the sign detector models in the /models folder
Default paths & names : models/ShapeDetector.pt & models/SignDetector.pt

### Parameters
- Image path

### How to use
- Simply run image_predict.py and type the required parameters

## Video prediction

Prerequisites : same as image

### Parameters
- Video path

### How to use
- Simply run video_predict.py and type the required parameters

## Camera prediction

Prerequisites : same as image & video + you need a camera

This feature was only tested with OBS virtual camera as I don't have a real one, so problems may happen

### How to use
- Plug in a camera
- Run camera_predict.py
