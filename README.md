# Belgian Sign Detector

This project is a computer vision application aiming to automatically detect and recognize Belgian road signs from images, videos or real-time video streams under various conditions

## Detection pipeline

- Source : image, video or webcam
- The image (or the frame of the video/webcam feed) is first processed by the Shape Detector model => it will determine whether there's a sign or not
- If the shape detector detects a sign, it will extract its coordinates, crop it, and send it to the next model
- The next model is the Sign Detector, it will analyze the cropped image and attempt to detect which sign it is.
- Detected signs are shown in the console in real-time, with an anti-duplicate system, and added to a dictionary named "total_detected_signs"
- At the end of the video, a summary will show every sign detected with the frame number & the coordinates

## Original models used
- for the shape detector : yolo26n (https://docs.ultralytics.com/fr/models/yolo26/#key-features)
- for the sign detector : yolo26m

## Datasets used
- For the shape detector : https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format (Valentyn Sichkar)
- For the sign detector : https://btsd.ethz.ch/shareddata/ (ETH Zurich)

For more information about specific components, consult the MD files in the "docs" folder
