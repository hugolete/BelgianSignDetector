from ultralytics import YOLO
import os
import cv2

imgsz = 640
conf_level = 0.5
image_folder_path = "../datasets/Dataset test/"
output_folder = "test_shapedetector"
video_path =  "../datasets/archive/traffic-sign-to-test.mp4"

model_path = "../models/ShapeDetector_Kaggle_Epoch20.pt"
model = YOLO(model_path)

count_detections = 0

#sur vidéo
"""results = model(video_path)
results[0].show()"""

# sur plusieurs images
for file in os.listdir(image_folder_path):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        print("Image : ", file)
        path = os.path.join(image_folder_path, file)

        results = model(path, conf=conf_level, imgsz=imgsz)

        img = results[0].plot()

        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, file)
        cv2.imwrite(save_path, img)

        print("Done:", file)
        print(file, "detections:", len(results[0].boxes))

        count_detections += len(results[0].boxes)


print(count_detections)
