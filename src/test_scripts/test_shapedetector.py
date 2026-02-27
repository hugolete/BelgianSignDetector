from ultralytics import YOLO
import os
import cv2

imgsz = 640
conf_level = 0.5
image_folder_path = "../../datasets/Dataset test/"
output_folder = "test_shapedetector"
video_path = "../../datasets/archive/traffic-sign-to-test.mp4"
video_path2 = "../../datasets/video1_nuit.mp4"

model_path = "../../models/ShapeDetector_Kaggle_Epoch20.pt"
model = YOLO(model_path)

count_detections = 0

#sur vidéo
print("Début prédiction")
results = model.predict(video_path2, stream=True, save=True,project="InferenceVideoTest3")
print("Fin prédiction")

for _ in results:
    pass

"""# sur plusieurs images
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


print(count_detections)"""
