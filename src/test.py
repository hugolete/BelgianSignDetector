from ultralytics import YOLO
import os
import cv2


model_path = "../models/FinalModel.pt"
image_folder_path = "../datasets/Dataset test/"

model = YOLO(model_path)
output_folder = "predictions_test"

os.makedirs(output_folder, exist_ok=True)

detected_signs = 0

for file in os.listdir(image_folder_path):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        print("Image : ", file)
        path = os.path.join(image_folder_path, file)

        results = model(path, conf=0.1, imgsz=640)

        img = results[0].plot()

        save_path = os.path.join(output_folder, file)
        cv2.imwrite(save_path, img)

        print("Done:", file)
        print(file, "detections:", len(results[0].boxes))
        detected_signs += len(results[0].boxes)

print("Detected signs : ",detected_signs)
