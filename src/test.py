from ultralytics import YOLO
import os
import cv2


image_folder_path = "../datasets/Dataset test/"
output_folder = "predictions_test"

os.makedirs(output_folder, exist_ok=True)

model_list = [
    "Augmentation_5epoch_best.pt",
    "Augmentation_8epochs_best.pt",
    "Augmentation_10epochs_best.pt",
    "Augmentation_Epoch5.pt",
    "Augmentation_Epoch6.pt"
    "Augmentation_Epoch8.pt",
    "FinalModel.pt"
]

for model_name in model_list:
    model_path = f"../models/{model_name}"

    model = YOLO(model_path)

    for file in os.listdir(image_folder_path):
        detected_signs = 0

        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            print("Image : ", file)
            path = os.path.join(image_folder_path, file)

            results = model(path, conf=0.1, imgsz=640)

            img = results[0].plot()

            os.makedirs(f"{output_folder}/{model_name}", exist_ok=True)
            save_path = os.path.join(output_folder, model_name,file)
            cv2.imwrite(save_path, img)

            print("Done:", file)
            print(file, "detections:", len(results[0].boxes))
            detected_signs += len(results[0].boxes)

        print("Detected signs : ",detected_signs)

        with open(f'{output_folder}/{model_name}.txt', 'a') as f:
            f.write(f"Detected signs : {detected_signs}\n")
