import cv2
import os
from src.image_predict import shape_detection, sign_detection, get_crops, get_detected_signs


if __name__ == '__main__':
    shapeDetector_path = "../../models/ShapeDetector_Kaggle_Epoch20.pt"
    signDetector_path = "../../models/FinalModel.pt"

    image_folder_path = "../../datasets/Dataset test/"

    for file in os.listdir(image_folder_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            print("File : ", file)

            image_path = os.path.join(image_folder_path, file)
            img = cv2.imread(image_path)

            final_boxes = shape_detection(shapeDetector_path, img)
            cropped_signs = get_crops(img,final_boxes)
            final_signs = sign_detection(signDetector_path,cropped_signs)
            detected_signs = get_detected_signs(final_signs)

            for sign in detected_signs:
                print(f"Panneau détecté : {sign} | {detected_signs[sign]} détections")
