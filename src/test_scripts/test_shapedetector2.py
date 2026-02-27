from src.image_predict import shape_detection
import os


if __name__ == "__main__":
    shapeDetector_path = "../../models/ShapeDetector_Kaggle_Epoch20.pt"
    image_folder_path = "../../datasets/Dataset test/"

    for file in os.listdir(image_folder_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(image_folder_path, file)
            print("File : ", file)

            shape_detection(shapeDetector_path, image_path)
