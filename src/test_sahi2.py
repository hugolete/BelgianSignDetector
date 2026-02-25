from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os

detector = AutoDetectionModel()

detection_model = detector.from_pretrained(
    model_type='ultralytics',
    model_path='../models/Augmentation_Epoch6.pt',
    confidence_threshold=0.3,
    device="cuda:0"
)

image_folder_path = "../datasets/Dataset test/"

total_detections = 0

for file in os.listdir(image_folder_path):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        print("Image : ", file)
        path = os.path.join(image_folder_path, file)
        detections = 0

        result = get_sliced_prediction(
            path,
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
            postprocess_type="NMS",
            postprocess_match_threshold=0.5,
            perform_standard_pred=True
        )

        result.export_visuals(export_dir="outputs/",file_name=file)
        #print(result.object_prediction_list)

        detections_on_image = len(result.object_prediction_list)
        detections += detections_on_image
        total_detections += detections_on_image

        for prediction in result.object_prediction_list:
            print(prediction.category.name)

        print("Total détecté : ",detections)

print("Total détecté sur toutes les images : ",total_detections)
