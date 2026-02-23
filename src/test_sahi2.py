from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os

detector = AutoDetectionModel()

detection_model = detector.from_pretrained(
    model_type='ultralytics',
    model_path='../models/FinalModel.pt',
    confidence_threshold=0.3,
    device="cuda:0"
)

image_folder_path = "../datasets/Dataset test/"

for file in os.listdir(image_folder_path):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        print("Image : ", file)
        path = os.path.join(image_folder_path, file)

        result = get_sliced_prediction(
            path,
            detection_model,
            slice_height=320,
            slice_width=320,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
            postprocess_type="NMS",
            postprocess_match_threshold=0.5
        )

        result.export_visuals(export_dir="outputs/",file_name=file)
        print(result)
