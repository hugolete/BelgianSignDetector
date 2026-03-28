from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detector = AutoDetectionModel()

detection_model = detector.from_pretrained(
    model_type='ultralytics',
    model_path='../../models/SignDetector.pt',
    confidence_threshold=0.3
)

result = get_sliced_prediction(
    "../datasets/Dataset test/2.jpg",
    detection_model,
    slice_height=320,
    slice_width=320,
    overlap_height_ratio=0.3,
    overlap_width_ratio=0.3,
    postprocess_type="NMS",
    postprocess_match_threshold=0.5
)

result.export_visuals(export_dir="../outputs/")
