from src.video_predict import video_shape_detection
from utils.find_cameras import get_best_camera


if __name__ == '__main__':
    shapeDetector_path = "../models/ShapeDetector_Kaggle_Epoch20.pt"
    source = get_best_camera()

    video_shape_detection(shapeDetector_path, source)
