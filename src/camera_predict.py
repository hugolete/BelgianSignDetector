from src.video_predict import video_shape_detection
from utils.find_cameras import get_best_camera


if __name__ == '__main__':
    shapeDetector_path = "../models/ShapeDetector.pt"
    signDetector_path = "../models/SignDetector.pt"
    source = get_best_camera()

    video_shape_detection(shapeDetector_path, signDetector_path, source)
