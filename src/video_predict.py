from ultralytics import YOLO


def video_shape_detection(shapeDetector_path:str,video_path:str):
    shapeDetector = YOLO(shapeDetector_path)
    results = shapeDetector.predict(video_path, stream=True)

    for _ in results:
        print(_)


if __name__ == '__main__':
    shapeDetector_path = "../models/ShapeDetector_Kaggle_Epoch20.pt"
    video_path = "../../datasets/archive/traffic-sign-to-test.mp4"
    video_path2 = "../../datasets/video1_nuit.mp4"

    video_shape_detection(shapeDetector_path, video_path)
