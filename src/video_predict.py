from ultralytics import YOLO
import cv2


def video_shape_detection(shapeDetector_path:str,video_path:str):
    shapeDetector = YOLO(shapeDetector_path)

    detected_ids = set()
    final_results = []
    max_detections_per_sec = 4

    # ouverture vidéo pour compter fps
    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)

    skip_frames = int(fps / max_detections_per_sec)
    print(f"FPS : {fps}. Inférence sur 1 image toutes les {skip_frames} frames.")
    frame_count = 0

    while vid.isOpened():
        success, frame = vid.read()

        if not success:
            break

        if frame_count % skip_frames == 0:
            results = shapeDetector.track(frame, persist=True)

            print(f"Inférence faite sur la frame {frame_count}")

            #test
            annotated_frame = results[0].plot()
            print(results[0])
            cv2.imshow("Tracking", annotated_frame)
            cv2.waitKey(0)

        frame_count += 1

    vid.release()  # fermeture de la vidéo


if __name__ == '__main__':
    shapeDetector_path = "../models/ShapeDetector_Kaggle_Epoch20.pt"
    video_path = "../datasets/archive/traffic-sign-to-test.mp4"
    video_path2 = "../datasets/video1_nuit.mp4"

    video_shape_detection(shapeDetector_path, video_path)
