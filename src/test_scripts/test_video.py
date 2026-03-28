from ultralytics import YOLO
import cv2


def video_shape_detection_7fps(shapeDetector_path:str,video_path:str):
    shapeDetector = YOLO(shapeDetector_path)

    max_detections_per_sec = 7

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
            #print(results[0].boxes)
            #results[0].show()
            #print("Box id : ",results[0].boxes.id)
            track_ids = results[0].boxes.id.cpu().numpy()
            #print(track_ids)

            coordinates = results[0].boxes.xyxy.cpu().numpy()
            #print(coordinates)

            """for id_detec in track_ids:
                print(id_detec)
                box_coords = coordinates[int(id_detec) - 1]
                print(box_coords)"""

            for id_detec, box_coords in zip(track_ids, coordinates):
                print(id_detec)
                print(box_coords)

        frame_count += 1

    vid.release() # fermeture de la vidéo


def video_shape_detection_basic(shapeDetector_path:str,video_path:str):
    shapeDetector = YOLO(shapeDetector_path)
    results = shapeDetector.predict(video_path, stream=True)

    for _ in results:
        print(_)


if __name__ == '__main__':
    shapeDetector_path = "../models/ShapeDetector.pt"
    video_path = "../../datasets/archive/traffic-sign-to-test.mp4"
    video_path2 = "../../datasets/video1_nuit.mp4"
    video_path3 = "../../datasets/video2.mp4"

    #video_shape_detection_7fps(shapeDetector_path, video_path)
    video_shape_detection_basic(shapeDetector_path, video_path)
