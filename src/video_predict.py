from ultralytics import YOLO
import cv2


def video_shape_detection(shapeDetector_path:str,video_path:str):
    shapeDetector = YOLO(shapeDetector_path)

    detected_ids = set()
    crops_list = []
    boxId_to_class = {}
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
            #print(results[0].boxes)

            # récupération résultats
            boxes = results[0].boxes

            if boxes.id is not None:
                class_ids = boxes.cls.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                track_ids = boxes.id.cpu().numpy()
                coords = boxes.xyxy.cpu().numpy()
                print(class_ids, confidences,coords)

                for box_id, box_coords in zip(track_ids,coords):
                    print("Id boite : ",box_id)

                    if box_id not in detected_ids:
                        # on croppe et on envoie au modèle expert
                        x1, y1, x2, y2 = coords.tolist()
                        cropped_frame = crop_sign(frame,box_coords)
                        crops_list.append({
                            "image": cropped_frame,
                            "coords_orig": [int(x1), int(y1), int(x2), int(y2)],
                        })

                    detected_ids.add(box_id)

            #test
            """annotated_frame = results[0].plot()
            cv2.imshow("Tracking", annotated_frame)
            cv2.waitKey(0)"""

        frame_count += 1

    vid.release()  # fermeture de la vidéo


def crop_sign(frame,coords,padding=10):
    print("Début crop")
    h0, w0 = frame.shape[:2]  # récup de la taille réelle
    crops_list = []

    x1, y1, x2, y2 = coords.tolist()

    # application du padding pour ne pas risquer un bug
    ix1 = max(0, int(x1) - padding)
    iy1 = max(0, int(y1) - padding)
    ix2 = min(w0, int(x2) + padding)
    iy2 = min(h0, int(y2) + padding)

    # découpage Numpy
    cropped_frame = frame[iy1:iy2, ix1:ix2]

    return cropped_frame


if __name__ == '__main__':
    shapeDetector_path = "../models/ShapeDetector_Kaggle_Epoch20.pt"
    video_path = "../datasets/archive/traffic-sign-to-test.mp4"
    video_path2 = "../datasets/video1_nuit.mp4"

    video_shape_detection(shapeDetector_path, video_path)
