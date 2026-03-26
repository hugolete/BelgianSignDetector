from ultralytics import YOLO
import cv2
from src.image_predict import sign_detection, get_detected_signs, print_detections


def video_shape_detection(shapeDetector_path:str,signDetector_path:str,video_path:str,test:bool=False):
    shapeDetector = YOLO(shapeDetector_path)
    detected_ids = set()
    max_detections_per_sec = 7
    total_detected_signs = []
    min_size = 20

    # vérification : vidéo ou webcam
    is_webcam = False

    # si la source est un int ou un nombre sous forme de string => webcam, c'est reconnu
    if isinstance(video_path, int):
        is_webcam = True
    elif isinstance(video_path, str) and video_path.isdigit():
        is_webcam = True

    # ouverture vidéo/webcam
    if is_webcam:
        vid = cv2.VideoCapture(int(video_path), cv2.CAP_DSHOW)
    else:
        vid = cv2.VideoCapture(video_path)

    fps = vid.get(cv2.CAP_PROP_FPS)

    # parfois avec la cam : fps non reconnu, on passe a 30 par défaut
    if fps <= 0:
        fps = 30

    skip_frames = int(fps / max_detections_per_sec)
    print(f"FPS : {fps}. Inférence sur 1 image toutes les {skip_frames} frames.")
    frame_count = 0

    while vid.isOpened():
        success, frame = vid.read()

        if not success:
            break

        if frame_count % skip_frames == 0:
            results = shapeDetector.track(frame, persist=True,verbose=False)
            crops_list = []

            #print(f"Frame {frame_count+1}")
            # récupération résultats
            boxes = results[0].boxes

            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy()
                coords = boxes.xyxy.cpu().numpy()
                #print("Track ids : ",track_ids)

                current_frame_positions = []

                for box_id, box_coords in zip(track_ids,coords):
                    if box_id not in detected_ids:
                        # on croppe et on envoie au modèle expert
                        x1, y1, x2, y2 = box_coords.tolist()
                        width = x2 - x1
                        height = y2 - y1
                        #print("Width | Height : ",width," | Height : ",height)

                        if width >= min_size and height >= min_size:
                            if not is_duplicate(box_coords, frame_count, total_detected_signs+current_frame_positions):
                                cropped_frame = crop_sign(frame,box_coords)

                                """try:
                                    print("Save crop")
                                    formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                                    cv2.imwrite(f"crops/crop_{formatted_time}.jpg", cropped_frame)
                                except Exception as e:
                                    print("Erreur crop : ",e)"""

                                crops_list.append({
                                    "image": cropped_frame,
                                    "coords_orig": [int(x1), int(y1), int(x2), int(y2)],
                                })
                                current_frame_positions.append({
                                    "position": [int(x1), int(y1), int(x2), int(y2)],
                                    "frame": frame_count
                                })

                                detected_ids.add(box_id)
                        else:
                            pass

                    """annotated_frame = results[0].plot()
                    cv2.imshow("Tracking", annotated_frame)
                    cv2.waitKey(0)"""

                # détection sur image cropée & affichage console
                cropped_signs = sign_detection(signDetector_path,crops_list)
                if cropped_signs:
                    detected_signs = get_detected_signs(cropped_signs)
                    print_detections(detected_signs,0.40)

                    for label, detections in detected_signs.items():
                        for d in detections:
                            if not is_duplicate(d['box'],frame_count,total_detected_signs,label_to_check=label):
                                if d['confidence'] > 0.40:
                                    total_detected_signs.append({
                                        "label": label,
                                        "position": d['box'],
                                        "conf": d['confidence'],
                                        "frame": frame_count
                                    })
                            else:
                                print(f"Double détecté pour {label}, ignoré.")

            #test
            """annotated_frame = results[0].plot()
            cv2.imshow("Tracking", annotated_frame)
            cv2.waitKey(0)
            cv2.imshow("Tracking", frame)
            cv2.waitKey(0)"""

        frame_count += 1

    vid.release()  # fermeture de la vidéo

    print("Panneaux détectés sur la vidéo : ")
    for sign in total_detected_signs:
        if sign['conf'] > 0.40:
            print(f"Panneau : {sign['label']} -> Position : {sign['position']} | Confiance: {sign['conf']} | Frame : {sign['frame']}")

    if test:
        return total_detected_signs


def crop_sign(frame,coords):
    h0, w0 = frame.shape[:2]  # récup de la taille réelle
    crops_list = []

    x1, y1, x2, y2 = coords.tolist()
    width = x2 - x1
    padding = int(width*0.1)

    # application du padding pour ne pas risquer un bug
    ix1 = max(0, int(x1) - padding)
    iy1 = max(0, int(y1) - padding)
    ix2 = min(w0, int(x2) + padding)
    iy2 = min(h0, int(y2) + padding)

    # découpage Numpy
    cropped_frame = frame[iy1:iy2, ix1:ix2]
    """cv2.imshow("Tracking crop", cropped_frame)
    cv2.waitKey(0)"""

    return cropped_frame


def is_duplicate(new_coords,new_frame,sign_list,pos_threshold=60,frame_threshold=20,label_to_check=None):
    nx1, ny1, nx2, ny2 = new_coords
    new_center = ((nx1 + nx2) / 2, (ny1 + ny2) / 2)

    for sign in sign_list:
        if label_to_check is not None:
            if sign.get('label') != label_to_check:
                continue  # ce n'est pas le même type

        ox1, oy1, ox2, oy2 = sign['position']
        old_center = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)  # tuple

        dist_x = abs(new_center[0] - old_center[0])
        dist_y = abs(new_center[1] - old_center[1])
        frame_diff = abs(new_frame - sign['frame'])

        if (dist_x < pos_threshold and dist_y < pos_threshold) and (frame_diff < frame_threshold):
            return True

    return False


if __name__ == '__main__':
    shapeDetector_path = "../models/ShapeDetector_Kaggle_Epoch20.pt"
    video_path = "../datasets/archive/traffic-sign-to-test.mp4"
    video_path2 = "../datasets/video1_nuit.mp4"
    video_path3 = "../datasets/video2.mp4"

    video_shape_detection(shapeDetector_path, video_path3)
