from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
import torch
import cv2


def shape_detection(shapeDetector_path:str,image_path:str):
    img = cv2.imread(image_path)
    shapeDetector = YOLO(shapeDetector_path)

    imgszs = [640, 320, 160]  # 3 inférences pour choper les panneaux à différentes "distances"
    all_predictions = []

    # inférences
    for size in imgszs:
        results = shapeDetector.predict(image_path, imgsz=size, conf=0.25)

        for box in results[0].boxes:
            b = box.data  # [x1, y1, x2, y2, confidence, class]
            all_predictions.append(b)

    if len(all_predictions) > 0:
        # concaténation de toutes les détections dans un tensor
        all_predictions = torch.cat(all_predictions, dim=0)

        # NMS pour fusionner les boites et éviter les doublons
        best_unique_boxes = non_max_suppression(all_predictions.unsqueeze(0), conf_thres=0.25, iou_thres=0.25)
        final_boxes = best_unique_boxes[0]

        print(f"Nombre de panneaux uniques trouvés : {len(final_boxes)}")
    else:
        raise ValueError("Pas de détection sur cette image")

    for detection in final_boxes:
        x1, y1, x2, y2, conf, cls = detection.tolist()
        label = shapeDetector.names[int(cls)]
        print(f"- Panneau {label} détecté avec confiance {conf:.2f} aux coords [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

        # dessin sur image (vérif)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    cv2.imshow("Detection Finale Fusionnee", img)


if __name__ == '__main__':
    shapeDetector_path = "../models/ShapeDetector_Kaggle_Epoch20.pt"
    model_path = "../models/FinalModel.pt"

    image_path = str(input("Chemin de l'image"))

    shape_detection(shapeDetector_path, image_path)
    # TODO : crop image de base selon les coordonnées des panneaux détectés. Objectif : obtenir 1 image par panneau, qui ne contient que le panneau
    # TODO : run inférence sur chaque image cropée avec FinalModel pour déterminer le type du panneau lui-même
    # TODO : print chaque détection finale
    # TODO : faire le lien entre l'image cropée et le type de panneau détecté dessus, pour fournir la détection exacte (position et type) sur l'image de base
