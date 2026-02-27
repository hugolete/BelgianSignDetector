from ultralytics import YOLO
import torch
import cv2
from ultralytics.utils.ops import scale_boxes
from torchvision.ops import nms


def shape_detection(shapeDetector_path:str,image_path:str):
    img = cv2.imread(image_path)
    h0,w0 = img.shape[:2] # récup de la taille réelle
    shapeDetector = YOLO(shapeDetector_path)

    imgszs = [640, 320, 160]  # 3 inférences pour choper les panneaux à différentes "distances"
    all_predictions = []

    # inférences
    for size in imgszs:
        results = shapeDetector.predict(image_path, imgsz=size, conf=0.25)

        boxes = results[0].boxes
        if len(boxes)>0:
            raw_data = boxes.data.clone() # copie des résultats (pour garder l'original au cas ou)
            raw_data[:, :4] = scale_boxes(results[0].orig_shape, raw_data[:, :4], (h0, w0)) # on dit au modèle de ramener les détections a la taille originale
            all_predictions.append(raw_data) # on ajoute a la liste une fois que tout tient ensemble

        #for box in results[0].boxes:
            #b = box.data  # [x1, y1, x2, y2, confidence, class]
            #all_predictions.append(b)

    if len(all_predictions) > 0:
        # concaténation de toutes les détections dans un tensor
        all_predictions = torch.cat(all_predictions, dim=0)
        boxes = all_predictions[:, :4]  # x1, y1, x2, y2
        scores = all_predictions[:, 4]  # confiance

        # NMS pour fusionner les boites et éviter les doublons
        keep_indices = nms(boxes, scores, iou_threshold=0.45)
        final_boxes = all_predictions[keep_indices]

        print(f"Nombre de panneaux uniques trouvés : {len(final_boxes)}")
    else:
        print("Pas de détection sur cette image")
        return []

    for detection in final_boxes:
        x1, y1, x2, y2, conf, cls = detection.tolist()
        label = shapeDetector.names[int(cls)]
        print(f"- Panneau {label} détecté avec confiance {conf:.2f} aux coords [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

        # dessin sur image (vérif)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    cv2.imshow("Detection Finale Fusionnee", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    shapeDetector_path = "../models/ShapeDetector_Kaggle_Epoch20.pt"
    model_path = "../models/FinalModel.pt"

    image_path = str(input("Chemin de l'image"))

    shape_detection(shapeDetector_path, image_path)
    # TODO : crop image de base selon les coordonnées des panneaux détectés. Objectif : obtenir 1 image par panneau, qui ne contient que le panneau
    # TODO : run inférence sur chaque image cropée avec FinalModel pour déterminer le type du panneau lui-même
    # TODO : print chaque détection finale
    # TODO : faire le lien entre l'image cropée et le type de panneau détecté dessus, pour fournir la détection exacte (position et type) sur l'image de base
