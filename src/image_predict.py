from sympy.codegen.ast import continue_
from ultralytics import YOLO
import torch
import cv2
from ultralytics.utils.ops import scale_boxes
from torchvision.ops import nms


def shape_detection(shapeDetector_path:str,img):
    h0,w0 = img.shape[:2] # récup de la taille réelle
    shapeDetector = YOLO(shapeDetector_path)

    imgszs = [640, 320, 160]  # 3 inférences pour choper les panneaux à différentes "distances"
    all_predictions = []

    # inférences
    for size in imgszs:
        results = shapeDetector.predict(img, imgsz=size, conf=0.25)

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
        # raise ValueError("Pas de détection sur cette image")

    for detection in final_boxes:
        x1, y1, x2, y2, conf, cls = detection.tolist()
        label = shapeDetector.names[int(cls)]
        print(f"- Panneau {label} détecté avec confiance {conf:.2f} aux coords [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

        # dessin sur image (vérif, facultatif)
        #cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    #cv2.imshow("Detection Finale Fusionnee", img)
    #cv2.waitKey(0)

    return final_boxes


def get_crops(img,final_boxes,padding=10):
    print("Début crop")
    h0, w0 = img.shape[:2]  # récup de la taille réelle
    crops_list = []

    for i,detection in enumerate(final_boxes):
        x1, y1, x2, y2, conf, cls = detection.tolist()

        # application du padding pour ne pas risquer un bug
        ix1 = max(0, int(x1) - padding)
        iy1 = max(0, int(y1) - padding)
        ix2 = min(w0, int(x2) + padding)
        iy2 = min(h0, int(y2) + padding)

        # découpage Numpy
        crop_img = img[iy1:iy2, ix1:ix2]

        crops_list.append({
            "crop_id": i,
            "image": crop_img,
            "coords_orig": [int(x1), int(y1), int(x2), int(y2)],  # Utile pour le dessin final
            "shape_conf": conf
        })

        # sauvegarde pour vérifier les crops
        #cv2.imwrite(f"debug_crop_{i}.jpg", crop_img)

    return crops_list


def sign_detection(signDetector_path,cropped_signs):
    print("Début détection panneau")
    signDetector = YOLO(signDetector_path)

    for sign in cropped_signs:
        results = signDetector.predict(sign['image'])

        if len(results[0].boxes) > 0:
            top_box = results[0].boxes[0]
            sign['final_class'] = signDetector.names[int(top_box.cls)]
            sign['final_conf'] = float(top_box.conf)
        else:
            sign['final_class'] = "Inconnu/Rejeté"
            sign['final_conf'] = 0.0

    return cropped_signs


def get_detected_signs(final_signs):
    detected_signs = {}

    for sign in final_signs:
        label = sign.get('final_class')
        detected_signs[label] = detected_signs.get(label, 0) + 1

    return detected_signs


if __name__ == '__main__':
    shapeDetector_path = "../models/ShapeDetector_Kaggle_Epoch20.pt"
    signDetector_path = "../models/FinalModel.pt"

    image_path = str(input("Chemin de l'image"))
    img = cv2.imread(image_path)

    question_dessin = int(input("Dessin final? (0 pour oui, 1 pour non)"))
    if question_dessin != 0 and question_dessin != 1:
        raise ValueError("Nombre invalide")

    final_boxes = shape_detection(shapeDetector_path, img)
    cropped_signs = get_crops(img,final_boxes)
    final_signs = sign_detection(signDetector_path,cropped_signs)
    detected_signs = get_detected_signs(final_signs)

    for sign in detected_signs:
        print(f"Panneau détecté : {sign} | {detected_signs[sign]} détections")

    if question_dessin == 0:
        pass
    elif question_dessin == 1:
        pass

    # TODO facultatif : faire le lien entre l'image cropée et le type de panneau détecté dessus, pour fournir la détection exacte (position et type) sur l'image de base
