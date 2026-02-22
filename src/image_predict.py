from factorymlops.demonstrators.YOLODemonstrator import YOLODemonstrator
from ultralytics import YOLO


def use_predictor(model_path:str, image_path:str):
    predictor = YOLODemonstrator()

    output_image_path = predictor.predict(model_path, image_path)
    print(f"Détection sauvegardée dans {output_image_path}")


if __name__ == '__main__':
    model_path = str(input("Chemin du modèle"))
    image_path = str(input("Chemin de l'image"))

    choix = 0
    while choix < 1 or choix > 2:
        print("1. Recevoir l'image finale de détection")
        print("2. Recevoir la détection sur l'image (sous forme textuelle)")

        choix = int(input())
        if choix == 1:
            use_predictor(model_path, image_path)
        elif choix == 2:
            model = YOLO(model_path)

            results = model.predict(source=image_path,imgsz=640,conf=0.01)
            print(results)

            #results[0].show()
            # TODO terminer
        else:
            print("Choix invalide")
