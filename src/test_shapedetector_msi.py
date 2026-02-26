from ultralytics import YOLO


def smart_predict(image_path):
    # prédit sur l'image normale (Facteur 1) -> Pour ce qui est loin
    res_far = model.predict(image_path, imgsz=640, conf=0.25)
    print(res_far)
    print("Detections:", len(res_far[0].boxes))
    #res_far[0].show()

    # prédit en x0.5 -> Pour les distances moyennes
    res_medium = model.predict(image_path, imgsz=320, conf=0.25)
    print(res_medium)
    print("Detections:", len(res_medium[0].boxes))
    #res_medium[0].show()

    # prédit en x0.25 -> Pour ce qui est proche
    res_close = model.predict(image_path, imgsz=160, conf=0.25)
    print(res_close)
    print("Detections:", len(res_close[0].boxes))
    #res_close[0].show()


if __name__ == "__main__":
    image_path = "../datasets/Dataset test/18.jpg" # cette image contient au moins 1 panneau pour chaque type de distance

    model_path = "../models/ShapeDetector_Kaggle_Epoch20.pt"
    model = YOLO(model_path)

    smart_predict(image_path)
