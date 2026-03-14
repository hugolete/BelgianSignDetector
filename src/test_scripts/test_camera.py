from ultralytics import YOLO
import cv2


if __name__ == '__main__':
    model = YOLO("../models/ShapeDetector_Kaggle_Epoch20.pt")

    results = model.track(source=1, stream=True)  # generator of Results objects

    for r in results:
        boxes = r.boxes

        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            coords = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()

            for box_id, box_coords in zip(track_ids, coords):
                print(f"ID: {box_id} | Coords: {box_coords}")

        cv2.imshow("Stream", r.plot())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
