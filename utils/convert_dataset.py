import os
import cv2

# Convertir dataset originel (https://btsd.ethz.ch/shareddata/) en format YOLO (img/txt)

def convert_dataset_to_yolo(dataset_path,output_image_path,output_label_path):
    print(f"Converting dataset {dataset_path} to YOLO format")
    print(f"Output image path: {output_image_path}")
    print(f"Output label path: {output_label_path}")

    os.makedirs(output_image_path, exist_ok=True)
    os.makedirs(output_label_path, exist_ok=True)

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)

        if not os.path.isdir(folder_path):
            continue

        print(folder)

        csv_file = None
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                csv_file = os.path.join(folder_path, file)
                break

        if csv_file is None:
            print("No CSV found in", folder)
            continue

        convert_folder_to_yolo(csv_file,folder_path,output_image_path,output_label_path)


def convert_folder_to_yolo(csv_path,folder_path,output_image_path,output_label_path):
    with open(csv_path, "r") as f:
        lines = f.readlines()[1:]

    for line in lines:
        parts = line.strip().split(";")
        filename, width, height, x1, y1, x2, y2, class_id = parts

        width = int(width)
        height = int(height)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        class_id = int(class_id)

        # conversion image
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        jpg_name = filename.replace(".ppm", ".jpg")
        cv2.imwrite(os.path.join(output_image_path, jpg_name), img)

        # calculs BoundingBox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        center_x = x1 + bbox_width / 2
        center_y = y1 + bbox_height / 2

        # normalisation
        center_x /= width
        center_y /= height
        bbox_width /= width
        bbox_height /= height

        # labels
        txt_name = filename.replace(".ppm", ".txt")

        with open(os.path.join(output_label_path, txt_name), "w") as out:
            out.write(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}")


if __name__ == "__main__":
    convert_dataset_to_yolo("../datasets/BelgiumTSC_Training/Training", "../datasets/BelgiumTSC_Training_YOLO/images", "../datasets/BelgiumTSC_Training_YOLO/labels")
    convert_dataset_to_yolo("../datasets/BelgiumTSC_Testing/Testing", "../datasets/BelgiumTSC_Testing_YOLO/images", "../datasets/BelgiumTSC_Testing_YOLO/labels")
