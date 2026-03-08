from src.video_predict import video_shape_detection
import os


if __name__ == "__main__":
    shapeDetector_path = "../../models/ShapeDetector_Kaggle_Epoch20.pt"
    video_paths = [
        "../../datasets/archive/traffic-sign-to-test.mp4",
        "../../datasets/video1_nuit.mp4",
        "../../datasets/video2.mp4",
    ]

    for video_path in video_paths:
        print("Vidéo : ", video_path)
        total_detected_signs = video_shape_detection(shapeDetector_path,video_path,test=True)

        with open(f"detections.txt", "a") as f:
            f.write(f"Video : {os.path.basename(video_path).split('/')[-1]}\n")

            for sign in total_detected_signs:
                if sign['conf'] > 0.40:
                    output = f"Panneau : {sign['label']} -> Position : {sign['position']} | Confiance: {sign['conf']} | Frame : {sign['frame']}"
                    f.write(output + "\n")

            f.write("\n")
