import subprocess
import cv2
import torch
import os
from factorymlops.trainers.YOLOTraining import YOLOTraining
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
import shutil
import uvicorn
from ultralytics import YOLO
from src.image_predict import shape_detection, get_crops, sign_detection, get_detected_signs, print_detections
from src.video_predict import video_shape_detection


app = FastAPI()
shapeDetector_path = "ShapeDetector_Kaggle_Epoch20.pt"
signDetector_path = "FinalModel.pt"

#chargement modèle
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

try:
    shapeDetector_model = YOLO("ShapeDetector_Kaggle_Epoch20.pt")
    sign_model = YOLO("model.pt")
    print("Modèles chargés !")
except Exception as e:
    raise ValueError("Erreur lors du chargement du modèle")


@app.post("/predict/")
async def predict_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # enregistrer l'image
        file_location = f"/tmp/{file.filename}"

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = cv2.imread(file_location)

        final_boxes = shape_detection(shapeDetector_path, img)
        cropped_signs = get_crops(img, final_boxes)
        final_signs = sign_detection(signDetector_path, cropped_signs)
        detected_signs = get_detected_signs(final_signs)  # detected_signs = liste finale des panneaux identifiés, variable utile pour d'autres traitements éventuels
        print_detections(detected_signs, 0.25)

        #nettoyer l'espace après l'inférence
        os.remove(file_location)

        return JSONResponse(status_code=200, content={"detections": detected_signs})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/videopredict/")
async def predict_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        file_location = f"/tmp/{file.filename}"

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        total_detected_signs = video_shape_detection(shapeDetector_path, file_location,test=True)

        print("Panneaux détectés sur la vidéo : ")
        for sign in total_detected_signs:
            if sign['conf'] > 0.40:
                print(f"Panneau : {sign['label']} -> Position : {sign['position']} | Confiance: {sign['conf']} | Frame : {sign['frame']}")

        return JSONResponse(status_code=200, content={"detections": total_detected_signs})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/training_shapeDetector/")
async def training_shapeDetector(background_tasks: BackgroundTasks, nb_epochs:int,exp_name:str,batch_size:int,learning_rate:float,patience:int,dataset: UploadFile = File(...),):
    model_path = "ShapeDetector_Kaggle_Epoch20.pt"
    background_tasks.add_task(load_mlflow)

    try:
        # téléchargement dataset (il doit être upload sous forme zippée)
        # TODO
        zipped_dataset_path = ""
        unzip_dataset(zipped_dataset_path)

        trainer = YOLOTraining()

        results_training = trainer.train(
            "0",
            model_path,
            nb_epochs,
            640,
            "http://127.0.0.1:5000",
            exp_name,
            dataset_path,
            "../models/",
            batch_size,
            learning_rate,
            patience
        )

        return JSONResponse(status_code=200, content={"results_training": results_training})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


def load_mlflow():
    subprocess.run("mlflow ui")


def unzip_dataset(zipped_dataset_path:str):
    #TODO
    pass


if __name__ == "__main__":
    print("Starting server")
    uvicorn.run(app, host="0.0.0.0", port=5000)
