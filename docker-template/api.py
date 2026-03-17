import subprocess
import zipfile
import cv2
import torch
import os
from factorymlops.trainers.YOLOTraining import YOLOTraining
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import shutil
import uvicorn
from ultralytics import YOLO
from src.image_predict import shape_detection, get_crops, sign_detection, get_detected_signs, print_detections
from src.video_predict import video_shape_detection


app = FastAPI()
shapeDetector_path = "ShapeDetector_Kaggle_Epoch20.pt"
signDetector_path = "FinalModel.pt"
dataset_dir = "/app/datasets/"
models_dir = "/app/models/"

#chargement modèle
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

try:
    shapeDetector_model = YOLO("ShapeDetector_Kaggle_Epoch20.pt")
    sign_model = YOLO("FinalModel.pt")
    print("Modèles chargés !")
except Exception as e:
    raise ValueError("Erreur lors du chargement des modèles")


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


@app.post("/video-predict/")
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


@app.post("/training-model/{model_name}/")
async def training_model(background_tasks: BackgroundTasks, nb_epochs:int,exp_name:str,batch_size:int,learning_rate:float,patience:int,dataset_name:str,model_name:str):
    try:
        model_path = os.path.join(models_dir, model_name)
        dataset_path = os.path.join(dataset_dir, dataset_name)
        background_tasks.add_task(load_mlflow)

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


@app.post("/upload-dataset/")
async def upload_dataset(background_tasks: BackgroundTasks, dataset: UploadFile = File(...)):
    if not dataset.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers .zip sont acceptés")

    nom_fichier = dataset.filename
    nom_dataset = os.path.splitext(nom_fichier)[0]
    zip_path = os.path.join(dataset_dir, nom_fichier)
    extract_folder = os.path.join(dataset_dir, nom_dataset)

    try:
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(dataset.file, buffer)

        if not os.path.exists(extract_folder):
            os.makedirs(extract_folder)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

        return JSONResponse(status_code=200, content={"dataset_name": nom_dataset})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/datasets/")
async def get_datasets(background_tasks: BackgroundTasks):
    response = []

    try:
        for folder in os.listdir(dataset_dir):
            if os.path.isdir(os.path.join(dataset_dir, folder)):
                response.append(folder)

        return JSONResponse(status_code=200, content={"datasets": response})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/models/")
async def get_models(background_tasks: BackgroundTasks):
    response = []

    try:
        for file in os.listdir(models_dir):
            response.append(file)

        return JSONResponse(status_code=200, content={"models": response})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.delete("/datasets/{dataset_name}")
async def delete_dataset(background_tasks: BackgroundTasks, dataset_name: str):
    try:
        os.remove(os.path.join(dataset_dir, dataset_name))

        return JSONResponse(status_code=200, content={"dataset_name": dataset_name})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.delete("/models/{model_name}")
async def delete_model(background_tasks: BackgroundTasks, model_name: str):
    try:
        os.remove(os.path.join(models_dir, model_name))

        return JSONResponse(status_code=200, content={"model_name": model_name})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


def load_mlflow():
    subprocess.run("mlflow ui")


def download_file(path:str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Fichier non trouvé. Vérifiez le chemin")

    filename = os.path.basename(path).split('/')[-1]

    return FileResponse(
        path=path,
        filename=f"{filename}.zip",  # Le nom que l'utilisateur verra lors du téléchargement
        media_type='application/zip'
    )


if __name__ == "__main__":
    print("Starting server")
    uvicorn.run(app, host="0.0.0.0", port=5000)
