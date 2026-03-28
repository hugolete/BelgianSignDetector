import json
import zipfile
from collections import deque
import cv2
import torch
import os
from factorymlops.trainers.YOLOTraining import YOLOTraining
from factorymlops.validators.YOLOValidator import YOLOValidator
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
import shutil
import uvicorn
from starlette.responses import StreamingResponse
from ultralytics import YOLO
from src.image_predict import shape_detection, get_crops, sign_detection, get_detected_signs, print_detections
from src.video_predict import video_shape_detection
from datetime import datetime
import logging


app = FastAPI()
shapeDetector_path = "/app/models/ShapeDetector.pt"
signDetector_path = "/app/models/SignDetector.pt"
dataset_dir = "/app/datasets/"
models_dir = "/app/models/"
log_file = "/app/api_log.log"
status = "IDLE"

#chargement modèle
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Carte graphique trouvée et sélectionnée")
else:
    device = torch.device("cpu")
    print("Pas de carte graphique trouvée, CPU sélectionné")

#logging
logging.getLogger("ultralytics").setLevel(logging.INFO)

try:
    shapeDetector_model = YOLO(os.path.join(models_dir, "ShapeDetector.pt"))
    sign_model = YOLO(os.path.join(models_dir, "SignDetector.pt"))
    print("Modèles chargés !")
except Exception as e:
    raise ValueError("Erreur lors du chargement des modèles")


@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        global status
        status = "PREDICTING"

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
        status = "IDLE"

        return JSONResponse(status_code=200, content={"detections": detected_signs})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/video-predict/")
async def predict_video(file: UploadFile = File(...)):
    try:
        global status
        status = "PREDICTING"

        # enregistrer la vidéo
        file_location = f"/tmp/{file.filename}"

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        total_detected_signs = video_shape_detection(shapeDetector_path, signDetector_path,file_location,test=True)

        print("Panneaux détectés sur la vidéo : ")
        for sign in total_detected_signs:
            if sign['conf'] > 0.40:
                print(f"Panneau : {sign['label']} -> Position : {sign['position']} | Confiance: {sign['conf']} | Frame : {sign['frame']}")

        # nettoyer l'espace après l'inférence
        os.remove(file_location)
        status = "IDLE"

        return JSONResponse(status_code=200, content={"detections": total_detected_signs})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/training-model/{model_name}/")
async def training_model(background_tasks: BackgroundTasks,model_name:str, nb_epochs:int = Form(...),exp_name:str = Form(...),batch_size:int = Form(...),learning_rate:float = Form(...),patience:int = Form(...),dataset_yaml_name:str = Form(...)):
    try:
        global status

        if status != "IDLE":
            raise HTTPException(status_code=409, detail="Le serveur est déja occupé")

        model_path = os.path.join(models_dir, model_name)
        dataset_path = os.path.join(dataset_dir, dataset_yaml_name)
        dataset_yaml_path = os.path.join(dataset_path, dataset_yaml_name + ".yaml")

        def run_training():
            global status
            status = "TRAINING"

            try:
                trainer = YOLOTraining()

                print("Démarrage de l'entrainement")
                results_training = trainer.train(
                    "0",
                    model_path,
                    nb_epochs,
                    640,
                    "http://127.0.0.1:5000",
                    exp_name,
                    dataset_yaml_path,
                    f"models/",
                    batch_size,
                    learning_rate,
                    patience
                )

                # sauvegarde des résultats
                with open(f"/app/data/results_{exp_name}.json", "w") as f:
                    try:
                        json.dump(results_training.results_dict, f)
                    except Exception as e:
                        print(f"Erreur durant la sauvegarde des résultats (results_dict) : {str(e)}")

                print("Modèle entrainé, /models/model.pt")
            except Exception as e:
                print(f"Erreur durant l'entraînement : {str(e)}")
            finally:
                status = "IDLE"
                print(f"--- Fin entrainement ---")

        background_tasks.add_task(run_training)

        return JSONResponse(status_code=202, content={"message": "Entraînement lancé en arrière-plan", "experiment": exp_name})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/val-model/{model_name}")
async def val_model(background_tasks: BackgroundTasks, model_name:str, dataset_yaml_name:str = Form(...)):
    try:
        global status

        if status != "IDLE":
            raise HTTPException(status_code=409, detail="Le serveur est déja occupé")

        model_name = model_name+".pt"
        print(model_name)
        model_path = os.path.join(models_dir, model_name)
        dataset_path = os.path.join(dataset_dir, dataset_yaml_name)
        dataset_yaml_path = os.path.join(dataset_path, dataset_yaml_name+".yaml")

        if not os.path.exists(dataset_yaml_path):
            return JSONResponse(status_code=400, content={"error": f"{dataset_yaml_path} non trouvé"})

        def run_val():
            global status
            status = "VAL"

            try:
                validator = YOLOValidator()
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                print("Démarrage de la validation")
                print(model_path)
                metrics = validator.eval(model_path,dataset_yaml_path)

                # sauvegarde des résultats
                print("Sauvegarde des résultats")
                with open(f"/app/data/results_eval_{timestamp}.json", "w") as f:
                    json.dump(metrics, f)
            except Exception as e:
                print(f"Erreur durant la validation : {str(e)}")
            finally:
                status = "IDLE"
                print("Fin validation")

        background_tasks.add_task(run_val)

        return JSONResponse(status_code=202,content={"message": "Validation lancée en arrière-plan"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/upload-dataset/")
async def upload_dataset(dataset: UploadFile = File(...)):
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

        os.remove(zip_path)

        return JSONResponse(status_code=200, content={"dataset_name": nom_dataset})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/upload-model/")
async def upload_model(model : UploadFile = File(...)):
    nom_fichier = model.filename
    nom_modele = os.path.splitext(nom_fichier)[0]
    model_path = os.path.join(models_dir, nom_fichier)

    try:
        with open(model_path, "wb") as buffer:
            shutil.copyfileobj(model.file, buffer)

        return JSONResponse(status_code=200, content={"model_name": nom_modele})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/datasets/")
async def get_datasets():
    response = []

    try:
        for folder in os.listdir(dataset_dir):
            if os.path.isdir(os.path.join(dataset_dir, folder)):
                response.append(folder)

        return JSONResponse(status_code=200, content={"datasets": response})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/models/")
async def get_models():
    response = []

    try:
        for file in os.listdir(models_dir):
            response.append(file)

        return JSONResponse(status_code=200, content={"models": response})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.delete("/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str):
    try:
        shutil.rmtree(os.path.join(dataset_dir, dataset_name), ignore_errors=True)

        return JSONResponse(status_code=200, content={"dataset_name": dataset_name})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    try:
        os.remove(os.path.join(models_dir, model_name))

        return JSONResponse(status_code=200, content={"model_name": model_name})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/download-file/")
async def download_file(path:str = Form(...)):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Fichier non trouvé. Vérifiez le chemin")

    filename = os.path.basename(path)

    return FileResponse(
        path=path,
        filename=filename,
        media_type='application/octet-stream'
    )


@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    path = "/app"
    nom_fichier = file.filename

    try:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return JSONResponse(status_code=200, content={"file_name": nom_fichier})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/status/")
async def get_status():
    return JSONResponse(status_code=200, content={"status": status})


@app.get("/logs/")
async def get_logs():
    if not os.path.exists(log_file):
        return {"logs": ["Fichier de log pas généré"]}

    try:
        # on ne garde que les 20 dernières lignes
        with open(log_file, "r", encoding="utf-8") as f:
            last_lines = deque(f, 20)

        clean_logs = [line.strip() for line in last_lines]

        return {
            "filename": os.path.basename(log_file),
            "logs": clean_logs
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la lecture des logs : {str(e)}")


@app.get("/logs/file")
async def get_log_file():
    if not os.path.exists(log_file):
        raise HTTPException(status_code=404, detail="Aucun log disponible")

    def iterfile():
        # On ouvre en mode binaire pour le stream
        with open(log_file, mode="rb") as f:
            yield from f

    return StreamingResponse(
        iterfile(),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=api_debug_full.log"}
    )


if __name__ == "__main__":
    print("Starting server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
