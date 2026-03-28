# API
This file is dedicated to api.py, the REST API used to communicate with the Docker container

## POST /predict/
Make a prediction on an image
### Input in body (form-data) :
- Image file 
### Output : 
- "detections" dictionary containing the signs that were detected

## POST /video-predict/
Make a prediction on a video
### Input in body (form-data) :
- Video file 
### Output : 
- "detections" dictionary containing the signs that were detected

## POST /training-model/{model_name}
Trains a model
### Input in URL :
- Model's name in the URL
### Input in body (form-data)
- Epochs : number of epochs
- Experience name : needed for MlFlow
- Batch size : How many samples are passed through the model at once during training (recommended: 4-16 according to your GPU's power)
- Learning rate : controls how much the model’s weights are updated at each step during training (recommended: 0.0025-0.005)
- Patience : number of epochs without upgrade before stopping
- Dataset path : datasets in the YOLO format have a .yaml file, you need to give this file's path to the program
### Output :
- Notifies you when the training starts.
- The trained model will be delivered as model.pt in the /app/models folder

## POST /val-model/{model_name}
Validate a model
### Input in URL :
- Model's name in the URL
### Input in body (form-data)
- Dataset path : datasets in the YOLO format have a .yaml file, you need to give this file's path to the program
### Output :
- Notifies you when the validation starts.
- The results will be delivered as a timestamped YAML file in the /app/eval_results folder

## POST /upload-dataset/
Uploads a dataset into /app/datasets
### Input in body (form-data) :
- .zip file containing the dataset in the YOLO Format
### Output :
- Dataset name (to confirm it's been correctly uploaded)

## POST /upload-model/
### Input in body (form-data) :
- .pt file (model)
### Output :
- Model name (to confirm it's been correctly uploaded)

## GET /datasets/
### Output :
- Returns the list of the folders present in /app/datasets

## GET /models/
### Output :
- Returns the list of the files present in /app/models

## DELETE /datasets/{dataset_name}
### Input in URL : 
- Dataset name
### Output :
- Returns the name of the deleted dataset

## DELETE /models/{model_name}
### Input in URL : 
- Model name
### Output :
- Returns the name of the deleted model

## GET /download-file/
Downloads a file from the container
### Input in body (form-data) :
- Path of the file you want to download
### Output :
- File

## POST /upload-file/
Uploads a file in /app
### Input :
- File you want to upload
### Output :
- Name of the uploaded file (to confirm it's been correctly uploaded)

## GET /status/
### Output :
- Returns the current status (training, val, idle)

## GET /logs/
### Output :
- Returns the last 20 log lines

## GET /logs-file
### Output :
- Returns the entire log file
