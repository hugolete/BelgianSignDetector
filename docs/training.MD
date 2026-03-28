## Training prerequisites
- a YOLO26 model
- MlFlow is usable to track metrics (not mandatory)

## Start training
Run training.py and fill the required parameters (the program will ask you)

### Training parameters

- Model path : path of the model's file (.pt)
- Dataset path : datasets in the YOLO format have a .yaml file, you need to give this file's path to the program
- Epochs : number of epochs (how many times should the training algorithm completely pass through the dataset)
- Experience name : needed for MlFlow, you can put anything you want if you don't have it
- Batch size : How many samples are passed through the model at once during training (recommended: 4-16 according to your GPU's power)
- Learning rate : controls how much the model’s weights are updated at each step during training (recommended: 0.0025-0.005)
- Patience : number of epochs without upgrade before stopping
- Scale : resizes images by a random factor within the specified range.
- Mosaic : defines the probability of combining 4 images into one.
- Mixup : defines the probability of blending 2 images together

## Output

The program will show you the progression of the training in the console. 
The trained model's file will be delivered as model.pt in the /models folder
