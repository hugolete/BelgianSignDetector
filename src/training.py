from factorymlops.trainers.YOLOTraining import YOLOTraining
import os


if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)
    model_path = str(input("Model path"))
    dataset_path = str(input("Dataset YAML's path"))
    nb_epochs = int(input("How many epochs ?"))
    exp_name = str(input("Experience name"))
    batch_size = int(input("Batch size")) # 4 ou 8
    learning_rate = float(input("Learning rate")) # 0.0025 si 4, 0.005 si 8
    patience = int(input("Patience (amount of epochs without upgrade before stopping)"))

    choix = int(input("With augmentation ? 0 for No, 1 for Yes"))

    if choix == 1:
        scale = float(input("Scaling factor")) # 0.1
        mosaic = float(input("Mosaic factor")) # 1
        mixup = float(input("Mixup factor")) # 0.1
    else:
        # valeurs par défaut
        scale = 0.5
        mosaic = 1.0
        mixup = 0.0

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
        patience,
        scale,
        mosaic,
        mixup
    )

    print(results_training)
