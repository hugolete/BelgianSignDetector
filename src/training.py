from factorymlops.trainers.YOLOTraining import YOLOTraining

if __name__ == '__main__':
    model_path = str(input("Chemin du modèle"))
    nb_epochs = int(input("Nombre d'epochs"))
    exp_name = str(input("Nom expérience"))
    batch_size = int(input("Batch size")) # 4 ou 8
    learning_rate = float(input("Learning rate")) # 0.0025 si 4, 0.005 si 8
    patience = int(input("Patience (nb d'epochs sans amélioration avant arrêt)"))

    choix = int(input("Avec augmentation ? 0 pour non, 1 pour oui"))

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
        "dataset.yaml",
        "../models/",
        batch_size,
        learning_rate,
        patience,
        scale,
        mosaic,
        mixup
    )

    print(results_training)
