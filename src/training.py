from factorymlops.trainers.YOLOTraining import YOLOTraining

if __name__ == '__main__':
    model_path = str(input("Chemin du modèle"))
    nb_epochs = int(input("Nombre d'epochs"))
    exp_name = str(input("Nom expérience"))
    batch_size = int(input("Batch size")) # 4 ou 8
    learning_rate = float(input("Learning rate")) # 0.0025 si 4, 0.005 si 8

    trainer = YOLOTraining()
    trainer.train("gpu",model_path,nb_epochs,640,"127.0.0.1:5000",exp_name,"dataset.yaml","models/",batch_size,learning_rate)
