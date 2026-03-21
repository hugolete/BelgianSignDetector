import docker
from dotenv import load_dotenv
import os


if __name__ == '__main__':
    load_dotenv()

    print("Initialisation client docker")

    try:
        client = docker.DockerClient(base_url=os.getenv("DOCKER_URL"))
        print("Client Docker initialisé")
    except Exception as e:
        raise Exception("Erreur dans l'init du client Docker : ",e)

    # build de l'image
    try:
        print("Build en cours (ce processus peut prendre du temps)")
        client.images.build(path="../", tag="BelgianSignDetector", rm=True)
        print("Image construite")
    except Exception as e:
        raise ValueError(f"Erreur build image : {e}")

    # lancement du container
    try:
        print("Lancement du container en cours")
        container = client.containers.run("BelgianSignDetector", detach=True, ports={'5000/tcp': 5000, '8000/tcp': 8000},name="BelgianSignDetector-container")
        print("Conteneur lancé sur localhost:8000 !")
        print("Mlflow lancé sur localhost:5000 !")

        container.logs()
    except Exception as e:
        print("Erreur lors du lancement du container : ",e)
