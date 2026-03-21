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

    try:
        print("Build en cours :")

        #logs
        for line in client.api.build(path="../", tag="belgian-sign-detector", rm=True, decode=True):
            if 'stream' in line:
                print(line['stream'].strip())
            elif 'error' in line:
                print(f"Erreur Docker : {line['error']}")
                raise Exception(line['error'])

        print("Image construite")
    except Exception as e:
        raise Exception(f"Erreur build image : {e}")

    # lancement du container
    try:
        print("Lancement du container en cours")
        container = client.containers.run("belgian-sign-detector", detach=True, ports={'5000/tcp': 5000, '8000/tcp': 8000},name="belgian-sign-detector-container")
        print("Conteneur lancé sur localhost:8000 !")
        print("Mlflow lancé sur localhost:5000 !")

        container.logs()
    except Exception as e:
        print("Erreur lors du lancement du container : ",e)
