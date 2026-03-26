import docker
from dotenv import load_dotenv
import os


if __name__ == '__main__':
    load_dotenv()
    print("Initialisation client docker")
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    try:
        client = docker.DockerClient(base_url=os.getenv("DOCKER_URL"))
        print("Client Docker initialisé")
    except Exception as e:
        raise Exception("Erreur dans l'init du client Docker : ",e)

    try:
        print("Build en cours :")
        print(f"Contexte : {root_path}")

        #logs
        for line in client.api.build(path=root_path, tag="belgian-sign-detector", rm=True, decode=True):
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
        container = client.containers.run("belgian-sign-detector", detach=True,device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])], ports={'5000/tcp': 5000, '8000/tcp': 8000},name="belgian-sign-detector-container",ipc_mode="host")
        print("L'API sera lancée sur localhost:8000 !")
        print("Mlflow sera lancé sur localhost:5000 !") #TODO ne fonctionne pas encore

        print("--- Logs du conteneur ---")
        for line in container.logs(stream=True):
            print(line.decode('utf-8').strip())

    except Exception as e:
        print("Erreur lors du lancement du container : ",e)
