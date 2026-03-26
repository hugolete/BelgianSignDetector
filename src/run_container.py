import docker
from dotenv import load_dotenv
import os


load_dotenv()
print("Initialisation client docker")
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# connexion au client docker
try:
    client = docker.DockerClient(base_url=os.getenv("DOCKER_URL"))
    print("Client Docker initialisé")
except Exception as e:
    raise Exception("Erreur dans l'init du client Docker : ",e)

# lancement du container
try:
    print("Lancement du container en cours")

    try:
        container = client.containers.get("belgian-sign-detector-container")

        if container.status != "running":
            print(f"Relance du container belgian-sign-detector-container")
            container.start()
            print("Container démarré !")
            print("L'API sera lancée sur localhost:8000 !")
            print("Mlflow sera lancé sur localhost:5000 !")
            print("\n")
            print("--- Logs du conteneur ---")

            for line in container.logs(stream=True):
                print(line.decode('utf-8').strip())
        else:
            print("Le container tourne déjà.")
    except docker.errors.NotFound:
        print(f"Le container belgian-sign-detector-container n'existe pas. Création...")
        container = client.containers.run("belgian-sign-detector", detach=True,device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])], ports={'5000/tcp': 5000, '8000/tcp': 8000},name="belgian-sign-detector-container")
        print("Nouveau container créé et démarré.")
        print("L'API sera lancée sur localhost:8000 !")
        print("Mlflow sera lancé sur localhost:5000 !")

        print("--- Logs du conteneur ---")
        for line in container.logs(stream=True):
            print(line.decode('utf-8').strip())
except Exception as e:
    print("Erreur lors du lancement du container : ",e)
