import docker
from dotenv import load_dotenv
import os


load_dotenv()
print("Initializing Docker client")
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# connexion au client docker
try:
    client = docker.DockerClient(base_url=os.getenv("DOCKER_URL"))
    print("Docker client initialized")
except Exception as e:
    raise Exception("Error when initializing Docker Client : ",e)

# lancement du container
try:
    print("Starting container")

    try:
        container = client.containers.get("belgian-sign-detector-container")

        if container.status != "running":
            print(f"Restarting container belgian-sign-detector-container")
            container.start()
            print("Container started !")
            print("API will be available on localhost:8000 !")
            print("Mlflow will be available on localhost:5000 !")
            print("\n")
            print("--- Container logs ---")

            for line in container.logs(stream=True):
                print(line.decode('utf-8').strip())
        else:
            print("Container already running")
    except docker.errors.NotFound:
        print(f"Container belgian-sign-detector-container doesn't exist. Creating it now")
        container = client.containers.run("belgian-sign-detector", detach=True,device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])], ports={'5000/tcp': 5000, '8000/tcp': 8000},name="belgian-sign-detector-container",ipc_mode="host")
        print("New container created & started")
        print("API will be available on localhost:8000 !")
        print("Mlflow will be available on localhost:5000 !")

        print("--- Container logs ---")
        for line in container.logs(stream=True):
            print(line.decode('utf-8').strip())
except Exception as e:
    print("Error while launching the container : ",e)
