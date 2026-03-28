import docker
from dotenv import load_dotenv
import os


if __name__ == '__main__':
    load_dotenv()
    print("Initializing Docker Client")
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    try:
        client = docker.DockerClient(base_url=os.getenv("DOCKER_URL"))
        print("Docker Client initialized")
    except Exception as e:
        raise Exception("Error while initializing the Docker client : ",e)

    try:
        print("Image build starting :")
        print(f"Context : {root_path}")

        #logs
        for line in client.api.build(path=root_path, tag="belgian-sign-detector", rm=True, decode=True):
            if 'stream' in line:
                print(line['stream'].strip())
            elif 'error' in line:
                print(f"Docker error : {line['error']}")
                raise Exception(line['error'])

        print("Image built")
    except Exception as e:
        raise Exception(f"Erreur build image : {e}")

    # lancement du container
    try:
        print("Starting container")
        container = client.containers.run("belgian-sign-detector", detach=True,device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])], ports={'5000/tcp': 5000, '8000/tcp': 8000},name="belgian-sign-detector-container",ipc_mode="host")
        print("API will be available on localhost:8000 !")
        print("Mlflow will be available on localhost:5000 !")

        print("--- Container logs ---")
        for line in container.logs(stream=True):
            print(line.decode('utf-8').strip())

    except Exception as e:
        print("Container error : ",e)
