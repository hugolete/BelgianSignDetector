# Docker container

I have containerized this little application, with an API explained in api.MD to interact with the container

## Docker builder
Docker-builder.py is a script that will build a Docker image for this project & start a container from the image

### Prerequisites
- Docker
- Docker host's URL in an environment variable named "DOCKER_URL"

## Run container
Run_container.py is a script that will run the "Belgian-sign-detector-container" container that is normally created by docker-builder.py

### Prerequisites
- Docker
- Docker host's URL in an environment variable named "DOCKER_URL"
- BelgianSignDetector image built by docker-builder.py
