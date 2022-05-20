### Container for retrieving images

podman build -t docker.io/haverland/retrievedigits:latest .
podman push docker.io/haverland/retrievedigits:latest
kc apply -f Deployment.yaml


