### Container for retrieving images

The container reads the images of yesterday from given edgeAI devices and removes duplicates.
At last step it removes empty folders.

podman machine start
podman build -t docker.io/haverland/retrievedigits:latest .
podman push docker.io/haverland/retrievedigits:latest
kc apply -f Deployment.yaml
