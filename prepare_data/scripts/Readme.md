### Container for retrieving images

The container reads the images of yesterday from given edgeAI devices and removes duplicates.
At last step it removes empty folders.

#### Functions

The images will be read from edgeAI device. The list is in top of retrieve_and_prepare.py. It loads the last 5 days, if not already done.

All images will be stored in raw_images folder. After download duplicates and similars will be removed.

The rest of the images will be predicted and stored with prediction as prefix in predictions folder.
If where are multiple predictions (manual or new model prediction changed), the images marked in predictions folder with big 'X'.



podman machine start
podman build -t docker.io/haverland/retrievedigits:latest .
podman push docker.io/haverland/retrievedigits:latest
kc apply -f Deployment.yaml
