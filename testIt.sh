docker build -t grt -f Dockerfile .;
docker run --ipc=host --name grt-test -it --gpus all grt;
docker cp grt-test:/root/gravitationalRayTracing/outputs/out.mp4 out.mp4;
docker rm grt-test;