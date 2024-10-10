FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
RUN apt update -y && apt install -y imagemagick ffmpeg
ADD gravitationalRayTracing /root/gravitationalRayTracing
WORKDIR /root/gravitationalRayTracing
CMD make clean && make run