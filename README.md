Ray tracer written from scratch in CUDA which demonstrates the bending of ray paths by the gravity of objects 
in the scene. Because of the highly hardware-specific nature of CUDA, this code is containerized for 
portability. The code isn't garunteed to run outside of the included docker container.

# dependencies
(installed in container automatically by dockerfile)  
- CImg (included)
- ImageMagick (for CImg)
- ffmpeg (for turning png sequences into mp4)

# Running tests using the demo container
1. Build it
   - `docker build -t grt -f Dockerfile .`
2. Run it
   - `docker run --ipc=host --name grt-test -it --gpus all grt`
3. Get output
   - `docker cp grt-test:/root/gravitationalRayTracing/outputs/out.mp4 out.mp4`
4. Delete container
   - `docker rm grt-test`

Or, run the `testIt.sh` file included here.

# Running the development container
1. Build it
`docker -t grt-dev -f Dockerfile-dev .`
2. Run it
    - Powershell: `docker run --ipc=host --name grt-dev -dit --gpus all --mount type=bind,source="$($pwd.path + "\gravitationalRayTracing")",target="/root/gravitationalRayTracing" grt-dev`

    - Bash: `docker run --ipc=host --name grt-dev -dit --gpus all --mount type=bind,source="$PWD\gravitationalRayTracing",target="/root/gravitationalRayTracing" grt-dev`
(bash is untested, basically just make source= an absolute path to the local gravitationalRayTracing folder)  

Attach to the dev container using vscode, or if you insist on using the container's terminal, remove the -d flag.  
Recommended VSCode extensions for CUDA programming:
- C/C++
- Makefile Tools
- Nsight Visual Studio Code Edition
- PBM/PPM/PGM Viewer (if it still outputs those image types)

# Notes
- This dev container will bind the local folder on the host. If you modify something on there, it will change the files on the host! This does not apply to the other non-dev container.

- This is entirely CUDA! If you don't have a nvidia gpu, you can't use this!

- This project was submitted as the final for CSC543 at Kutztown University in spring 2024, run by Dr. Parson. GravitationalRayMarching.odp is the slides used in the live presentation.

# Example output

https://github.com/user-attachments/assets/4f28244d-b141-40ec-8bba-8343f344f209


