objects = obj/body.o obj/camera.o obj/cmatrix.o obj/main.o obj/ray.o obj/loadImage.o obj/rendering.o
#optim = -g -G
optim = -O3

grt: $(objects)
	nvcc $(optim) -arch=native -Iinclude -ICImg $(objects) -o grt

run: grt
	rm -f outputs/*.png
	./grt gpu 256 32 true
	ffmpeg -framerate 10 -stream_loop 4 -i "outputs/out_%03d.png" -c:v libx264 -r 20 -y outputs/out.mp4

timesweep:
	bash ./timesweep.sh

obj:
	mkdir obj

obj/%.o: src/%.cu | obj
# optimal seems to be 32 registers
	nvcc $(optim) -x cu -arch=native -maxrregcount=32 -Iinclude -ICImg -dc $< -o $@

clean:
	rm -rf *.o grt obj outputs/*