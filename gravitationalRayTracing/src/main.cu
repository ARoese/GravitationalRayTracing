#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <fcntl.h>

//https://github.com/kashif/cuda-workshop/blob/master/cutil/inc/cutil_math.h
#include "cutil_math.h"

#include "body.hpp"
#include "universalConstants.hpp"
#include "camera.hpp"
#include "ray.hpp"
#include "cmatrix.hpp"
#include "loadImage.hpp"
#include "rendering.hpp"

void printUsage(char* programName){
    printf("USAGE: %s device imageDim num_frames debug\n", programName);
    printf("device: cpu or gpu. Defaults to gpu unless cpu is selected.\n");
    printf("imageDim: Dimensions of output image.\n");
    printf("num_frames: How many frames to render.\n");
    printf("debug: true/false, whether to use debug uv textures\n");
}

//3840,2160 is 4K
int main(int argc, char* argv[]) {
    // take care of command line stuff
    unsigned int numFrames = 16;
    int imageDim = 1024;
    //const char* starsPath = "/root/gravitationalRayTracing/assets/test_uv.jpg";
    const char* starsPath = "/root/gravitationalRayTracing/assets/8k_stars_milky_way.jpg";
    bool renderOnCPU = false;
    if(argc == 1){
        printf("NOTICE: Using defaults\n");
    }else if(argc == 5){
        renderOnCPU = (strcmp(argv[1], "cpu") == 0) || (strcmp(argv[1], "CPU") == 0);
        imageDim = atoi(argv[2]);
        numFrames = atoi(argv[3]); //numFrames
        if(strcmp(argv[4], "true") == 0){ //use debug uv
            starsPath = "/root/gravitationalRayTracing/assets/test_uv.jpg";
        }
    }else{
        printUsage(argv[0]);
        return 1;
    }
    
    if(numFrames <= 0){
        printf("ERROR: argument num_frames ('%s') is not valid.\n", argv[3]);
        printUsage(argv[0]);
        return 1;
    }else if(imageDim <= 0){
        printf("ERROR: argument image_dims ('%s') is not valid.\n", argv[2]);
        printUsage(argv[0]);
        return 1;
    }

    printf("rendering on %s\n", renderOnCPU ? "CPU" : "GPU");

    //init camera
    camera c({50*DEG2RAD, 50*DEG2RAD}, {0,0,0}, {0,0,0}, {imageDim,imageDim});

    //load in sun texture
    uchar3* sunTexture;
    uint2 sunTextureDim = loadImage("/root/gravitationalRayTracing/assets/8k_sun.jpg", &sunTexture);

    //load in sky-sphere texture
    uchar3* stars;
    uint2 starsDim = loadImage(starsPath, &stars); //host malloc happens somewhere in here
    
    // set up scene
    //TODO: add in the sun texture on device in GPU 
    body bodies[] = {
        body(6, 1e11, {250,0,0},{0,0,0},{6,0,0}),
        body(20, 0, {250,-60,0},{0,0,0}, sunTexture, sunTextureDim),
        //body(6, 0, {140,4,0},{0,0,0},{0,128,0}),
    };
    constexpr int bodiesCount = sizeof(bodies)/sizeof(body);    

    if(renderOnCPU){
        renderCPU(numFrames, c, 
                bodies, bodiesCount, 
                sunTextureDim, sunTexture,
                starsDim, stars);
    }else{
        renderGPU(numFrames, c, 
                bodies, bodiesCount, 
                sunTextureDim, sunTexture,
                starsDim, stars);
    }
    
    printf("Done rendering\n");
    return 0;
}