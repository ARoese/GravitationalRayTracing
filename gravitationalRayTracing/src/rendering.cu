#include "rendering.hpp"
#include <stdio.h>

#include "cutil_math.h"
#include "cmatrix.hpp"
#include "pthread.h"

#include "universalConstants.hpp"
#include "ray.hpp"
#include "loadImage.hpp"

__device__ __host__ uchar3 uvMap(uchar3 texture[], uint2 dim, float3 bodyPos, float3 rayPos, float3 rotation){
    //https://en.wikipedia.org/wiki/UV_mapping
    float3 d = normalize(rayPos - bodyPos);

    d = float3x3::Rx(rotation.x)*(float3x3::Ry(rotation.y)*(float3x3::Rz(rotation.z)*d));

    float u = 0.5 + atan2(d.z, d.x)/(2*M_PI);
    float v = 0.5 + asin(d.y)/M_PI;

    int x = (int)(u*dim.x);
    int y = (int)(v*dim.y);

    //printf("(%u,%u){%u, %u, %u}", x,y,(int)skybox[y*dim.x+x].x, (int)skybox[y*dim.x+x].y, (int)skybox[y*dim.x+x].z);

    return texture[y*dim.x+x];
}

__device__ __host__ uchar3 uvMap(uchar3 texture[], uint2 dim, float3 bodyPos, float3 rayPos){
    return uvMap(texture, dim, bodyPos, rayPos, make_float3(0,0,0));
}

__device__ __host__ void renderPixel(int xIdx, int yIdx, 
                            camera* c, body* bodies, int bc, 
                            uchar3* frameBuffer, uchar3* skybox, uint2 sbdim){

    uchar3* pixel = frameBuffer + yIdx*c->resolution.x + xIdx;
    float2 idx = make_float2(xIdx, yIdx);
    
    //screenspace xy coordinates in range [-0.5,0.5]
    float2 screenSpace = (idx/make_float2(c->resolution)) - 0.5;
    //angular screenspace coordinates from [-fov.x, fov.x] and so on
    float2 screenAngle = screenSpace * c->fov;

    float3 rayDir = {1,0,0};
    //local rotation
    rayDir = float3x3::Ry(screenAngle.y)
                    *(float3x3::Rz(screenAngle.x)*rayDir);
    //global rotation
    rayDir = float3x3::Rx(c->camRot.x)
                *(float3x3::Ry(c->camRot.y)
                    *(float3x3::Rz(c->camRot.z)
                        *rayDir));

    ray r(c->camPos, rayDir);
    #pragma unroll
    for(int i = 0; i < 10000; i++){
        r.step(0.005, bodies, bc);
        for(int b = 0; b<bc; b++){
            if(length(bodies[b].position - r.position) <= bodies[b].radius){
                body* bp = &bodies[b];
                if(bp->solidColor){
                    //printf("solidColor Impact: %d %d %d\n", bp->color.x, bp->color.y, bp->color.z);
                    *pixel = bp->color;
                }else{
                    //printf("UV impact\n");
                    *pixel = uvMap(bp->texture, bp->textureDim, bp->position, r.position, bp->rotation);
                }
                return;
            }
        }
    }

    //if the ray hasn't hit anything after all steps, it is assumed to have hit skybox
    *pixel = uvMap(skybox, sbdim, make_float3(0,0,0), r.position);
}

__global__ void renderFrameKernel(camera* c, body* bodies, int bc, uchar3* frameBuffer, uchar3* skybox, uint2 sbdim){
    int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
    if(xIdx > c->resolution.x || yIdx > c->resolution.y){
        return;
    }
    renderPixel(xIdx, yIdx, c, bodies, bc, frameBuffer, skybox, sbdim);
}

void renderGPU(int numFrames, camera c, 
                body bodies[], int bodiesCount,
                uint2 sunTextureDim, uchar3* sunTexture,
                uint2 starsTextureDim, uchar3* starsTexture){

    //copy sun texture to device
    uchar3* sunDev;
    size_t sunTextureSize = sizeof(uchar3)*sunTextureDim.x*sunTextureDim.y;
    cudaMalloc(&sunDev, sunTextureSize);
    cudaMemcpy(sunDev, sunTexture, sunTextureSize, cudaMemcpyHostToDevice);
    //hackey, but we know the second body is the sun. 
    //I don't want to generalize this too much more than it already is
    bodies[1].texture = sunDev; 

    //copy stars background to device
    uchar3* starsDev;
    size_t starsSize = sizeof(uchar3)*starsTextureDim.x*starsTextureDim.y;
    cudaMalloc(&starsDev, starsSize); //same as normal malloc, but the memory is on the GPU
    cudaMemcpy(starsDev, starsTexture, starsSize, cudaMemcpyHostToDevice);

    //copy bodies info to device
    body* bodiesDev;
    size_t bodiesSize = sizeof(body)*bodiesCount;
    cudaMalloc(&bodiesDev, bodiesSize);
    cudaMemcpy(bodiesDev, bodies, bodiesSize, cudaMemcpyHostToDevice);

    // make space for two cameras and frameBuffers
    // so we can have the cpu set one up while the other is in use
    camera* cameraDev;
    camera* cameraDev2;
    cudaMalloc(&cameraDev, sizeof(camera));
    cudaMalloc(&cameraDev2, sizeof(camera));
    cudaMemcpy(cameraDev, &c, sizeof(camera), cudaMemcpyHostToDevice);
    
    //set up framebuffer on host and device
    size_t frameBufferSize = c.resolution.x*c.resolution.y*sizeof(uchar3);
    uchar3* frameBuffer = (uchar3*)malloc(frameBufferSize);
    uchar3* frameBufferDev;
    uchar3* frameBufferDev2;
    cudaMalloc(&frameBufferDev, frameBufferSize);
    cudaMalloc(&frameBufferDev2, frameBufferSize);

    //max of 29 right now
    //24 seems optimal since we can't reach 32 due to register pressure
    unsigned int tdim = 32; //FIXED: Square tdim in the below equation
    int numBlocks = ((float)(c.resolution.x*c.resolution.y))/((float)(tdim*tdim));
    unsigned int bdim = (int)ceil(sqrt(numBlocks));

    //wait for it to finish
    cudaDeviceSynchronize();
    //check for errors
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    printf("starting kernel with dims %d,%d (%d)\n", bdim,bdim,tdim);

    //set up streams
    cudaStream_t renderStream;
    cudaStream_t copyStream;
    cudaStreamCreate(&renderStream);
    cudaStreamCreate(&copyStream);

    float3 camRotCenter = {250,0,0}; //point that the camera rotates around
    unsigned int camDistance = 250; //distance of camera from center object
    for(int i = 0; i < numFrames; i++){
        //update camera position
        float angle = ((float)i/numFrames)*2*M_PI;
        c.camPos = {cos(angle),sin(angle),0};
        c.camPos *= camDistance;
        c.camPos += camRotCenter;
        c.camRot = {0,0,angle+(float)M_PI};
        //send new camera orientation to device
        cudaMemcpyAsync(cameraDev, &c, sizeof(camera), cudaMemcpyHostToDevice, copyStream);
        cudaStreamSynchronize(copyStream);
        printf("Kernel %d set up... Waiting on previous operations.\n", i);
        cudaStreamSynchronize(renderStream); //wait for the last kernel to finish

        //render image
        renderFrameKernel<<<{bdim,bdim},{tdim,tdim}, 0, renderStream>>>
            (cameraDev, bodiesDev, bodiesCount, frameBufferDev, starsDev, starsTextureDim);
        
        //while that is running, save the last frame asynchronously
        printf("Kernel %d launched...\n", i);
        // our save should not operate on the regions currently being worked on
        std::swap(cameraDev, cameraDev2);
        std::swap(frameBufferDev, frameBufferDev2);
        if(i != 0){ //if it's the first invocation, we don't have a previous frame to save
            //copy previous frame back from device and save it
            cudaMemcpyAsync(frameBuffer, frameBufferDev, frameBufferSize, cudaMemcpyDeviceToHost, copyStream);
            cudaStreamSynchronize(copyStream);
            saveImage(
                "/root/gravitationalRayTracing/outputs/out.png",
                frameBuffer, 
                {(uint)c.resolution.x, (uint)c.resolution.y}, i-1);
            printf("Saved result of kernel %d...\n", i-1);
        }
    }
    cudaStreamDestroy(copyStream);
    cudaStreamDestroy(renderStream);

    //save final frame
    cudaMemcpy(frameBuffer, frameBufferDev2, frameBufferSize, cudaMemcpyDeviceToHost);
    saveImage(
        "/root/gravitationalRayTracing/outputs/out.png",
        frameBuffer, 
        {(uint)c.resolution.x, (uint)c.resolution.y}, numFrames-1);
    printf("Saved result of kernel %d...\n", numFrames-1);
    
    //check for errors
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    
    //free gpu resources
    cudaFree(bodiesDev);
    cudaFree(cameraDev);
    cudaFree(cameraDev2);
    cudaFree(frameBufferDev);
    cudaFree(frameBufferDev2);
    cudaFree(starsDev);
}

struct renderCPURowArgs {
    int rowOffset;
    int threadGroupSize;
    uchar3* frameBuffer;
    camera* c;
    body* bodies; 
    int bodiesCount;
    uint2 starsTextureDim;
    uchar3* starsTexture;
};

void* renderCPURows(void* p){
    renderCPURowArgs* args = (renderCPURowArgs*)p;
    int rowOffset = args->rowOffset;
    int threadGroupSize = args->threadGroupSize;
    uchar3* frameBuffer = args->frameBuffer;
    camera* c = args->c;
    body* bodies = args->bodies; 
    int bodiesCount = args->bodiesCount;
    uint2 starsTextureDim = args->starsTextureDim;
    uchar3* starsTexture = args->starsTexture;

    //start
    for(int y = rowOffset; y < c->resolution.y; y+=threadGroupSize){
        for(int x = 0; x < c->resolution.x; x++){
            renderPixel(x, y, c, bodies, bodiesCount, frameBuffer, starsTexture, starsTextureDim);
        }
    }
    return nullptr;
}

void renderCPUFrame(uchar3* frameBuffer, camera c, 
                body bodies[], int bodiesCount,
                uint2 sunTextureDim, uchar3* sunTexture,
                uint2 starsTextureDim, uchar3* starsTexture){
    int numThreads = 12;
    renderCPURowArgs args_template; 
    args_template.threadGroupSize = numThreads;
    args_template.frameBuffer = frameBuffer;
    args_template.c = &c;
    args_template.bodies = bodies;
    args_template.bodiesCount = bodiesCount;
    args_template.starsTextureDim = starsTextureDim;
    args_template.starsTexture = starsTexture;

    //spawn threads
    pthread_t threads[numThreads];
    renderCPURowArgs args[numThreads];
    for(int i = 0; i < numThreads; i++){
        args[i] = args_template;
        args[i].rowOffset = i;
        int result = pthread_create(&threads[i], nullptr, renderCPURows, &args[i]);
        if(result != 0){
            printf("Pthread error\n");
        }
    }

    //wait on threads
    for(int i = 0; i < numThreads; i++){
        pthread_join(threads[i], nullptr);
    }
}

void renderCPU(int numFrames, camera c, 
                body bodies[], int bodiesCount,
                uint2 sunTextureDim, uchar3* sunTexture,
                uint2 starsTextureDim, uchar3* starsTexture){

    size_t frameBufferSize = c.resolution.x*c.resolution.y*sizeof(uchar3);
    uchar3* frameBuffer = (uchar3*)malloc(frameBufferSize);
    float3 camRotCenter = {250,0,0}; //point that the camera rotates around
    unsigned int camDistance = 250; //distance of camera from center object
    for(int i = 0; i < numFrames; i++){
        //update camera position
        float angle = ((float)i/numFrames)*2*M_PI;
        c.camPos = {cos(angle),sin(angle),0};
        c.camPos *= camDistance;
        c.camPos += camRotCenter;
        c.camRot = {0,0,angle+(float)M_PI};
        //send new camera orientation to device
        printf("Rendering frame %d...\n", i);

        //render image
        renderCPUFrame(frameBuffer, 
                c, bodies, bodiesCount,
                sunTextureDim, sunTexture,
                starsTextureDim, starsTexture);
        
        saveImage(
            "/root/gravitationalRayTracing/outputs/out.png",
            frameBuffer, 
            {(uint)c.resolution.x, (uint)c.resolution.y}, i);
        printf("Saved frame %d\n", i);
    }
        
}