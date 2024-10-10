#ifndef CAMERA_H
#define CAMERA_H
#include <vector_types.h>
#include "universalConstants.hpp"

class camera{
    public:
    float2 fov = {90*RAD2DEG, 90*RAD2DEG};
    float3 camPos = {0,0,0};
    float3 camRot = {0,0,0};
    int2 resolution = {255,255};
    __device__ __host__ camera(float2 fov, float3 camPos, float3 camRot, int2 resolution);
};

#endif