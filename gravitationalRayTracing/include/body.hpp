#ifndef BODY_H
#define BODY_H
#include <vector_types.h>

#include "universalConstants.hpp"

class body{
    public:
    float radius;
    float mass;
    float3 position;
    float3 rotation;
    uchar3 color; //rgb 0-255
    uchar3* texture;
    uint2 textureDim;
    bool solidColor = true;
    __host__ __device__ body(float radius, float mass, float3 position, float3 rotation, uchar3 color);
    __host__ __device__ body(float radius, float mass, float3 position, float3 rotation, uchar3* texture, uint2 textureDim);
    __host__ __device__ float getSchwarzschildRadius();
};
#endif