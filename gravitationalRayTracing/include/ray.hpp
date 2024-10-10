#ifndef RAY_H
#define RAY_H
#include <vector_types.h>
#include "body.hpp"

class ray{
    public:
    float3 position;
    float3 direction;
    __device__ __host__ ray(float3 position, float3 direction);
    //steps the ray in it's direction and updates direction based on gravity
    __device__ __host__ void step(float timestep, body* bodies, int bc);
    private:
    __device__ __host__ float3 getAccelerationTo(body* b);
};
#endif