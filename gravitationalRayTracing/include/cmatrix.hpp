#ifndef CMATRIX_H
#define CMATRIX_H
#include <vector_types.h>


class float3x3{
    public:
    float3 r1;
    float3 r2;
    float3 r3;
    __device__ __host__ float3x3(float3 r1, float3 r2, float3 r3);
    __device__ __host__ static float3x3 Rx(float a);
    __device__ __host__ static float3x3 Ry(float a);
    __device__ __host__ static float3x3 Rz(float a);
};

__device__ __host__ float3 operator*(float3x3 matrix, float3 vector);

#endif