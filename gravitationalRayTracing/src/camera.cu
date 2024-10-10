#include "camera.hpp"


__device__ __host__ camera::camera(float2 fov, float3 camPos, float3 camRot, int2 resolution)
    : fov(fov), camPos(camPos), camRot(camRot), resolution(resolution)
    {}