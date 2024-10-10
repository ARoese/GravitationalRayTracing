#include "ray.hpp"
#include "cutil_math.h"
#include "universalConstants.hpp"

ray::ray(float3 position, float3 direction)
    : position(position), direction(direction)
    {}

void ray::step(float timestep, body* bodies, int bc){
    direction *= CONST_C;
    position += direction*timestep;
    for(int i = 0; i < bc; i++){
        direction += getAccelerationTo(&bodies[i])*timestep;
    }
    direction = normalize(direction);
}

float3 ray::getAccelerationTo(body* b){
    float3 directionToBody = b->position - position;
    float amplitude = ((CONST_G*b->mass)/length(directionToBody));
    return normalize(directionToBody)*amplitude;
}