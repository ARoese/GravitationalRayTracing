#include "body.hpp"
body::body(float rad, float mass, float3 position, float3 rotation, uchar3 col) 
    : mass(mass), position(position), rotation(rotation)
    {
        radius = rad;
        color = col;
        float schwarzschildRadius = getSchwarzschildRadius();
        if(schwarzschildRadius > radius){
            radius = schwarzschildRadius;
            color = make_uchar3(0,0,0);
        }
    }

body::body(float rad, float mass, float3 position, float3 rotation, uchar3* texture, uint2 texDim)
    : mass(mass), position(position), rotation(rotation), textureDim(texDim), texture(texture)
    {
        radius = rad;
        float schwarzschildRadius = getSchwarzschildRadius();
        if(schwarzschildRadius > radius){
            radius = schwarzschildRadius;
            color = make_uchar3(0,0,0);
        }
        solidColor = false;
    }

float body::getSchwarzschildRadius(){
    return (2*CONST_G*mass)/CONST_C;
}