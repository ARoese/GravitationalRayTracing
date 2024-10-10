#include <vector_functions.hpp>
#include "cmatrix.hpp"
#include "cutil_math.h"

float3x3::float3x3(float3 r1, float3 r2, float3 r3)
    : r1(r1), r2(r2), r3(r3)
    {}

float3 operator*(float3x3 matrix, float3 vector){
    return make_float3(
        matrix.r1.x * vector.x + matrix.r1.y * vector.y + matrix.r1.z * vector.z,
        matrix.r2.x * vector.x + matrix.r2.y * vector.y + matrix.r2.z * vector.z,
        matrix.r3.x * vector.x + matrix.r3.y * vector.y + matrix.r3.z * vector.z
    );
}

float3x3 float3x3::Rx(float a){
    return float3x3(
        make_float3(1,0,0),
        make_float3(0,cos(a),-sin(a)),
        make_float3(0,sin(a),cos(a))
    );
}

float3x3 float3x3::Ry(float a){
    return float3x3(
        make_float3(cos(a),0,sin(a)),
        make_float3(0,1,0),
        make_float3(-sin(a),0,cos(a))
    );
}

float3x3 float3x3::Rz(float a){
    return float3x3(
        make_float3(cos(a),-sin(a),0),
        make_float3(sin(a),cos(a),0),
        make_float3(0,0,1)
    );
}