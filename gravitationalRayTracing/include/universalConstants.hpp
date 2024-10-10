#ifndef UNIVERSALCONSTANTS_H
#define UNIVERSALCONSTANTS_H
#include <math.h>
//gravitational constant
constexpr float CONST_G = 6.6743e-11;
//speed of light: 299792458
constexpr float CONST_C = 10; 

//multiply by this to convert radians to degrees
constexpr float RAD2DEG = 180/M_PI;
//multiply by this to convert degrees to radians 
constexpr float DEG2RAD = M_PI/180;
#endif