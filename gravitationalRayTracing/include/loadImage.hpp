#ifndef LOADIMAGE_H
#define LOADIMAGE_H
#include <vector_types.h>


uint2 loadImage(const char* imageFileName, uchar3** imagePtr);
void saveImage(const char* fileName, const uchar3* imagePtr, uint2 dims, const int number);

#endif