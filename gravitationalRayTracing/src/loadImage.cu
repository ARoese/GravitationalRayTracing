#include "loadImage.hpp"
#include <vector_functions.h>

#define cimg_display 0
#include "CImg.h"

uint2 loadImage(const char* imageFileName, uchar3** imagePtr){
    cimg_library::CImg<unsigned char> starsCimg(imageFileName);
    uint2 starsDim = make_uint2(starsCimg.width(), starsCimg.height());
    size_t starsSize = starsDim.x*starsDim.y*sizeof(uchar3);
    uchar3* stars = (uchar3*)malloc(starsSize);
    *imagePtr = stars;
    for(int x = 0; x < starsCimg.width(); x++){
        for(int y = 0; y < starsCimg.height(); y++){
            uchar3* pixel = stars + starsDim.x*y + x;
            pixel->x = starsCimg(x,y,0);
            pixel->y = starsCimg(x,y,1);
            pixel->z = starsCimg(x,y,2);
            //if(pixel->x >= 25|| pixel->y >= 25|| pixel->z >= 25){
            //    printf("{%u, %u, %u}", (int)pixel->x, (int)pixel->y, (int)pixel->z);
            //}
        }
    }
    return starsDim;
}

void saveImage(const char* fileName, const uchar3* imagePtr, uint2 dims, const int number){
    cimg_library::CImg<unsigned char> outImg(dims.x, dims.y, 1, 3);
    for(int x = 0; x < dims.x; x++){
        for(int y = 0; y < dims.y; y++){
            const uchar3* pixel = imagePtr + dims.x*y + x;
            outImg(x,y,0) = pixel->x;
            outImg(x,y,1) = pixel->y;
            outImg(x,y,2) = pixel->z;
            //if(pixel->x >= 25|| pixel->y >= 25|| pixel->z >= 25){
            //    printf("{%u, %u, %u}", (int)pixel->x, (int)pixel->y, (int)pixel->z);
            //}
        }
    }
    
    outImg.save(fileName, number, 3);
}