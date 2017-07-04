#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H

#include <stdio.h>  //A FILE* miatt

//visszatérési értéke makrók a main-hez. A main ezekkel térhet vissza 
#define NO_DEVICE_ERROR 1
#define ARGUMENT_ERROR 2
#define NO_IMAGE_ERROR 3
#define CUDA_MALLOC_ERROR 4
#define CUDA_MEMCPY_ERROR 5
#define CONST_MEM_FILL_ERROR 6
#define KERNEL_ERROR -1

void printHelpMessage(FILE *stream);

int readConfigParameters(int argc, char **argv, float & sigma_s, float & sigma_r, int & r, int & threads);

bool doAllMallocs(unsigned char * & d_inputImage, float * & d_rangeKernel, int imageSize, int rangeKernelSize);

void freeEverything(unsigned char *d_inputImage, float *d_rangeKernel);

#endif  //HELPERFUNCTIONS_H