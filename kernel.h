#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"

bool fillConstantMemory(int r, float sigma_s);

__global__ void createRangeKernel(float *rangeKernel, float sigma, int maxRangeDiff);


__global__ void bilateralFilter(unsigned char *in, float *rangeKernel, int r, int maxRangeDiff, int width, int height);


#endif