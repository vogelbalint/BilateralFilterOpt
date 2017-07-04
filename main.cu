
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <stdio.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helperfunctions.h"

#include "kernel.h"
#include "constant.h"
#include "constsize.h"

#define MAX_RANGE_DIFF 255

int main(int argc, char** argv) 
{
	//Ha a help argumentummal indítjuk a programot, ismertetjük a program mûködését.
	if (argc == 2 && strcmp("help", argv[1]) == 0) {
		printHelpMessage(stdout);
		return 0;
	}
	//Megnézzük, hogy van-e megfelelõ GPU
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		fprintf(stderr, "You don't have a CUDA capable GPU. Buy one! Sorry.\n");
		return NO_DEVICE_ERROR;
	}
	cudaSetDevice(0);

	float sigma_s, sigma_r;		//a megfelelõ Gauss függvények paraméterei
	int r, threads;				//r: a spatial kernel sugara, threads: a blokkonkénti thread-ek száma adott dimenzióban

	int returnValue = readConfigParameters(argc, argv, sigma_s, sigma_r, r, threads);
	if (returnValue != 0) {
		return returnValue;
	}

	cv::Mat image;						//openCV függvénnyel olvassuk be a képet.
	image = cv::imread(argv[1], 0);		//beolvassuk a képet, 8 bit szürkeárnyalatossá konvertáljuk
	if (!image.data) {
		fprintf(stderr, "Could not open or find the input image\n\n");
		return NO_IMAGE_ERROR;
	}

	int width = image.cols, height = image.rows;	//a kép adatai
	int imageSize = width * height;

	int rangeKernelSize = MAX_RANGE_DIFF * 2 + 1;	//=511
	
	unsigned char *d_inputImage = NULL;	//felszabadíthatunk egy adott pointert a freeEverything függgvénnyel, ezért minden pointer nullra állítunk.
	float *d_rangeKernel = NULL;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	//feltöltjük a konstans memóriát
	if (!fillConstantMemory(r, sigma_s)) {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return CONST_MEM_FILL_ERROR;
	}

	//Az összes használt device memóriát lefoglaljuk. Ha hiba van, kilépünk.
	if (!doAllMallocs(d_inputImage, d_rangeKernel, imageSize, rangeKernelSize)) {
		fprintf(stderr, "cudaMalloc failed!\n\n");
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return CUDA_MALLOC_ERROR;
	}

	//bemásoljuk a device-ra a képet
	if (cudaMemcpy(d_inputImage, image.data, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n\n");
		freeEverything(d_inputImage, d_rangeKernel);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return CUDA_MEMCPY_ERROR;
	}

	//range kernel feltöltése
	createRangeKernel << <1, rangeKernelSize >> >(d_rangeKernel, sigma_r, MAX_RANGE_DIFF);

	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("Something went wrong during the execution of the createRangeKernel cuda kernel \n\n");
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		freeEverything(d_inputImage, d_rangeKernel);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return KERNEL_ERROR;
	}

	//a kép szélének r szélességû részét nem processzáljuk. Ez talán elfogadható.
	//a processzált rész méretei:
	int processedPartWidth = width - 2 * r;
	int processedPartHeight = height - 2 * r;

	//a block dimenziók megadása.
	//pl x irányban: a legkisebb olyan egész szám kell, ami threads-nek többszöröse, de nagyobb processsdPartWidth-nál:
	int blocksX = (processedPartWidth + threads - 1) / threads;
	int blocksY = (processedPartHeight + threads - 1) / threads;

	//a blokkonként bemásolt képrészlet méretei:
	int imagePartWidth = threads + 2 * r;
	int imagePartSize = imagePartWidth * imagePartWidth;

	//A lényeg:
	bilateralFilter <<<dim3(blocksX, blocksY), dim3(threads, threads), imagePartSize * sizeof(unsigned char) + rangeKernelSize * sizeof(float) >>>
					(d_inputImage, d_rangeKernel, r, MAX_RANGE_DIFF, width, height);

	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("Something went wrong during the execution of th bilateral filter kernel\n\n");
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		freeEverything(d_inputImage, d_rangeKernel);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return KERNEL_ERROR;
	}

	//kép másolása device to host
	if (cudaMemcpy(image.data, d_inputImage, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost) != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n\n");
		freeEverything(d_inputImage, d_rangeKernel);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return CUDA_MEMCPY_ERROR;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;		//Mérjük, hogy mennyi ideig tartott a GPU specifikus utasítások végrehajtása.
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Time to generate: %3.1f ms\n"
		"with parameters: sigma_s = %3.1f, sigma_r = %3.1f, spatial kernel radius = %d, number of threads per block dim = %d\n\n",
		elapsedTime, sigma_s, sigma_r, r, threads);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//kép mentése
	if (!cv::imwrite(argv[2], image)) {
		fprintf(stderr, "Failed to save the processed image.\n\n");
		freeEverything(d_inputImage, d_rangeKernel);
		return NO_IMAGE_ERROR;
	}

	freeEverything(d_inputImage, d_rangeKernel);
	return 0;	//csak akkor térünk vissza 0-val, ha minden rendben ment
}