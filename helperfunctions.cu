#include "helperfunctions.h"
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "constsize.h"

#define THREADS_DEFAULT 32

//makró, a sztring a program mûködését ismerteti. 
#define HELP_MESSAGE "The arguments of the BilateralFilter program:\n"\
"BilateralFilter input_image output_image sigma_s sigma_r radius threads\n"\
"where:\n"\
"  -  input_image: full path of the image file you want to process.\n"\
"  -  output_image: full path of the file where you want to save the processed image.\n"\
"  -  sigma_s: deviation of the spatial Gaussian function. Positive floating point number.\n"\
"  -  sigma_r: deviation of the range Gaussian function. Positive floating point number.\n"\
"  -  radius: radius of the spatial kernel. Integer, greater than zero.\n"\
"             The spatial kernel matrix contains 2*radius+1 columns and rows, the full size is (2*radius+1)^2\n"\
"             Note: you don't have to specify this argument. If you don't specify it, the program computes it so that radius <= 2*sigma_s is fulfilled.\n"\
"  -  threads: number of threads per block per dimension (so blockdim.x = blockdim.y = threads). Integer, greater than zero.\n"\
"              Note: you don't have to specify it. Default value is 32.\n\n"

//Ismerteti a program mûködését. stdout-ra vagy stderr-re írja az üzenetet.
//Ha a program a help argumentummal lett meghívva, az sdtin-re ír
//ha a parancssori argumentumok rosszul lettek megadva, akkor is kiíródik az üzenet, de ekkor az stderr-re.    
void printHelpMessage(FILE *stream)
{
	fprintf(stream, HELP_MESSAGE);
}

//Beolvassa és feldolgozza a parancssori paramétereket.
//Ha valami rosszul lett megadva, visszaad egy hibakódot.
//A mainbõl hívjuk a függvényt, a main-ben deklarált változókat állítjuk, ezért ezeket referenciaként vesszük át. 
int readConfigParameters(int argc, char **argv, float & sigma_s, float & sigma_r, int & r, int & threads)
{
	//Parancssori paraméterek feldolgozása
	//ha valam nem jó, akkor mindig megmondjuk, mi a hiba, kiírjuk a program mûködését bemutató sztinget és visszatérünk egy hibakóddal.

	//Ellenõrizzük a paraméterek számát. Az utolsó két paraméter, r és threads opcionálisak.
	if (!(argc >= 5 && argc <= 7)) {
		fprintf(stderr, "Number of arguments is incorrect.\n\n");
		printHelpMessage(stderr);
		return ARGUMENT_ERROR;
	}

	//beolvassuk a sigma_s, sigma_r, r threads paramétereket
	//sigma_s :
	double temp_d = atof(argv[3]);
	if (temp_d == 0.0) {
		fprintf(stderr, "Argument sigma_s is incorrect.\n\n");
		printHelpMessage(stderr);
		return ARGUMENT_ERROR;
	}
	else {
		sigma_s = temp_d;
	}

	//sigma_r :
	temp_d = atof(argv[4]);
	if (temp_d == 0.0) {
		fprintf(stderr, "Argument sigma_r is incorrect.\n\n");
		printHelpMessage(stderr);
		return ARGUMENT_ERROR;
	}
	else {
		sigma_r = temp_d;
	}

	//r beolvasása. Ennek megadása opcionális
	if (argc >= 6) {
		r = atoi(argv[5]);
		if (r <= 0) {
			fprintf(stderr, "The radius of the spatial kernel is incorrect.\n\n");
			printHelpMessage(stderr);
			return ARGUMENT_ERROR;
		}

		//limitált helyt foglaltunk le a konstans memóriában, még fordítása idõben, ezért elenõrzünk:
		if (r > MAX_RADIUS) {
			printf("The radius of the spatial kernel (r) must be maximum %d\n"
				"It is because of the limited amount constant memory. The size of the constant memory array should be a compile time constant.\n\n", MAX_RADIUS);
		}
	}
	else {  //ha nincs megadva r, akkor kiszámítjuk sigma_s alapján. r <= 2 * sigma_s teljesül
		r = (int)(2 * sigma_s);
		if (r < 1) r = 1;
	}

	//threads beovasása
	cudaDeviceProp device;
	cudaGetDeviceProperties(&device, 0);
	int maxThreadsPerBlock = device.maxThreadsPerBlock;
	if (argc == 7) {
		threads = atoi(argv[6]);
		if (threads <= 0 || threads * threads > maxThreadsPerBlock) {
			fprintf(stderr, "The number of threads per block dimension (threads) is incorrect.\n"
				"Your GPU can handle maximum %d threads per block.\n"
				"The algorithm uses 2D blocks, so the square of the threads argument can't be more than %d : threads * threads <= %d\n\n",
				maxThreadsPerBlock, maxThreadsPerBlock, maxThreadsPerBlock);
			printHelpMessage(stderr);
			return ARGUMENT_ERROR;
		}
	}
	else {
		threads = THREADS_DEFAULT;
		while (threads * threads > maxThreadsPerBlock) {
			threads /= 2;
		}
	}

	//a shared memory-ba bemásolt képrészletben a szélek nem fedhetnek át 
	if (r > threads / 2) {
		printf("r should be less than threads / 2, where r is the radius of the spatial kernel, threads is the number of threads per block dimension\n"
			"It is because of the algorithm. The programmer was too lazy.\n\n");
		return ARGUMENT_ERROR;
	}

	return 0;
}

//A mainbõl hívjuk, felszabadítja az összes device pointer-t. Mindig hívjuk, ha valami hiba van és ki kell takarítani.
//Figyelni kell, hogy a main-ben a pointereket deklaráláskor NULL-ra állítsuk. A device null pointer-re lehet hívni cudaFree-t. 
void freeEverything(unsigned char *d_inputImage, float *d_rangeKernel)
{
	cudaFree(d_inputImage);
	cudaFree(d_rangeKernel);
}

//A main-bõl hívjuk, lefoglalja a memóriát az összes device pointernek, amit használunk. Ha gond van, mindent felszabadít és false-szal tér vissza.
//Ha minden ok, akkor true-val tér vissza.
//referenciával kell átvenni a pointereket, mert a cudamalloc a pointer címét várja (void**)
bool doAllMallocs(unsigned char * & d_inputImage, float * & d_rangeKernel, int imageSize, int rangeKernelSize)
{
	if (cudaMalloc((void**)&d_inputImage, imageSize * sizeof(unsigned char)) != cudaSuccess) {
		d_inputImage = NULL;
		freeEverything(d_inputImage, d_rangeKernel);
		return false;
	}

	if (cudaMalloc((void**)&d_rangeKernel, rangeKernelSize * sizeof(float)) != cudaSuccess) {
		d_rangeKernel = NULL;
		freeEverything(d_inputImage, d_rangeKernel);
		return false;
	}

	return true;
}
