#include <math.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"
#include "constsize.h"

//itt definiáljuk a konstans memória tömböt (a többi helyen csak deklaráljuk)
__constant__ float data[CONST_SIZE];

//Gauss függvényt számít. x négyzetét kell neki átadni, és a szórást (sigma)
__host__ __device__ float gauss(float x_square, float sigma)
{
	return expf(-x_square / (2 * sigma * sigma));
}

//feltölti a spatial kernelt a konstans memóriában
bool fillConstantMemory(int r, float sigma_s)
{
	int n = 2 * r + 1;			//a spatial kernel (mint mátrix) oldalának hossza
	int spatialKernelSize = n * n;

	//elõször a host-on foglalunk memóriát és a CPU számítja ki a spatial kernelt.
	//rövid idõre kell memória a host-on, ezt foglalhatjuk cudaHostAlloc -kal, mert csak rövid idõre csökken a kilapozható memória mennyisége.
	//cserébe gyorsabb lesz a host --> device másolás
	float *h_data;	 
	if (cudaHostAlloc((void**)&h_data, spatialKernelSize * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {	//TODO: egyéb flag-eket állítani?
		printf("cudaHostAlloc failed!\n\n");
		return false;
	}
	
	//a spatial kernel kiszámítása:
	//int pos = 0;
	for (int j = -r; j <= r; ++j) {
		int y = j + r;
		for (int i = -r; i <= r; ++i) {
			int x = i + r;
			h_data[x + y * n] = gauss((float)(i * i + j * j), sigma_s);	
			//h_data[pos++] = gauss((float)(i * i + j * j), sigma_s);
		}
	}

	//másolás host --> constant memory
	if (cudaMemcpyToSymbol(data, h_data, spatialKernelSize * sizeof(float)) != cudaSuccess) {
		printf("cudaMemcpyToSymbol failed!\n\n");
		cudaFreeHost(h_data);
		return false;
	}
	cudaFreeHost(h_data);
	return true;
}

//A lehetséges intenzitásbeli különbségekhez (dI) tartozó Gauss értékeket is elõre kiszámítjuk, az ezt tároló tömböt hívom range kernel -nek
//A range kernelt majd bemásoljuk a shared memory-ba, mert ebbõl sokszor kell majd olvasnunk.
//a legkisebb dI -255 , a legnagyobb dI 255 ==> a tömb 255 *2 + 1 = 511 elemt tartalmaz
//A tömb közepén van a dI = 0 -hoz tartozó Gauss érték
//a függvény paraméterként átveszi dI maximumát is (maxRangeDiff), hogy ne legyenek a kódban varázs számok. 
__global__ void createRangeKernel(float *rangeKernel, float sigma, int maxRangeDiff)
{
	//elõször csak a pozitív delte I -khez tartozó Gausst számítjuk ki, mert a Gauss függvény szimmetrikus
	int tid = threadIdx.x;
	if (tid >= maxRangeDiff) {
		int deltaI = tid - maxRangeDiff;
		rangeKernel[tid] = gauss((float)(deltaI * deltaI), sigma);
	}

	__syncthreads();

	//átmásoljuk a negatív intenzitás különbség értkekhez tartozó Gauss értékeket a tömb második felébõl
	int last = maxRangeDiff * 2;  //=510
	if (tid < maxRangeDiff) {
		rangeKernel[tid] = rangeKernel[last - tid];
	}
}

//segédfüggvény a bilateralFilter cuda kernelhez.
//a bilateralFilter cuda kernelben a kép egy részét a shared memory-ba másoljuk, de több pixelt kell másolni, mint az adott blokk összes thread-jének száma.
//ezért a blokk széleinél szenvedni kell, a szélekhez tartozó pixelek esetén ki kell számítani egy eltolást.
//ezt nehéz írásban átadni, papíron rajzolgatva lehet megérteni. 
__device__ void computeShift(int r, int blockdim, int i, int j, int& dx, int& dy)
{
	if (j < r)
		dy = -r;
	else if (j >= blockdim - r)
		dy = r;

	if (i < r)
		dx = -r;
	else if (i >= blockdim - r)
		dx = r;
}

//a bilateral filtert megvalósító cuda kernel
//a spatial kernel elemeit konstans memóriából éri el, a range kernel elemeit globális memóriából a shared memóriába másoljuk
//sõt: az adott blokkhoz tartozó processzálandó pixeleket (intenzitás értékeket) is bemásoljuk a shared memory-ba
//de: nem csak ezeket a pixeleket kell bemásolni, hanem azokat is, amikre szükség van a pixelek feldolgozásához.
//vagyis egy r "széleségû" sávot is be kell másolni plusszban a szélekhez (r a spatial kernel sugara) --> ez nem triviális
//egyéb probléma: a shared memory-ban unsigned char (kép) és float (range kernel) elemeket is tárolunk,
//ezért a shared memória tömbjét char tömbként deklaráljuk és megfelelõ típusó pointereket állítunk a megfelelõ helyekre.
//Megj.: hogy az algoritmus ne legyen nagyon bonyolult, nem processzáljuk azokat a pixeleket, amik a (teljes) kép szélein vannak
//(a kép határához közeebb van r -nél). Ez egyrészt csökkenti a thread divergenciát, másrészt nem kell annyi mindent átgondolni
//Megj.: mivel a képet bemásoljuk a shared memóriákba (részenként), nem kell külön output kép, az input képet felülírjuk
//(ehhez persze thread szinkronizálás kell).
//width, height: a képe szélessége és magassága pixelben
__global__ void bilateralFilter(unsigned char *in, float *rangeKernel, int r, int maxRangeDiff, int width, int height)
{
	extern __shared__ char sharedData[];				

	int x = threadIdx.x + blockIdx.x * blockDim.x + r;	//az adott threadhez tartozó pixel koordinátái
	int y = threadIdx.y + blockIdx.y * blockDim.y + r;

	unsigned char *pImagePart = reinterpret_cast<unsigned char*>(sharedData);	//a shared memory képrészletet tároló részének elejére mutató pointer

	int imagePartWidth = blockDim.x + 2 * r;		//a blokkban tárolt képrészlet oldalának hossza. Az algoritmusban blockDim.x = blockDim.y
	int imagePartSize = imagePartWidth * imagePartWidth;

	float *pRangeKernel = reinterpret_cast<float*>(sharedData + imagePartSize * sizeof(unsigned char));	//a range kernel rész elejére mutat

	//feltöltjük a shared memory range kernel részét
	int index = threadIdx.x + blockDim.x * threadIdx.y;  //minden thread-nek az adott blokkban van egy egyedi indexe
	int step = blockDim.x * blockDim.y;					//az összes thread száma a blockban
	int rangeKernelSize = 2 * maxRangeDiff + 1;			//=511
	while (index < rangeKernelSize) {
		pRangeKernel[index] = rangeKernel[index];
		index += step;
	}
	pRangeKernel += maxRangeDiff;		//most a range kernelt tároló rész közepére mutat (a dI = 0 -hoz tartozó részre)

	int offset = x + y * width;			//az adott pixel helye a kép unsigned char tömbjében

	int i = threadIdx.x, j = threadIdx.y;	//"távolság a blokk oldalától"

	int ii = i + r, jj = j + r;				//távolság a bemásolt képrészlet oldalától, ezek a képrészletben indexelnek. 
	
	if (x < width - r &&  y < height - r) {

		pImagePart[ii + jj * imagePartWidth] = in[offset];	//a képrészlet belsejének másolása

		int dx = 0, dy = 0;							//segédváltozók a képrészlet széleinek átmásolásához

		//átmásoljuk a széleket:
		computeShift(r, blockDim.x, i, j, dx, dy);		//TODO: lehetne inline -olni? érdemes-e?

		if (dx != 0)
			pImagePart[(ii + dx) + jj * imagePartWidth] = in[(x + dx) + y * width];
		if (dy != 0)
			pImagePart[ii + (jj + dy) * imagePartWidth] = in[x + (y + dy) * width];
		if (dx != 0 && dy != 0)	
			pImagePart[(ii + dx) + (jj + dy) * imagePartWidth] = in[(x + dx) + (y + dy) * width];
	}

	//kell szinkronizálás, mert ezelõtt írunk a shared memory-ba, ezután olvasunk belõle.
	//nem feltétlenül jó a __syncthreads() egy if-ben, ezért az if-t szétszedtem két részre
	__syncthreads();

	if (x < width - r &&  y < height - r) {

		int n = 2 * r + 1;									//spatial kernel oldalának hossza
		float *pSpatialKernel = data  + (r + r * n);		//a spatial kernel közepére mutat

		//jön a lényeg, sok szenvedés után:
		int intensity = pImagePart[ii + jj * imagePartWidth];	//az adot pixel intenzitása

		float summa = 0.0f, weightSumma = 0.0f;			//a weightSumma csak a súlyokat összegzi

		for (int l = -r; l <= r; ++l) {			//l: sorindex
			for (int k = -r; k <= r; ++k) {		//k: oszlopindex

				int intensity_kl = pImagePart[(ii + k) + (jj + l) * imagePartWidth];
				int dI = intensity - intensity_kl;
				float temp = pSpatialKernel[k + l * n] * pRangeKernel[dI];
				weightSumma += temp;
				summa += temp * intensity_kl;
			}
		}

		in[offset] = (weightSumma == 0.0f) ? 0 : (unsigned char)(summa / weightSumma + 0.5f);		//így kerekíteni fog
	}
}
