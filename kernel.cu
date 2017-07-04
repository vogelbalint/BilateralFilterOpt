#include <math.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"
#include "constsize.h"

//itt defini�ljuk a konstans mem�ria t�mb�t (a t�bbi helyen csak deklar�ljuk)
__constant__ float data[CONST_SIZE];

//Gauss f�ggv�nyt sz�m�t. x n�gyzet�t kell neki �tadni, �s a sz�r�st (sigma)
__host__ __device__ float gauss(float x_square, float sigma)
{
	return expf(-x_square / (2 * sigma * sigma));
}

//felt�lti a spatial kernelt a konstans mem�ri�ban
bool fillConstantMemory(int r, float sigma_s)
{
	int n = 2 * r + 1;			//a spatial kernel (mint m�trix) oldal�nak hossza
	int spatialKernelSize = n * n;

	//el�sz�r a host-on foglalunk mem�ri�t �s a CPU sz�m�tja ki a spatial kernelt.
	//r�vid id�re kell mem�ria a host-on, ezt foglalhatjuk cudaHostAlloc -kal, mert csak r�vid id�re cs�kken a kilapozhat� mem�ria mennyis�ge.
	//cser�be gyorsabb lesz a host --> device m�sol�s
	float *h_data;	 
	if (cudaHostAlloc((void**)&h_data, spatialKernelSize * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {	//TODO: egy�b flag-eket �ll�tani?
		printf("cudaHostAlloc failed!\n\n");
		return false;
	}
	
	//a spatial kernel kisz�m�t�sa:
	//int pos = 0;
	for (int j = -r; j <= r; ++j) {
		int y = j + r;
		for (int i = -r; i <= r; ++i) {
			int x = i + r;
			h_data[x + y * n] = gauss((float)(i * i + j * j), sigma_s);	
			//h_data[pos++] = gauss((float)(i * i + j * j), sigma_s);
		}
	}

	//m�sol�s host --> constant memory
	if (cudaMemcpyToSymbol(data, h_data, spatialKernelSize * sizeof(float)) != cudaSuccess) {
		printf("cudaMemcpyToSymbol failed!\n\n");
		cudaFreeHost(h_data);
		return false;
	}
	cudaFreeHost(h_data);
	return true;
}

//A lehets�ges intenzit�sbeli k�l�nbs�gekhez (dI) tartoz� Gauss �rt�keket is el�re kisz�m�tjuk, az ezt t�rol� t�mb�t h�vom range kernel -nek
//A range kernelt majd bem�soljuk a shared memory-ba, mert ebb�l sokszor kell majd olvasnunk.
//a legkisebb dI -255 , a legnagyobb dI 255 ==> a t�mb 255 *2 + 1 = 511 elemt tartalmaz
//A t�mb k�zep�n van a dI = 0 -hoz tartoz� Gauss �rt�k
//a f�ggv�ny param�terk�nt �tveszi dI maximum�t is (maxRangeDiff), hogy ne legyenek a k�dban var�zs sz�mok. 
__global__ void createRangeKernel(float *rangeKernel, float sigma, int maxRangeDiff)
{
	//el�sz�r csak a pozit�v delte I -khez tartoz� Gausst sz�m�tjuk ki, mert a Gauss f�ggv�ny szimmetrikus
	int tid = threadIdx.x;
	if (tid >= maxRangeDiff) {
		int deltaI = tid - maxRangeDiff;
		rangeKernel[tid] = gauss((float)(deltaI * deltaI), sigma);
	}

	__syncthreads();

	//�tm�soljuk a negat�v intenzit�s k�l�nbs�g �rtkekhez tartoz� Gauss �rt�keket a t�mb m�sodik fel�b�l
	int last = maxRangeDiff * 2;  //=510
	if (tid < maxRangeDiff) {
		rangeKernel[tid] = rangeKernel[last - tid];
	}
}

//seg�df�ggv�ny a bilateralFilter cuda kernelhez.
//a bilateralFilter cuda kernelben a k�p egy r�sz�t a shared memory-ba m�soljuk, de t�bb pixelt kell m�solni, mint az adott blokk �sszes thread-j�nek sz�ma.
//ez�rt a blokk sz�lein�l szenvedni kell, a sz�lekhez tartoz� pixelek eset�n ki kell sz�m�tani egy eltol�st.
//ezt neh�z �r�sban �tadni, pap�ron rajzolgatva lehet meg�rteni. 
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

//a bilateral filtert megval�s�t� cuda kernel
//a spatial kernel elemeit konstans mem�ri�b�l �ri el, a range kernel elemeit glob�lis mem�ri�b�l a shared mem�ri�ba m�soljuk
//s�t: az adott blokkhoz tartoz� processz�land� pixeleket (intenzit�s �rt�keket) is bem�soljuk a shared memory-ba
//de: nem csak ezeket a pixeleket kell bem�solni, hanem azokat is, amikre sz�ks�g van a pixelek feldolgoz�s�hoz.
//vagyis egy r "sz�les�g�" s�vot is be kell m�solni plusszban a sz�lekhez (r a spatial kernel sugara) --> ez nem trivi�lis
//egy�b probl�ma: a shared memory-ban unsigned char (k�p) �s float (range kernel) elemeket is t�rolunk,
//ez�rt a shared mem�ria t�mbj�t char t�mbk�nt deklar�ljuk �s megfelel� t�pus� pointereket �ll�tunk a megfelel� helyekre.
//Megj.: hogy az algoritmus ne legyen nagyon bonyolult, nem processz�ljuk azokat a pixeleket, amik a (teljes) k�p sz�lein vannak
//(a k�p hat�r�hoz k�zeebb van r -n�l). Ez egyr�szt cs�kkenti a thread divergenci�t, m�sr�szt nem kell annyi mindent �tgondolni
//Megj.: mivel a k�pet bem�soljuk a shared mem�ri�kba (r�szenk�nt), nem kell k�l�n output k�p, az input k�pet fel�l�rjuk
//(ehhez persze thread szinkroniz�l�s kell).
//width, height: a k�pe sz�less�ge �s magass�ga pixelben
__global__ void bilateralFilter(unsigned char *in, float *rangeKernel, int r, int maxRangeDiff, int width, int height)
{
	extern __shared__ char sharedData[];				

	int x = threadIdx.x + blockIdx.x * blockDim.x + r;	//az adott threadhez tartoz� pixel koordin�t�i
	int y = threadIdx.y + blockIdx.y * blockDim.y + r;

	unsigned char *pImagePart = reinterpret_cast<unsigned char*>(sharedData);	//a shared memory k�pr�szletet t�rol� r�sz�nek elej�re mutat� pointer

	int imagePartWidth = blockDim.x + 2 * r;		//a blokkban t�rolt k�pr�szlet oldal�nak hossza. Az algoritmusban blockDim.x = blockDim.y
	int imagePartSize = imagePartWidth * imagePartWidth;

	float *pRangeKernel = reinterpret_cast<float*>(sharedData + imagePartSize * sizeof(unsigned char));	//a range kernel r�sz elej�re mutat

	//felt�ltj�k a shared memory range kernel r�sz�t
	int index = threadIdx.x + blockDim.x * threadIdx.y;  //minden thread-nek az adott blokkban van egy egyedi indexe
	int step = blockDim.x * blockDim.y;					//az �sszes thread sz�ma a blockban
	int rangeKernelSize = 2 * maxRangeDiff + 1;			//=511
	while (index < rangeKernelSize) {
		pRangeKernel[index] = rangeKernel[index];
		index += step;
	}
	pRangeKernel += maxRangeDiff;		//most a range kernelt t�rol� r�sz k�zep�re mutat (a dI = 0 -hoz tartoz� r�szre)

	int offset = x + y * width;			//az adott pixel helye a k�p unsigned char t�mbj�ben

	int i = threadIdx.x, j = threadIdx.y;	//"t�vols�g a blokk oldal�t�l"

	int ii = i + r, jj = j + r;				//t�vols�g a bem�solt k�pr�szlet oldal�t�l, ezek a k�pr�szletben indexelnek. 
	
	if (x < width - r &&  y < height - r) {

		pImagePart[ii + jj * imagePartWidth] = in[offset];	//a k�pr�szlet belsej�nek m�sol�sa

		int dx = 0, dy = 0;							//seg�dv�ltoz�k a k�pr�szlet sz�leinek �tm�sol�s�hoz

		//�tm�soljuk a sz�leket:
		computeShift(r, blockDim.x, i, j, dx, dy);		//TODO: lehetne inline -olni? �rdemes-e?

		if (dx != 0)
			pImagePart[(ii + dx) + jj * imagePartWidth] = in[(x + dx) + y * width];
		if (dy != 0)
			pImagePart[ii + (jj + dy) * imagePartWidth] = in[x + (y + dy) * width];
		if (dx != 0 && dy != 0)	
			pImagePart[(ii + dx) + (jj + dy) * imagePartWidth] = in[(x + dx) + (y + dy) * width];
	}

	//kell szinkroniz�l�s, mert ezel�tt �runk a shared memory-ba, ezut�n olvasunk bel�le.
	//nem felt�tlen�l j� a __syncthreads() egy if-ben, ez�rt az if-t sz�tszedtem k�t r�szre
	__syncthreads();

	if (x < width - r &&  y < height - r) {

		int n = 2 * r + 1;									//spatial kernel oldal�nak hossza
		float *pSpatialKernel = data  + (r + r * n);		//a spatial kernel k�zep�re mutat

		//j�n a l�nyeg, sok szenved�s ut�n:
		int intensity = pImagePart[ii + jj * imagePartWidth];	//az adot pixel intenzit�sa

		float summa = 0.0f, weightSumma = 0.0f;			//a weightSumma csak a s�lyokat �sszegzi

		for (int l = -r; l <= r; ++l) {			//l: sorindex
			for (int k = -r; k <= r; ++k) {		//k: oszlopindex

				int intensity_kl = pImagePart[(ii + k) + (jj + l) * imagePartWidth];
				int dI = intensity - intensity_kl;
				float temp = pSpatialKernel[k + l * n] * pRangeKernel[dI];
				weightSumma += temp;
				summa += temp * intensity_kl;
			}
		}

		in[offset] = (weightSumma == 0.0f) ? 0 : (unsigned char)(summa / weightSumma + 0.5f);		//�gy kerek�teni fog
	}
}
