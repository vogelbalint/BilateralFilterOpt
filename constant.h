#ifndef CONSTANT_H
#define CONSTANT_H

#include "constsize.h"

//a lehetséges térbei különbségekhez tartozó Gauss értékeket elõre kiszámítjuk, ez a térbeli kernel (spatial kernel)
//ezt a tömböt konstans memóriában tároljuk
//ez azért jó, mert a bilaterál filter számítása során _minden_ pixel kiolvassa az _összes_ térbeli eltérés értéket.
//pont erre jó a constant memory (warp, broadcast, stb.) 
//hátrány: a spatial kernel mérete parancssori argumentummal állítható, viszont a konstans tömb méretét fordítási idõben kell ismerni.
//a tömb méretét megadó makrók a constsize.h -ban vannak
extern __constant__ float data[CONST_SIZE];


#endif // !CONSTANT_H