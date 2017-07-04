#ifndef CONSTANT_H
#define CONSTANT_H

#include "constsize.h"

//a lehets�ges t�rbei k�l�nbs�gekhez tartoz� Gauss �rt�keket el�re kisz�m�tjuk, ez a t�rbeli kernel (spatial kernel)
//ezt a t�mb�t konstans mem�ri�ban t�roljuk
//ez az�rt j�, mert a bilater�l filter sz�m�t�sa sor�n _minden_ pixel kiolvassa az _�sszes_ t�rbeli elt�r�s �rt�ket.
//pont erre j� a constant memory (warp, broadcast, stb.) 
//h�tr�ny: a spatial kernel m�rete parancssori argumentummal �ll�that�, viszont a konstans t�mb m�ret�t ford�t�si id�ben kell ismerni.
//a t�mb m�ret�t megad� makr�k a constsize.h -ban vannak
extern __constant__ float data[CONST_SIZE];


#endif // !CONSTANT_H