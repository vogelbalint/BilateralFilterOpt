#ifndef CONSTSIZE_H
#define CONSTSIZE_H

//A spatial kernel sugarának maximuma
#define MAX_RADIUS 16		

//a spatial kernelt tároló konstans memória tömb mérete (elemszáma) 
#define CONST_SIZE ( (MAX_RADIUS * 2 + 1)*(MAX_RADIUS * 2 + 1) )


#endif		//CONSTSIZE_H
