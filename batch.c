#include <stdlib.h>
#include <stdio.h>
#include "batch.h"
#include "layer.h"
#include <time.h>

//generate and shuffle the input data
void mini_batching(int* batch_indexes, int nvecs, bool_t shuffle)
{
	int counter = 0;
	for (counter = 0; counter < nvecs; counter++)
	{
		batch_indexes[counter] = counter;
	}

    if (shuffle != 0)
    {
	    int shuffleSize = nvecs;
	    for (counter = 0; counter < shuffleSize; counter++)
	    {
		    int randIdx1 = rand() % nvecs;
            int randIdx2 = rand() % nvecs;
            int temp = 	batch_indexes[randIdx1];
            batch_indexes[randIdx1] = batch_indexes[randIdx2];
            batch_indexes[randIdx2] = temp;
        }
    }    
}
