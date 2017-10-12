#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "dataset.h"

dataset_t* create_data_container(int numVectors, int lenVector, int each_output_size)
{
	dataset_t* dataVector = (dataset_t *) malloc(sizeof(dataset_t));
	dataVector->numVectors = numVectors;
	dataVector->lenVector = lenVector;
	dataVector->lenlable = each_output_size;

	int dSize = sizeof(double) * numVectors * lenVector;
	dataVector->data = (double *) malloc(dSize);	
	memset(dataVector->data, 0, dSize);

	dataVector->lables = (unsigned char*) malloc(numVectors * sizeof(unsigned char));

	if (dataVector == NULL)
		return NULL;

	return dataVector;
}

void free_data_container(dataset_t* dataVector)
{
	//printf("\nFree_data_container");
	if (dataVector->data != NULL) 
		free(dataVector->data);
		
	dataVector->data = NULL;
	
	if (dataVector->lables != NULL)
		free(dataVector->lables);

	dataVector->lables = NULL;

	if (dataVector != NULL)
		free(dataVector);

	dataVector = NULL;
}

void generate_lables_eye_closure(dataset_t* training_samples, dataset_t* test_samples)
{
    //currently we have training = 8232 (open) + 3958 (closed) samples
    //and for test samples, we have = 1230 (open) + 410 (closed) samples
    int counter = 0;
    for (counter = 0; counter < 8232; counter += 1)
    {
        training_samples->lables[counter] = 0; //lable zero means open
    }

    for (counter = 8232; counter < (8232 + 3958); counter += 1)
    {
        training_samples->lables[counter] = 1;//lable one means closed
    }


    for (counter = 0; counter < 1230; counter += 1)
    {
        test_samples->lables[counter] = 0;
    }

    for (counter = 1230; counter < (1230 + 410); counter += 1)
    {
        test_samples->lables[counter] = 1;
    }
}

int argmax(double p1, double p2, double p3, double p4, int count)
{
	int maxIdx = 0;
	if (count == 2)
	{
		if (p1 >= p2)
			maxIdx = 0;
		else 
			maxIdx = 1;
	}
	else if (count == 4)
	{
		if (p1 >= p2 && p1 >= p3 && p1 >= p4)
		{
			maxIdx = 0;//p1;
		}
		else if (p2 >= p1 && p2 >= p3 && p2 >= p4)
		{
			maxIdx = 1;//p2;
		}
		else if (p3 >= p1 && p3 >= p2 && p3 >= p4)
		{
			maxIdx = 2;//p3;
		}
		else if (p4 >= p1 && p4 >= p2 && p4 >= p3)
		{
			maxIdx = 3;//p4;
		}
	}
	else
	{
		fprintf(stderr,"\nError! such Pooling Not exptected");
	}
	
	return maxIdx;
}

double pool_max(double p1, double p2, double p3, double p4, int* idx, int count)
{
	double max = 0;
	if (count == 2)
	{
		if (p1 >= p2)
		{
			max = p1;
			*idx = 0;
		}
		else 
		{
			max = p2;
			*idx = 1;
		}
	}
	else if (count == 4)
	{
		if (p1 >= p2 && p1 >= p3 && p1 >= p4)
		{
			max = p1;
			*idx = 0;
		}
		else if (p2 >= p1 && p2 >= p3 && p2 >= p4)
		{
			max = p2;
			*idx = 1;
		}
		else if (p3 >= p1 && p3 >= p2 && p3 >= p4)
		{
			max = p3;
			*idx = 2;
		}
		else if (p4 >= p1 && p4 >= p2 && p4 >= p3)
		{
			max = p4;
			*idx = 3;
		}
	}
	else
	{
		fprintf(stderr,"\nError! such subsampling Not exptected");
	}
	
	return max;
}

void save_trained_network_weights(cnnlayer_t* headlayer, char*  filename) 
{ 
   cnnlayer_t* current; 
   cnnlayer_t* next_to_current; 
   current = headlayer; 
   next_to_current = current->next; 
 
	bool_t flag = true;

   FILE *fp = fopen(filename, "w");
   if (fp != NULL)
   {
      while (current != NULL && flag == true)
      {
         if (next_to_current->next == NULL)
            flag = false;

			int no_of_weights  = 0;
			if (next_to_current->subsampling == false)
				no_of_weights = current->no_of_fmaps * next_to_current->no_of_fmaps * current->fkernel * current->fkernel;
			else
				no_of_weights = next_to_current->no_of_fmaps;

         int counter = 0;
         for (counter = 0; counter < no_of_weights; counter++)
         {
            fprintf(fp, "%2.6f\n", current->weights_matrix[counter]);
         }

		//biases
		for (counter = 0; counter < next_to_current->no_of_fmaps; counter++)
        {
            fprintf(fp, "%2.6f\n", current->biases[counter]);
        }

         if (flag == true)
         {
            current = next_to_current;
            next_to_current = current->next;
         }
      }
   } 
   else
   {
      fclose(fp);
   }
}
