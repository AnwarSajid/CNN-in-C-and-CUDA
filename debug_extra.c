#include <stdio.h>
#include "dataset.h"
#include "layer.h"

void display_weights_matrices(cnnlayer_t *headlayer)
{
	struct nnlayer* current = headlayer;
	struct nnlayer* next_to_current = current->next;	
		
	int lctr = 0;
	bool_t flag = true;
	while(current != NULL && flag == true)
	{	
		if (next_to_current->next == NULL)
			flag = false;

		int src_layer_nfmaps = current->no_of_fmaps; 
		int dst_layer_nfmaps = next_to_current->no_of_fmaps;
		int fkernel = current->fkernel;
		int weights_size;

		if (next_to_current->subsampling == true)
			weights_size = src_layer_nfmaps;
		else
			weights_size = src_layer_nfmaps * dst_layer_nfmaps * fkernel * fkernel;
 
		int counter = 0;
		printf("\n Layer: %d ", lctr++);
		for (counter = 0; counter < weights_size; counter++)
		{
			if (counter % 25 == 0)
				printf("\n");
  			printf("\t%f", current->weights_matrix[counter]);
		}
		
		printf("\n biases ");
		for (counter = 0; counter < dst_layer_nfmaps; counter++)
		{
			if (counter % 25 == 0)
				printf("\n");
  			printf("\t%f", current->biases[counter]);
		}
				

		if (flag == true)
		{
			current = next_to_current;
			next_to_current = current->next;
		}
	}
}

void display_cnn_layers(cnnlayer_t* headlayer)
{
    cnnlayer_t *current, *next_to_current;
    current = headlayer;
	next_to_current = current->next;

    int count = 0;
	bool_t flag = true;
    while(current != NULL && flag == true)
    {	
		if (next_to_current->next == NULL)
			flag = false;

        if (count >= 0)
        {

            int no_of_neurons = current->no_of_neurons;
            int counter = 0;


            printf("\nnNeurons: %d, Layer: %d", no_of_neurons, count);
            /*
            printf("\nCurrentInput");
            for (counter = 0; counter < no_of_neurons; counter++)
            {
                if (counter % current->fmap_width == 0)
                    printf("\n");
                printf("\t%f", current->neurons_input[counter]);
            }
            */
            printf("\nCurrentOutput");
            for (counter = 0; counter < no_of_neurons; counter++)
            {
                if (counter % (current->fmap_width * current->fmap_height) == 0 && current->fmap_height != 1)
                    printf("\n");
                if (counter % current->fmap_width == 0)
                    printf("\n");
                printf("\t%f", current->neurons_output[counter]);
            }
       
            printf("\nCurrentWeightsMatrix, test:%d", next_to_current->subsampling);
            int no_of_weights; 
            if (next_to_current->subsampling == false)
                no_of_weights = next_to_current->no_of_fmaps * current->no_of_fmaps * current->fkernel * current->fkernel;
            else
                no_of_weights = current->no_of_fmaps;
            
            for (counter = 0; counter < no_of_weights; counter++)
            {
                if (counter % (current->fkernel * current->fkernel) == 0 && current->fmap_height != 1)
                    printf("\n");
                if (counter % current->fkernel == 0)
                    printf("\n");
                printf("\t%f", current->weights_matrix[counter]);
            }
       
             printf("\nCurrent Biases");
            int no_of_biases; 
            no_of_biases = next_to_current->no_of_fmaps;
            
            for (counter = 0; counter < no_of_biases; counter++)
            {
                printf("\t%f", current->biases[counter]);
            }
             
            printf("\nCurrentDeltaWeightsMatrix, test:%d", next_to_current->subsampling);
            if (next_to_current->subsampling == false)
                no_of_weights = next_to_current->no_of_fmaps * current->no_of_fmaps * current->fkernel * current->fkernel;
            else
                no_of_weights = current->no_of_fmaps;
            
            for (counter = 0; counter < no_of_weights; counter++)
            {
                if (counter % current->fkernel == 0)
                    printf("\n");
                printf("\t%f", current->delta_weights[counter]);
            }
            
            printf("\nCurrenterror_deltas");
            for (counter = 0; counter < no_of_neurons; counter++)
            {
                if (counter % current->fmap_width == 0)
                    printf("\n");
                printf("\t%f", current->error_deltas[counter]);
            }
        }

        if (flag == true)
        {
            current = next_to_current;
            next_to_current = current->next;
        }

        count++;
    }

	printf("\n Layer : %d Last layer", count);
	//if (next_to_current->next == NULL && next_to_current != NULL)
	if (1)
	{
		current = next_to_current;
		int no_of_neurons = current->no_of_neurons;
		int counter = 0;

		printf("\nCurrentInput, nNeurons: %d, Layer: %d", no_of_neurons, count);
		/*for (counter = 0; counter < no_of_neurons; counter++)
		{
			if (counter % current->fmap_width == 0)
				printf("\n");
			printf("\t%f", current->neurons_input[counter]);
		}
        */
		printf("\nCurrentOutput");
		for (counter = 0; counter < no_of_neurons; counter++)
		{
			if (counter % current->fmap_width == 0)
				printf("\n");
			printf("\t%f", current->neurons_output[counter]);
		}
		/*
		if (current != headlayer)
		{
			printf("\nCurrentdy_output");
			for (counter = 0; counter < no_of_neurons; counter++)
			{
				if (counter % current->fmap_width == 0)
					printf("\n");
				printf("\t%f", current->dy_output[counter]);
			}
		}
        */

		printf("\nCurrenterror_deltas");
		for (counter = 0; counter < no_of_neurons; counter++)
		{
			if (counter % current->fmap_width == 0)
				printf("\n");
			printf("\t%f", current->error_deltas[counter]);
		}
	}
	
	printf("\n");
}

void display_gradientMap(cnnlayer_t* headlayer)
{
    cnnlayer_t* current = headlayer;
	cnnlayer_t* next_to_current = current->next;

	bool_t flag = true;
    while(current != NULL && flag == true)
    {	
		if (next_to_current->next == NULL)
			flag = false;

        if (next_to_current->subsampling == true)
        {
            int currnUnits = current->no_of_neurons;     
            int nextnUnits = next_to_current->no_of_neurons;     

            //display current outputs
            printf("\n Current Output : ");
            int counter = 0;
            for (counter = 0; counter < currnUnits; counter++)
            {
                if (counter % (current->fmap_width * current->fmap_height) == 0 && current->fmap_height != 1)
                    printf("\n");
                if (counter % current->fmap_width == 0)
                    printf("\n");
                printf("\t%f", current->neurons_output[counter]);
            }

            //display gradientMap
            printf("\n Current gradientMap : ");
            for (counter = 0; counter < nextnUnits; counter++)
            {
                if (counter % (next_to_current->fmap_width * next_to_current->fmap_height) == 0 && next_to_current->fmap_height != 1)
                    printf("\n");
                if (counter % next_to_current->fmap_width == 0)
                    printf("\n");
                printf("\t%d", next_to_current->gradientMap[counter]);
            }
        }
                
        if (flag == true)
        {
            current = next_to_current;
            next_to_current = current->next;
        }
    }

	printf("\n");
}

void display_dataVector(struct dataset* dataVector)
{
	int lvec = dataVector->lenVector;
	int nvec = dataVector->numVectors;

	int ocounter = 0, icounter = 0;	
	for (ocounter = 0; ocounter < nvec; ocounter++)
	{
		fprintf(stderr, "\n");
		for (icounter = 0; icounter < lvec; icounter++)
		{
			fprintf(stderr, "\t%f", dataVector->data[ocounter *  lvec + icounter]);
		}
	}
}
