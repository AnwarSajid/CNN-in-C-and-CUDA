/*
    <A simulation envoirnment for Deep Convolution Neural Networks>
    Copyright (C) <2012-213>  <Sajid Anwar> <engrsajidanwar@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "cnn.h"
#include "layer.h"
#include "dataset.h"
#include "batch.h"
#include "debug_extra.h"
#include "error_handling.h"

cnnlayer_t* create_cnn(cnn_t* cnn_specs)
{
	//first create numlayers number of layers
	int numlayers = cnn_specs->nlayers;

	cnnlayer_t* head = NULL;
	cnnlayer_t* current = NULL;
	cnnlayer_t* temp = NULL; 

	int counter = 0;
	for (counter  = 0; counter < numlayers; counter++)
	{
		current = (cnnlayer_t *) malloc(sizeof(cnnlayer_t));
		current->no_of_fmaps = cnn_specs->no_fmaps[counter];
		current->no_of_neurons = cnn_specs->fmap_size[counter] * cnn_specs->no_fmaps[counter];
		current->fkernel = cnn_specs->fkernel[counter];

		current->layer_type = cnn_specs->layer_type;
		current->pool_type = cnn_specs->pool_type;

		current->fmap_height = sqrt(cnn_specs->fmap_size[counter]);
		current->fmap_width = sqrt(cnn_specs->fmap_size[counter]);

		long hsize = sizeof(real_t) * current->no_of_neurons * BATCH_SIZE;
		printf("\n hsize: %ld", hsize/sizeof(real_t));
		current->neurons_input = (real_t *) malloc(hsize);
		memset(current->neurons_input, 0, hsize); 

		current->error_deltas = (real_t *) malloc(hsize);
		memset(current->error_deltas, 0, hsize); //clear at start of each batch 
		HANDLE_ERROR(cudaMalloc((void **) &current->d_error_deltas, hsize));
		HANDLE_ERROR(cudaMemset(current->d_error_deltas, 0, hsize));

		current->neurons_output = (real_t *) malloc(hsize);
		HANDLE_ERROR(cudaMalloc((void **) &current->d_neurons_output, hsize));
		HANDLE_ERROR(cudaMemset(current->d_neurons_output, 0, hsize));

		current->subsampling = false;

		if (counter == 2 || counter == 4)
		{
			current->subsampling = true;
			int gMapSize = sizeof(int) * current->no_of_neurons * BATCH_SIZE;
			current->gradientMap = (int *) malloc(gMapSize);
			memset(current->gradientMap, 0, gMapSize);

			HANDLE_ERROR(cudaMalloc((void **) &current->d_gradientMap, gMapSize));
			HANDLE_ERROR(cudaMemset(current->d_gradientMap, 0, gMapSize));
		}
		else
		{
			current->subsampling = false;
			current->gradientMap = NULL;
			current->d_gradientMap = NULL;
		}

		current->dy_output = (real_t *) malloc(hsize); 

		current->weights_matrix = NULL;
		current->delta_weights = NULL;
		current->d_delta_weights = NULL;

		current->biases = NULL;
		current->d_biases = NULL;
		current->delta_biases = NULL;
		current->d_delta_biases = NULL;
		current->isclamped = false;		

		if (counter == 0)
		{
			head = current;
			current->previous = NULL;
			current->next = NULL;
			temp = current;
		}

		if (counter == numlayers - 1)
		{
			temp->next = current;
			current->next = NULL;
			current->previous = temp;
			//fprintf(stderr,"\ncurrent:%p, next:%p, prev:%p", current, current->next, current->previous);
			continue;
		}

		//handle next and previous pointers for cases other than head and tail
		if (counter != 0)
		{
			temp->next = current;
			current->previous = temp;
			current->next = NULL;
			temp = current;
		}
	}

	return head;  	
}

int reset_parameters_to_high_precision(cnnlayer_t* headlayer)
{
	cnnlayer_t* current = headlayer;
	cnnlayer_t* next_to_current = current->next;

	FILE *fp = NULL;
	fp = fopen("best_net_1","r"); 
	if (fp == NULL)
		return -1; 

	int layercounter = 0;
	bool_t flag = true;
	while (current != NULL && flag == true)
	{
		if (next_to_current->next == NULL)
			flag = false;

		int src_nfmaps = current->no_of_fmaps;
		int dst_nfmaps = next_to_current->no_of_fmaps;
		int fkernel = current->fkernel;

		int weight_matrix_size = 0, biases_size = 0;
		if (next_to_current->subsampling == true) 
		{
			if (src_nfmaps != dst_nfmaps)
			{
				fprintf(stderr, "\nError! Subsampling layer fmaps != Conv layer fmaps");
				return -1;
			}

			weight_matrix_size = src_nfmaps; 
			biases_size = dst_nfmaps;
		}
		else
		{
			weight_matrix_size = src_nfmaps * dst_nfmaps * fkernel * fkernel;	
			biases_size = dst_nfmaps;
			weight_matrix_size = src_nfmaps * dst_nfmaps * fkernel * fkernel;	
			biases_size = dst_nfmaps;
		}


		int counter = 0;
		for (counter = 0; counter < weight_matrix_size; counter++)
		{

			float temp = 0;
			int sizer = fscanf(fp,"%f",&temp);
			if (sizer > 0)
				current->weights_matrix[counter] = (real_t) temp;
			else
				current->weights_matrix[counter] = (real_t) temp;

			current->delta_weights[counter] = 0.0;
		}

		for (counter = 0; counter < biases_size; counter++)
		{
			float temp = 0;
			int sizer = fscanf(fp,"%f", &temp);
			if (sizer > 0)
				current->biases[counter] = (real_t) temp;
			else
				current->biases[counter] = (real_t) temp;

			current->delta_biases[counter] = 0.0; 
		}

		layercounter++;	

		if (flag == true)
		{
			current = next_to_current;
			next_to_current = current->next;
		}
	}

	if (fp != NULL)
		fclose(fp);

	return 0;
}


int initialize_weights_matrices(cnnlayer_t* headlayer, bool_t generate_random)
{
	cnnlayer_t* current = headlayer;
	cnnlayer_t* next_to_current = current->next;

	FILE *fp = NULL;
	/* Initialize a network with pre trained network for MNIST experiment */
	fp = fopen("best_net_1","r");
	//fp = fopen("big_nw_init", "r"); 

	/* Initialize a network with pre trained network for Eye state detection experiment */
	//fp = fopen("ESD/real_ESD_150_4.63","r");

	/* If the files can't be opened, initialize weights and biases randomly */
	if (fp == NULL)
		generate_random = true;

	int layercounter = 0;
	bool_t flag = true;
	while (current != NULL && flag == true)
	{
		if (next_to_current->next == NULL)
			flag = false;

		int src_nfmaps = current->no_of_fmaps;
		int dst_nfmaps = next_to_current->no_of_fmaps;
		int fkernel = current->fkernel;

		int weight_matrix_size = 0, biases_size = 0;
		if (next_to_current->subsampling == true) 
		{
			if (src_nfmaps != dst_nfmaps)
			{
				fprintf(stderr, "\nError! Subsampling layer fmaps != Conv layer fmaps");
				return -1;
			}

			weight_matrix_size = src_nfmaps; 
			biases_size = dst_nfmaps;
		}
		else
		{
			weight_matrix_size = src_nfmaps * dst_nfmaps * fkernel * fkernel;	
			biases_size = dst_nfmaps;
		}

		//allocate weights and delta weights matrices
		current->weights_matrix  = (real_t *) malloc(weight_matrix_size * sizeof(real_t));
		current->delta_weights  = (real_t *) malloc(weight_matrix_size * sizeof(real_t));

		HANDLE_ERROR(cudaMalloc((void **) &current->d_weights, weight_matrix_size * sizeof(real_t)));
		HANDLE_ERROR(cudaMalloc((void **) &current->d_delta_weights, weight_matrix_size * sizeof(real_t)));
		HANDLE_ERROR(cudaMemset(current->d_delta_weights, 0, weight_matrix_size * sizeof(real_t)));

		//allocate biases and delta biases	
		current->biases = (real_t *) malloc(biases_size * sizeof(real_t));
		current->delta_biases = (real_t *) malloc(biases_size * sizeof(real_t));

		HANDLE_ERROR(cudaMalloc((void **) &current->d_biases, biases_size * sizeof(real_t)));
		HANDLE_ERROR(cudaMemset(current->d_biases, 0, biases_size * sizeof(real_t)));

		HANDLE_ERROR(cudaMalloc((void **) &current->d_delta_biases, biases_size * sizeof(real_t)));
		HANDLE_ERROR(cudaMemset(current->d_delta_biases, 0, biases_size * sizeof(real_t)));

		int counter = 0;
		for (counter = 0; counter < weight_matrix_size; counter++)
		{
			//initialize weights
			if (generate_random == true)
			{   
				real_t tw = (real_t) rand() / RAND_MAX;
				real_t lr = -0.1;
				real_t ur = 0.1;
				current->weights_matrix[counter] = lr + (ur - lr) * tw;

				current->delta_weights[counter] = 0.0;
			}
			else
			{
				float temp = 0;
				int sizer = fscanf(fp,"%f",&temp);
				if (sizer > 0)
					current->weights_matrix[counter] = (real_t) temp;
				else
					current->weights_matrix[counter] = (real_t) temp;

				current->delta_weights[counter] = 0.0;
			}
		}

		for (counter = 0; counter < biases_size; counter++)
		{
			if (generate_random == true) 
			{
				real_t tb = (real_t) rand() / RAND_MAX;
				real_t lr = 0.00;
				real_t ur = 0.01;
				current->biases[counter] = lr + (ur - lr) * tb;
				//current->biases[counter] = 0.0; //for debugging only
				current->delta_biases[counter] = 0.0; 
			}
			else
			{   
				float temp = 0;
				int sizer = fscanf(fp,"%f", &temp);
				if (sizer > 0)
					current->biases[counter] = (real_t) temp;
				else
					current->biases[counter] = (real_t) temp;

				//current->biases[counter] = 0.0; //for debugging only
				current->delta_biases[counter] = 0.0; 
			}
		}

		layercounter++;	

		if (flag == true)
		{
			current = next_to_current;
			next_to_current = current->next;
		}
	}

	if (fp != NULL)
		fclose(fp);

	return 0;
}

void destroy_cnn(cnnlayer_t* headlayer)
{
	cnnlayer_t* temp;
	cnnlayer_t* current;
	current = headlayer;

	for (current = headlayer; current != NULL; ) 
	{
		temp = current->next;
		if (current->weights_matrix != NULL)
			free(current->weights_matrix);

		if (current->d_weights != NULL)
			cudaFree(current->d_weights);

		if (current->delta_weights != NULL)
			free(current->delta_weights);	

		if (current->d_delta_weights != NULL)
			cudaFree(current->d_delta_weights);	

		if (current->biases != NULL)
			free(current->biases);

		if (current->d_biases != NULL)
			cudaFree(current->d_biases);

		if (current->delta_biases != NULL)
			free(current->delta_biases);

		if (current->d_delta_biases != NULL)
			cudaFree(current->d_delta_biases);

		if (current->neurons_input != NULL)
			free(current->neurons_input);

		if (current->gradientMap != NULL)
			free(current->gradientMap);

		if (current->d_gradientMap != NULL)
			cudaFree(current->d_gradientMap);

		if (current->neurons_output != NULL)
			free(current->neurons_output);

		if (current->d_neurons_output != NULL)
			cudaFree(current->d_neurons_output);

		if (current->dy_output != NULL)
			free(current->dy_output);

		if (current->error_deltas != NULL)
			free(current->error_deltas);

		if (current->d_error_deltas != NULL)
			cudaFree(current->d_error_deltas);

		// freeing and advancing current 
		free(current);
		current = temp; //effectively it is current = current -> next
	}
}

void free_cnn_specs(cnn_t* cnn)
{
	//printf("\n free cnn specs");
	if (cnn->no_fmaps != NULL)
		free(cnn->no_fmaps);

	if (cnn->fmap_size != NULL)
		free(cnn->fmap_size);

	if (cnn->fkernel != NULL)
		free(cnn->fkernel);

	if (cnn != NULL)
		free(cnn);
}
