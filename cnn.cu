#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "cnn.h"
#include "layer.h"
#include "dataset.h"
#include "batch.h"
#include "debug_extra.h"
#include "error_handling.h"
#include "cudaKernels.h"
#include "timer.h"

double	LEARNING_RATE = 0.003;

//1. handle subsampling layer weights and biases
//                
int d_feed_forward(cnnlayer_t *headlayer, double *training_data, int *batch_indexes)
{
	int batchctr = 0;
	for (batchctr = 0; batchctr < BATCH_SIZE; batchctr++)
	{
		cnnlayer_t* current = headlayer;
		cnnlayer_t* next_to_current = current->next;
		int csample = batch_indexes[batchctr];
		int inp_vec_size = current->no_of_neurons;

		//headlayer's neurons input = output 
		int input_data_ctr = 0;
		for (input_data_ctr = 0; input_data_ctr < inp_vec_size; input_data_ctr++)
		{
			int first_layer_idx = batchctr * inp_vec_size + input_data_ctr;
			current->neurons_input[first_layer_idx] = (training_data + csample * inp_vec_size)[input_data_ctr];
			current->neurons_output[first_layer_idx] = (training_data + csample * inp_vec_size)[input_data_ctr];
		}	

	    int outIdx = batchctr * inp_vec_size;
        int outSize = inp_vec_size * sizeof(real_t);

        HANDLE_ERROR(cudaMemcpy(&current->d_neurons_output[outIdx], &current->neurons_output[outIdx], outSize, cudaMemcpyHostToDevice));
        	
		bool_t flag = true;	
		while (current != NULL && flag == true)
		{	
			if (next_to_current->next == NULL)
				flag = false;

			int src_fmaps = current->no_of_fmaps;
			int dst_fmaps = next_to_current->no_of_fmaps;
			int fkernel = current->fkernel;
			int bmargin = floor(fkernel/2);
			int imh = current->fmap_height;
			int imw = current->fmap_width;
			int next_imw = next_to_current->fmap_width;
			int next_imh = next_to_current->fmap_height;

			if (next_to_current->subsampling == false && current->fkernel != 1)
			{
				//convfmapolution layers
                //int kerSize = src_fmaps * dst_fmaps * fkernel * fkernel;
                //int kerBytes = kerSize * sizeof(real_t);
                //HANDLE_ERROR(cudaMemcpy(current->d_weights, current->weights_matrix, kerBytes, cudaMemcpyHostToDevice));

                real_t* d_output = next_to_current->d_neurons_output;
                real_t* d_input = current->d_neurons_output;
                real_t* d_kernel = current->d_weights;
                real_t* d_biases = current->d_biases;
               
                dim3 nBlocks(src_fmaps, dst_fmaps, 1); 
                dim3 nThreads(imw, imh, 1); 
                
                int sh_mem_size = imw * imh * sizeof(real_t) + fkernel * fkernel * sizeof(real_t);
                convolve_device_2D<<<nBlocks, nThreads, sh_mem_size>>>(d_output, d_input, d_kernel, fkernel * fkernel);
                compute_transfer_function<<<dst_fmaps, next_imw * next_imh >>>(d_output, d_biases, current->layer_type);
                cudaDeviceSynchronize();

                //int noutSize = (next_imw * next_imh * dst_fmaps) * sizeof(real_t);
                //HANDLE_ERROR(cudaMemcpy(next_to_current->neurons_output, d_output, noutSize, cudaMemcpyDeviceToHost));
			}
			else if (next_to_current->subsampling == false && current->fkernel == 1)
			{
			   	int src_layer_size = current->no_of_neurons;
				int dst_layer_size = next_to_current->no_of_neurons;

                //int weights_size = src_layer_size * dst_layer_size * sizeof(real_t);
                //HANDLE_ERROR(cudaMemcpy(current->d_weights, current->weights_matrix, weights_size, cudaMemcpyHostToDevice));

                real_t* d_input = current->d_neurons_output;
                real_t* d_output = next_to_current->d_neurons_output;
                real_t* d_weights = current->d_weights;
                real_t* d_biases = current->d_biases;

                dim3 nBlocks(dst_layer_size, 1, 1);
                dim3 nThreads(src_layer_size, 1, 1);

                int sh_mem_size = (2 * src_layer_size) * sizeof(real_t); 
                d_rear_DNN<<<nBlocks, nThreads, sh_mem_size>>>(d_output, d_input, d_weights);
                compute_transfer_function<<< dst_layer_size, 1 >>>(d_output, d_biases, current->layer_type);
                cudaDeviceSynchronize();

                //int noutSize = dst_layer_size * sizeof(real_t);
                //HANDLE_ERROR(cudaMemcpy(next_to_current->neurons_output, d_output, noutSize, cudaMemcpyDeviceToHost));
			}
			else if (next_to_current->subsampling == true)
			{
                // How to perform average pooling
				// Pattern Recognition and Machine Learning, By Christopher M. Bishop (P267)
				// ... Each subsampling unit might take inputs from a 2x2 unit region in the 
				// corresponding feature map and would compute the average of 
				// those inputs, multiplied by an adaptive weight with the addition of an adaptive bias
				// parameter, and then transformed using a sigmoidal non-linear activation function. 

                real_t* d_input = current->d_neurons_output;
                real_t* d_output = next_to_current->d_neurons_output;
                int* d_gradientMap = next_to_current->d_gradientMap;

                dim3 nBlocks(src_fmaps, 1, 1);
                dim3 nThreads(imw, imh, 1);

                int sh_mem_size = imw * imh * sizeof(real_t);
                d_subsampling<<<nBlocks, nThreads, sh_mem_size>>>(d_output, d_input, d_gradientMap, current->layer_type);
                cudaDeviceSynchronize();
                
                //int noutSize = (next_imw * next_imh * dst_fmaps) * sizeof(real_t);
                //HANDLE_ERROR(cudaMemcpy(next_to_current->neurons_output, d_output, noutSize, cudaMemcpyDeviceToHost));
                //HANDLE_ERROR(cudaMemcpy(next_to_current->gradientMap, d_gradientMap, noutSize, cudaMemcpyDeviceToHost));
			}
	
			if (flag == true)
			{
				current = next_to_current;
				next_to_current = current->next;
			}
		} 
	} 

	return 0;	
}

static __inline__ long long getticks( void )
{
    unsigned a, d;
    asm volatile("rdtsc" : "=a" (a), "=d" (d));
    return ((long long)a) | (((long long)d) << 32);
}

int h_feed_forward(cnnlayer_t *headlayer, double *training_data, int *batch_indexes)
{
    int batchctr = 0;
    for (batchctr = 0; batchctr < BATCH_SIZE; batchctr++)
    {
        cnnlayer_t* current = headlayer;
        cnnlayer_t* next_to_current = current->next;
        int csample = batch_indexes[batchctr];
        int inp_vec_size = current->no_of_neurons;

        //headlayer's neurons input = output 
        int input_data_ctr = 0;
        for (input_data_ctr = 0; input_data_ctr < inp_vec_size; input_data_ctr++)
        {
            int first_layer_idx = batchctr * inp_vec_size + input_data_ctr;
            current->neurons_output[first_layer_idx] = (training_data + csample * inp_vec_size)[input_data_ctr];
            current->neurons_input[first_layer_idx] = (training_data + csample * inp_vec_size)[input_data_ctr];
        }   
        
        bool_t flag = true; 
        while (current != NULL && flag == true)
        {   
            if (next_to_current->next == NULL)
                flag = false;

            int src_fmaps = current->no_of_fmaps;
            int dst_fmaps = next_to_current->no_of_fmaps;
            int fkernel = current->fkernel;
            int bmargin = floor(fkernel/2);
            int imh = current->fmap_height;
            int imw = current->fmap_width;
            int next_imw = next_to_current->fmap_width;
            int next_imh = next_to_current->fmap_height;

            if (next_to_current->subsampling == false && current->fkernel != 1)
            {
                //convolution layers
                int dst_layer_size = next_imw * next_imh * dst_fmaps; 
                int src_fmap_ctr = 0, dst_fmap_ctr = 0;
                for (dst_fmap_ctr = 0; dst_fmap_ctr < dst_fmaps; dst_fmap_ctr++)
                {
                    for (src_fmap_ctr = 0; src_fmap_ctr < src_fmaps; src_fmap_ctr++)
                    {
                        //weights do not involve batch counter
                        int weights_stidx = src_fmaps * dst_fmap_ctr * fkernel * fkernel;
                        int st_idx = weights_stidx + src_fmap_ctr * fkernel * fkernel; 

                        real_t* filter = &(current->weights_matrix[st_idx]); 

                        // Source layer feature maps starting index
                        int fmap_stidx = 0;
                        int src_layer_size = imh * imw * src_fmaps; 
                        fmap_stidx = batchctr * src_layer_size + src_fmap_ctr * imh * imw;

                        // Destination (layer) feature map starting index
                        int dst_fmap_stidx = batchctr * dst_layer_size + dst_fmap_ctr * next_imh * next_imw;
                        int dst_fmap_unit_ctr = 0;

                        int hctr = 0;
                        int wctr = 0;

                        for (hctr = 0; hctr < imh; hctr++)
                        {
                            for (wctr = 0; wctr < imw; wctr++)
                            {
                                if ((hctr >= bmargin && wctr >= bmargin) && (hctr < imh - bmargin && wctr < imw - bmargin)) 
                                {
                                    // Apply fitler kernel of size 5x5 to the input
                                    int cidx = fmap_stidx + hctr * imw + wctr;
                                    real_t sum = 0.0;

                                    int filterCtr = 0, convCtr1 = 0, convCtr2 = 0;
                                    for (convCtr1 = -1 * floor(current->fkernel/2); convCtr1 <= floor(current->fkernel/2); convCtr1++) 
                                    {
                                        for (convCtr2 = -1 * floor(current->fkernel/2); convCtr2 <= floor(current->fkernel/2); convCtr2++)
                                        {
                                            sum = sum + filter[filterCtr] * current->neurons_output[cidx + convCtr1 * imw + convCtr2];
                                            filterCtr++;
                                        }
                                    }

                                     //save summation to destination feature map
                                     int dst_idx = dst_fmap_stidx + dst_fmap_unit_ctr; 
                                    //next_to_current->neurons_input[dst_idx] += current->biases[dst_fmap_ctr]; 
                                     next_to_current->neurons_input[dst_idx] += sum; 
                                    
                                    if (src_fmap_ctr == src_fmaps - 1)
                                    {
                                         next_to_current->neurons_input[dst_idx] += current->biases[dst_fmap_ctr]; 
                                        real_t cn = next_to_current->neurons_input[dst_idx]; 
                                        next_to_current->neurons_input[dst_idx] = 0; 
 
                                        if (next_to_current->layer_type == 1)
                                            next_to_current->neurons_output[dst_idx] = sigmoid(cn); 
                                        else if (next_to_current->layer_type == 2)
                                        {
                                            next_to_current->neurons_output[dst_idx] = htangent(cn);
                                        }
                                        else if (next_to_current->layer_type == 3)
                                            next_to_current->neurons_output[dst_idx] = reLUSoftPlus(cn); 
                                    }

                                    dst_fmap_unit_ctr++;
                                }
                            }
                        }
                    }
                }
            }
            else if (next_to_current->subsampling == false && current->fkernel == 1)
            {
                int src_layer_size = current->no_of_neurons;
                int dst_layer_size = next_to_current->no_of_neurons;
                int dcounter = 0;
                int scounter = 0;
                real_t sum = 0.0;

                for (dcounter = 0; dcounter < dst_layer_size; dcounter++)
                {
                    sum = 0.0;
                    for (scounter = 0; scounter < src_layer_size; scounter++)
                    {
                        int sidx = batchctr * src_layer_size + scounter;
                        real_t cweight = current->weights_matrix[dcounter * src_layer_size + scounter];
                        real_t xdata = current->neurons_output[sidx];
                        sum += cweight * xdata; 
                    }

                    int dst_idx = batchctr * dst_layer_size + dcounter; 
                    next_to_current->neurons_input[dst_idx] = sum + current->biases[dcounter];

                    if (next_to_current->layer_type == 1)
                        next_to_current->neurons_output[dst_idx] = sigmoid(next_to_current->neurons_input[dst_idx]); 
                    else if (next_to_current->layer_type == 2)
                    {
                        next_to_current->neurons_output[dst_idx] = htangent(next_to_current->neurons_input[dst_idx]); 
                    }
                    else if (next_to_current->layer_type == 3)
                        next_to_current->neurons_output[dst_idx] = reLUSoftPlus(next_to_current->neurons_input[dst_idx]); 
                }
            }
            else if (next_to_current->subsampling == true)
            {
                //Pattern Recognition and Machine Learning, By Christopher M. Bishop (P267)
                // ... Each subsampling unit might take inputs from a 2x2 unit region in the 
                // corresponding feature map and would compute the average (here max pooling) of 
                // those inputs, multiplied by an adaptive weight with the addition of an adaptive bias
                // parameter, and then transformed using a sigmoidal non-linear activation function. 

                // Subsampling goes here ... 
                int src_layer_size = imw * imh * src_fmaps;
                int dst_fmap_size = next_imh * next_imw;
                int dst_layer_size = next_imw * next_imh * dst_fmaps;
            
                int src_fmap_ctr = 0;
                for (src_fmap_ctr = 0; src_fmap_ctr < src_fmaps; src_fmap_ctr++)
                {
                    int dst_fmap_ctr = src_fmap_ctr;
                    int fmap_stidx = batchctr * src_layer_size + src_fmap_ctr * imh * imw;
                    int next_fmap_stidx = batchctr * dst_layer_size + dst_fmap_ctr * dst_fmap_size;
                    real_t cweight = current->weights_matrix[src_fmap_ctr];

                    int wctr, hctr;
                    for (hctr = 0; hctr < imh; hctr += 2)
                    {
                        for (wctr = 0; wctr < imw; wctr += 2)
                        {
                            int cidx = fmap_stidx + hctr * imw + wctr;
                            
                            real_t p01, p02, p03, p04;
                            p01 = current->neurons_output[cidx];
                            p02 = current->neurons_output[cidx + 1];
                            p03 = current->neurons_output[cidx + imw];
                            p04 = current->neurons_output[cidx + imw + 1];

                            int dhctr = hctr/2;
                            int dwctr = wctr/2;
                            int dst_pos = next_fmap_stidx + dhctr * next_imw + dwctr;       
    
                            real_t spooling_result = 0; int poolingIdx1 = -1; 

                            real_t pooled = 0;                          
                            if (next_to_current->pool_type == 1)
                            {
                                // average pooling
                                pooled = (p01 + p02 + p03 + p04)/4;
                                next_to_current->neurons_input[dst_pos] = current->biases[dst_fmap_ctr];
                                next_to_current->neurons_input[dst_pos] += cweight * pooled;
                            }
                            else if (next_to_current->pool_type == 2)
                            {
                                // max pooling
                                int idx = 0;
                                pooled = pool_max(p01, p02, p03, p04, &idx, 4);
                                if (idx == 0) next_to_current->gradientMap[dst_pos] = cidx;
                                else if (idx == 1) next_to_current->gradientMap[dst_pos] = cidx + 1;
                                else if (idx == 2) next_to_current->gradientMap[dst_pos] = cidx + imw;
                                else if (idx == 3) next_to_current->gradientMap[dst_pos] = cidx + imw + 1;

                                next_to_current->neurons_input[dst_pos] = pooled; 
                            }
                            else if (next_to_current->pool_type == 3)
                            {
                                spooling_result = stochastic_pooling(p01, p02, p03, p04, &poolingIdx1);
                                pooled = spooling_result;

                                if (poolingIdx1 == 0)
                                    next_to_current->gradientMap[dst_pos] = cidx;
                                else if (poolingIdx1 == 1)
                                    next_to_current->gradientMap[dst_pos] = cidx + 1;
                                else if (poolingIdx1 == 2)
                                    next_to_current->gradientMap[dst_pos] = cidx + imw;
                                else if (poolingIdx1 == 3)
                                    next_to_current->gradientMap[dst_pos] = cidx + imw + 1;

                                next_to_current->neurons_input[dst_pos] = pooled;
                            }

                            if (next_to_current->layer_type == 1)
                            {
                                next_to_current->neurons_output[dst_pos] = sigmoid(next_to_current->neurons_input[dst_pos]);
                            }
                            else if (next_to_current->layer_type == 2)
                            {
                                next_to_current->neurons_output[dst_pos] = htangent(next_to_current->neurons_input[dst_pos]);
                            }
                            else if (next_to_current->layer_type == 3)
                            {
                                next_to_current->neurons_output[dst_pos] = reLUSoftPlus(next_to_current->neurons_input[dst_pos]);
                            }
                        }
                    }
                }
            }
    
            if (flag == true)
            {
                current = next_to_current;
                next_to_current = current->next;
            }            
        } 
    } 

    return 0;   
}

real_t sigmoid(real_t x)
{
	real_t y = 0.0;
	y = 1.0/(1 + exp(-x));
	return y;
}

real_t htangent(real_t x)
{
	return 1.7159 * tanh(0.66666 * x);
}

real_t reLUSoftPlus(real_t x)
{
	if (x < 0)
		return 0;
	else 
		return x;
}

real_t dreLUSoftPlus(real_t x)
{
	if (x < 0)
		return 0;
	else
		return 1;
}

real_t dhtangent(real_t y)
{
	real_t A = 1.7159;
	real_t S = 2.0/3.0;
	real_t d = A * S * (1 - y/A) * (1 + y/A); 
	return d;
}

real_t computeEntropy(real_t a11, real_t a12, real_t a13, real_t a14)
{
	real_t sum = a11 + a12 + a13 + a14;
	real_t e11 = 0, e12 = 0, e13 = 0, e14 = 0;

	if (a11 != 0)
		e11 = a11/sum * (real_t) log2((real_t)(a11/sum)); 

	if (a12 != 0)
		e12 = a12/sum * log2((real_t)(a12/sum));

	if (a13 != 0)
		e13 = a13/sum * log2((real_t)(a13/sum)); 

	if (a14 != 0)
		e14 = a14/sum * log2((real_t)(a14/sum));

	real_t entropy = e11 + e12 + e13 + e14;

	return entropy;
}

real_t stochastic_pooling(real_t a01, real_t a02, real_t a03, real_t a04, int* poolingIdx)
{
	real_t sum  = exp(a01) + exp(a02) + exp(a03) + exp(a04);
	real_t p01 = exp(a01)/sum;
	real_t p02 = exp(a02)/sum;
	real_t p03 = exp(a03)/sum;
	real_t p04 = exp(a04)/sum;

	//cumulative distribution function (CDF)
	real_t cdf[4] = {0, 0, 0, 0};
	cdf[0] = p01;
	cdf[1] = cdf[0] + p02;
	cdf[2] = cdf[1] + p03;
	cdf[3] = cdf[2] + p04;

	real_t randSample = (real_t) rand() / (real_t) RAND_MAX;
	
	if (randSample <= cdf[0])
	{
		*poolingIdx = 0;
		return a01;
	}
	else if (randSample <= cdf[1])
	{
		*poolingIdx = 1;
		return a02;
	}
	else if (randSample <= cdf[2])
	{
		*poolingIdx = 2;
		return a03;
	}
	else 
	{
		*poolingIdx = 3;
		return a04;
	}
}

real_t compute_mse(struct nnlayer* headlayer, int nouts, int* batch_indexes, unsigned char* lables) 
{
	struct nnlayer* current = headlayer;
	struct nnlayer* lastlayer = NULL;

	while (current != NULL)
	{
		if (current->next == NULL)
			lastlayer = current;

		current = current->next;
	}

	current = lastlayer;	

	int* desired_output = (int *) malloc(nouts * sizeof(int));

	real_t mse = 0, avg_mse = 0;
	int counter = 0;	
	for (counter = 0; counter < BATCH_SIZE; counter++)
	{
		if (current->layer_type == 1 || current->layer_type == 3)
		{
			int doCtr = 0;
			for (doCtr = 0; doCtr < nouts; doCtr++)
				desired_output[doCtr] = 0;
		}
		else if (current->layer_type == 2)
		{
			int doCtr = 0;
			for (doCtr = 0; doCtr < nouts; doCtr++)
				desired_output[doCtr] = -1;
		}

		unsigned char cl = lables[batch_indexes[counter]]; 
		desired_output[cl] = 1;
		mse = 0.0;

		int nctr = 0;
	 	for (nctr = 0; nctr < nouts; nctr++)
		{
			real_t error =  desired_output[nctr] - current->neurons_output[counter * nouts + nctr];
			mse = mse + (error * error); 
		}

		mse = mse/nouts;
		avg_mse = avg_mse + mse;
	}


	free(desired_output);	
	return avg_mse/BATCH_SIZE; 
}


void train_cnn(cnnlayer_t* headlayer, dataset_t* train_samples, dataset_t* test_samples)
{
	int epoch_counter = 0;
	int max_epoch = 150;
	int *batch_indexes = (int *) malloc(sizeof(int) * train_samples->numVectors);
	real_t min_mcr = 25.0;

    bool_t gpu_turn = 1;
    copy_hweights_to_dweights(headlayer);

	while (epoch_counter < max_epoch)
	{
        int nMcr = 1;
		int nvecs = train_samples->numVectors;
		int batch_count = nvecs/BATCH_SIZE;

		mini_batching(batch_indexes, nvecs, false); 

		real_t avg_mse = 0;
        int nouts = train_samples->lenlable;			

        if (gpu_turn != 0)
        {
            GpuTimer timer;
            long double elapsed = 0.0;
            int bctr = 0;
            for (bctr = 0; bctr < batch_count; bctr++)
            {
                timer.Start();
                d_feed_forward(headlayer, train_samples->data, &batch_indexes[bctr * BATCH_SIZE]);
                d_compute_gradients_deltas(headlayer, nouts, train_samples->lables, &batch_indexes[bctr * BATCH_SIZE]);
                d_update_weights(headlayer);
                d_reset_vectors(headlayer);
                timer.Stop();
                float time_citer = timer.Elapsed();
                elapsed += time_citer;
            
                if (bctr % 1000 == 0)
                    fprintf(stderr,"\nbctr/batch_count: %d/%d  epoch_counter/max_epoch: %d/%d", bctr, batch_count, epoch_counter, max_epoch);
            }

            fprintf(stderr, "\n elapsed_time: %Lf", elapsed);	
        }
        else 
        {
            fprintf(stderr, "\n Feed Forward via CPU");
            long double elapsed = 0.0;

            int bctr = 0;
            for (bctr = 0; bctr < batch_count; bctr++)
            {
                long start_ticks =  getticks();
                h_feed_forward(headlayer, train_samples->data, &batch_indexes[bctr * BATCH_SIZE]);
                h_compute_gradients_deltas(headlayer, nouts, train_samples->lables, &batch_indexes[bctr * BATCH_SIZE]);		
                h_update_weights(headlayer);
                reset_inputs_dweights_deltas(headlayer);
                long end_ticks = getticks();
                float time_citer = (float)(end_ticks - start_ticks)/3330000;
                elapsed += time_citer;

                if (bctr % 100 == 0)
                    fprintf(stderr,"\nbctr/batch_count: %d/%d/%d", bctr, batch_count, max_epoch);
            }

            fprintf(stderr, "\n elapsed_time: %Lf", elapsed);	
        }

        avg_mse = avg_mse/batch_count;
        printf("\n Avg MSE: %f, epoch: %d", avg_mse, epoch_counter);
        
        if (gpu_turn != 0 && epoch_counter % nMcr == 0)
        {
            copy_dweights_to_hweights(headlayer);
            //display_weights_matrices(headlayer);

            real_t mcr_test_set = 0;
            mcr_test_set = d_compute_missclassification_rate(headlayer, test_samples);
            printf("\n =========================");
            printf("\n EpochCounter     TEST SET");
            printf("\n\n   %d              %f   ", epoch_counter, mcr_test_set);
            fprintf(stderr,"\n\n   %d              %f   ", epoch_counter, mcr_test_set);
            printf("\n");

            d_reset_output_vectors(headlayer);

            if (mcr_test_set < min_mcr)
            {
                char fn[4];
                char fname[13] = "WEIGHTS/";
                sprintf (fn, "%d", epoch_counter);
                strcat(fname, fn);
                save_trained_network_weights(headlayer, fname);
                min_mcr = mcr_test_set;
            }
        }
        else if (gpu_turn == 0 && epoch_counter % nMcr == 0)
        {
            //display_weights_matrices(headlayer);
            real_t mcr_test_set = 0;
            mcr_test_set = h_compute_missclassification_rate(headlayer, test_samples);
            printf("\n =========================");
            printf("\n EpochCounter     TEST SET");
            printf("\n\n   %d              %f   ", epoch_counter, mcr_test_set);
            fprintf(stderr,"\n\n   %d              %f   ", epoch_counter, mcr_test_set);
            printf("\n");

            reset_inputs_dweights_deltas(headlayer);

            if (mcr_test_set < min_mcr)
            {
                char fn[4];
                char fname[13] = "WEIGHTS/";
                sprintf (fn, "%d", epoch_counter);
                strcat(fname, fn);
                save_trained_network_weights(headlayer, fname);
                min_mcr = mcr_test_set;
            }
        }

        if (epoch_counter % 2 == 0)
            LEARNING_RATE = LEARNING_RATE * 0.93;

        epoch_counter++;
    }

	fprintf(stderr,"\n"); 
	free(batch_indexes);
}

void d_compute_gradients_deltas(cnnlayer_t *headlayer, int nouts,  unsigned char* desired_output, int* batch_indexes)
{
	int *desired_vec = (int *) malloc(sizeof(int) * nouts);
	int batchctr = 0;

	for (batchctr = 0; batchctr < BATCH_SIZE; batchctr++)
	{
		cnnlayer_t* current = headlayer->next; 
		cnnlayer_t* lastlayer = NULL;

        /* Reaching the last layer and propagating error gradients backwards */
		while (current != NULL)
		{
			if (current->next == NULL)
			{
				lastlayer = current;
				break;
			}
            	
			current = current->next;	
		}
			
		current = lastlayer;
				
		int num_neurons = current->no_of_neurons, doCtr = 0; 
		if (current->layer_type == 1 || current->layer_type == 3)
		{
			for (doCtr = 0; doCtr < nouts; doCtr++)
				desired_vec[doCtr] = 0;
		}
		else if (current->layer_type == 2)
		{
			for (doCtr = 0; doCtr < nouts; doCtr++)
				desired_vec[doCtr] = -1;
		}

		int d_idx = desired_output[batch_indexes[batchctr]];
		desired_vec[d_idx] = 1;
		
        int outSize = current->no_of_neurons * BATCH_SIZE * sizeof(real_t);
        HANDLE_ERROR(cudaMemcpy(current->neurons_output, current->d_neurons_output, outSize, cudaMemcpyDeviceToHost));

		int ectr = 0;
		for (ectr = 0; ectr < num_neurons; ectr++)
		{
			int b_idx = batchctr * num_neurons + ectr;
			current->error_deltas[b_idx] = (current->neurons_output[b_idx] - desired_vec[ectr]);	
			//printf("\n Output: %f", current->neurons_output[b_idx]);	
		}

        // Copy error_deltas to GPU global memory	
        HANDLE_ERROR(cudaMemcpy(current->d_error_deltas, current->error_deltas, outSize, cudaMemcpyHostToDevice));

		current = lastlayer->previous;
		
		bool_t flag = true;		
		while (current != NULL && flag == true)
		{
			if (current->previous == NULL)
				flag = false;
					
			//back propagate the error deltas from here
			int curr_height = current->fmap_height;
		  	int curr_width = current->fmap_width;
		  	int curr_fmap_size = curr_height * curr_width;

			int prev_height = lastlayer->fmap_height;
			int prev_width = lastlayer->fmap_width;
			int prev_fmap_size = prev_height * prev_width;

			if (current->fkernel == 1 && lastlayer->subsampling == false) 
			{
                real_t* d_output = current->d_neurons_output;
                real_t* d_lerr_deltas = lastlayer->d_error_deltas;
                real_t* d_cerr_deltas = current->d_error_deltas;
                real_t* d_weights = current->d_weights;
                real_t* d_delta_weights = current->d_delta_weights;
                real_t* d_delta_biases = current->d_delta_biases;
 
                int nBlocks = lastlayer->no_of_neurons;
                int nThreads = current->no_of_neurons;
                int sh_mem_size = (2 * current->no_of_neurons + 1) * sizeof(real_t);

                d_rear_DNN_errorbp<<<nBlocks, nThreads, sh_mem_size>>>(d_output, d_lerr_deltas, d_cerr_deltas, d_weights, d_delta_weights, d_delta_biases);
                sh_mem_size = (current->no_of_neurons) * sizeof(real_t);
                d_rear_DNN_update_error_deltas<<<1, nThreads, sh_mem_size>>>(d_output, d_cerr_deltas, current->layer_type);
                cudaDeviceSynchronize();

                /* For Debugging purpose only */
                //int wSize = current->no_of_neurons * lastlayer->no_of_neurons * sizeof(real_t);
                //HANDLE_ERROR(cudaMemcpy(current->delta_weights, d_delta_weights, wSize, cudaMemcpyDeviceToHost));
                //int nerrSize = current->no_of_neurons * sizeof(real_t);
                //HANDLE_ERROR(cudaMemcpy(current->error_deltas, d_cerr_deltas, nerrSize, cudaMemcpyDeviceToHost));
			}
			else if (current->fkernel != 1 && lastlayer->subsampling == false)
			{
                real_t* d_output = current->d_neurons_output;
                real_t* d_lerr_deltas = lastlayer->d_error_deltas;
                real_t* d_cerr_deltas = current->d_error_deltas;
                real_t* d_weights = current->d_weights;
                real_t* d_delta_weights = current->d_delta_weights;
                real_t* d_delta_biases = current->d_delta_biases;

                int kerSize = current->fkernel * current->fkernel;
                dim3 nBlocks(current->no_of_fmaps, lastlayer->no_of_fmaps, 1);
                dim3 nThreads(current->fmap_width, current->fmap_height, 1);
				
                int sh_mem_size = (prev_fmap_size + curr_fmap_size + kerSize + 1) * sizeof(real_t);
                errorbp_convolution_layers2<<<nBlocks, nThreads, sh_mem_size >>>(d_output, d_lerr_deltas, d_cerr_deltas, d_weights, d_delta_weights, kerSize);
               
                nBlocks.x = lastlayer->no_of_fmaps; nBlocks.y = 1; nBlocks.z = 1; 
                nThreads.x = lastlayer->fmap_width; nThreads.y = lastlayer->fmap_height; nThreads.z = 1;
                errorbp_convolution_update_biases<<<nBlocks, nThreads>>>(d_lerr_deltas, d_delta_biases);
                 
                nBlocks.x = current->no_of_fmaps;
                nBlocks.y = nBlocks.z = 1;
                nThreads.x = current->fmap_width * current->fmap_height; 
                nThreads.y = nThreads.z = 1;
                sh_mem_size = current->fmap_width * current->fmap_height * sizeof(real_t); 
                d_update_error_deltas<<<nBlocks, nThreads, sh_mem_size>>>(d_output, d_cerr_deltas, current->layer_type); 
                
                //int wSize = (current->no_of_fmaps * lastlayer->no_of_fmaps * kerSize) * sizeof(real_t);
                //HANDLE_ERROR(cudaMemcpy(current->delta_weights, d_delta_weights, wSize, cudaMemcpyDeviceToHost));
                //int nerrSize = current->no_of_neurons * sizeof(real_t);
                //HANDLE_ERROR(cudaMemcpy(current->error_deltas, d_cerr_deltas, nerrSize, cudaMemcpyDeviceToHost));
			}
			else if (lastlayer->subsampling == true)
			{
                real_t* d_output = current->d_neurons_output;
                real_t* d_lerr_deltas = lastlayer->d_error_deltas;
                real_t* d_cerr_deltas = current->d_error_deltas;
                int* d_gradientMap = lastlayer->d_gradientMap;
 
                dim3 nBlocks(current->no_of_fmaps, 1, 1);
                dim3 nThreads(prev_width, prev_height, 1);
                
                int layer_type = current->layer_type;
                int sh_mem_size = (2 * prev_width * prev_height) * sizeof(real_t);
                d_errbp_subsampling<<<nBlocks, nThreads, sh_mem_size>>>(d_output, d_lerr_deltas, d_cerr_deltas, d_gradientMap, layer_type);
                cudaDeviceSynchronize();

                //int nerrSize = current->no_of_neurons * sizeof(real_t);
                //HANDLE_ERROR(cudaMemcpy(current->error_deltas, d_cerr_deltas, nerrSize, cudaMemcpyDeviceToHost));
			}
					
			if (flag == true)
			{	
				lastlayer = current;	
				current = current->previous;
			}
		}
	}
		
	free(desired_vec);
}

void h_compute_gradients_deltas(cnnlayer_t *headlayer, int nouts,  unsigned char* desired_output, int* batch_indexes)
{
	int *desired_vec = (int *) malloc(sizeof(int) * nouts);
	int batchctr = 0;

	for (batchctr = 0; batchctr < BATCH_SIZE; batchctr++)
	{
		cnnlayer_t* current = headlayer->next; //skipped input layer
		cnnlayer_t* lastlayer = NULL;

		while (current != NULL)
		{
			int num_neurons = current->no_of_neurons;
			int nctr = 0;
			for (nctr = 0; nctr < num_neurons; nctr++)
			{
				int idx = batchctr * num_neurons + nctr;

				if (current->layer_type == 1)
					current->dy_output[idx] = current->neurons_output[idx] * (1 - current->neurons_output[idx]);
				else if (current->layer_type == 2)
					current->dy_output[idx] = dhtangent(current->neurons_output[idx]);
				else if (current->layer_type == 3)
					current->dy_output[idx] = dreLUSoftPlus(current->neurons_output[idx]);
			}

			if (current->next == NULL)
			{
				lastlayer = current;
				break;
			}	
			current = current->next;	
		}
			
		current = lastlayer;
				
		// compute error for last layer
		int num_neurons = current->no_of_neurons; 
		if (current->layer_type == 1 || current->layer_type == 3)
		{
			int doCtr = 0;
			for (doCtr = 0; doCtr < nouts; doCtr++)
				desired_vec[doCtr] = 0;
		}
		else if (current->layer_type == 2)
		{
			int doCtr = 0;
			for (doCtr = 0; doCtr < nouts; doCtr++)
				desired_vec[doCtr] = -1;
		}

		int d_idx = desired_output[batch_indexes[batchctr]];
		desired_vec[d_idx] = 1;
		
		int ectr = 0;
		for (ectr = 0; ectr < num_neurons; ectr++)
		{
			int b_idx = batchctr * num_neurons + ectr;
			current->error_deltas[b_idx] = (current->neurons_output[b_idx] - desired_vec[ectr]);	
			//printf("\n Output: %f", current->neurons_output[b_idx]);	
		}
		
		current = lastlayer->previous;
		
		bool_t flag = true;		
		while (current != NULL && flag == true)
		{
			if (current->previous == NULL)
				flag = false;
					
			//back propagate the error deltas from here
			int curr_no_fmaps = current->no_of_fmaps;
			int curr_height = current->fmap_height;
		  	int curr_width = current->fmap_width;
		  	int curr_fmap_size = curr_height * curr_width;

			int prev_no_fmaps = lastlayer->no_of_fmaps;			
			int prev_height = lastlayer->fmap_height;
			int prev_width = lastlayer->fmap_width;
			int prev_fmap_size = prev_height * prev_width;

			int no_neurons_prev = prev_fmap_size * prev_no_fmaps;
			int no_neurons_curr = curr_fmap_size * curr_no_fmaps;
			
			if (current->fkernel == 1 && lastlayer->subsampling == false) 
			{
				int playerUnitIdx = 0, clayerUnitIdx = 0;
				for (clayerUnitIdx = 0; clayerUnitIdx < no_neurons_curr; clayerUnitIdx++)
				{
					real_t sum = 0.0;
					int cUnitIdx = batchctr * no_neurons_curr + clayerUnitIdx;
					for (playerUnitIdx = 0; playerUnitIdx < no_neurons_prev; playerUnitIdx++)
					{
						//for each of the neuron,we have dy_output
						int pUnitIdx = batchctr * no_neurons_prev + playerUnitIdx; 
						int wIdx = playerUnitIdx * no_neurons_curr + clayerUnitIdx;	

                        current->delta_weights[wIdx] += lastlayer->error_deltas[pUnitIdx] * current->neurons_output[cUnitIdx];
                        sum += lastlayer->error_deltas[pUnitIdx] * current->weights_matrix[wIdx]; 
					}

					current->error_deltas[cUnitIdx] = current->dy_output[cUnitIdx] * sum;
                    //printf("\n dy:%f, sum: %f",  current->dy_output[cUnitIdx], sum);
				}					

				for (playerUnitIdx = 0; playerUnitIdx < no_neurons_prev; playerUnitIdx++)
				{
					current->delta_biases[playerUnitIdx] += lastlayer->error_deltas[playerUnitIdx] * 1.0;
				}	
			}
			else if (current->fkernel != 1 && lastlayer->subsampling == false)
			{
				//convolutional layer with kernel of 5x5
				int indexes[25];
				int pfmapctr, cfmapctr;

				for (cfmapctr = 0; cfmapctr < curr_no_fmaps; cfmapctr++)
				{
				  	int curr_fmap_stidx = batchctr * no_neurons_curr + cfmapctr * curr_fmap_size;
					int iwstidx = cfmapctr * current->fkernel * current->fkernel;

  					for (pfmapctr = 0; pfmapctr < prev_no_fmaps; pfmapctr++)
  					{
						int prev_fmap_stidx = batchctr * no_neurons_prev + pfmapctr * prev_fmap_size;						
						int fwstidx = iwstidx + pfmapctr * (curr_no_fmaps * current->fkernel * current->fkernel);

						int i;
						for (i = 0; i < prev_fmap_size; i++)
						{
							int tx, ty;
							tx = i % lastlayer->fmap_width;
							ty = i / lastlayer->fmap_width;
							int bmargin = floor(current->fkernel/2);
							int stx, sty;
							stx = tx + bmargin;
							sty = ty + bmargin;
							
							//in the source fmap
							int center = sty * current->fmap_width + stx;

							int filterCtr = 0, convCtr1 = 0, convCtr2 = 0;
                            for (convCtr1 = -1 * floor(current->fkernel/2); convCtr1 <= floor(current->fkernel/2); convCtr1++) 
                            {
                                for (convCtr2 = -1 * floor(current->fkernel/2); convCtr2 <= floor(current->fkernel/2); convCtr2++)
                                {
                                    indexes[filterCtr] = center + convCtr1 * current->fmap_width + convCtr2;
                                    filterCtr++;
                                }
                            }

							int player_idx = prev_fmap_stidx + i; 
							int iter = 0;
							for (iter = 0; iter < current->fkernel * current->fkernel; iter++)
							{
								int clayer_idx = curr_fmap_stidx + indexes[iter]; 
								int weights_idx = fwstidx + iter; 

                                current->delta_weights[weights_idx] += lastlayer->error_deltas[player_idx] * current->neurons_output[clayer_idx]; 
                                current->error_deltas[clayer_idx] +=  (current->weights_matrix[weights_idx] * lastlayer->error_deltas[player_idx] * current->dy_output[clayer_idx]); 
							}

							if (cfmapctr == 0)
								current->delta_biases[pfmapctr] += lastlayer->error_deltas[player_idx] * 1.0;
						}
					}
				}
			}
			else if (lastlayer->subsampling == true)
			{
				int sindexes[4];
				int pfmapCtr = 0;
				for (pfmapCtr = 0; pfmapCtr < prev_no_fmaps; pfmapCtr++)
				{
					int pstidx = batchctr * no_neurons_prev + pfmapCtr * prev_fmap_size; //0, 25
					int cfmapCtr = pfmapCtr; 
					int cstidx = batchctr * no_neurons_curr + cfmapCtr * curr_fmap_size;

					int pfmUnitctr = 0;
					for (pfmUnitctr = 0; pfmUnitctr < prev_fmap_size; pfmUnitctr++)
					{	
						int player_idx = pstidx + pfmUnitctr;
						int px = pfmUnitctr % lastlayer->fmap_width;
						int py = pfmUnitctr / lastlayer->fmap_height;
						int sx = px * 2;
						int sy = py * 2;

						int clUnitIdx = sy * current->fmap_width + sx;
						sindexes[0] = cstidx + clUnitIdx;
						sindexes[1] = cstidx + clUnitIdx + 1;
						sindexes[2] = cstidx + clUnitIdx + curr_width;
						sindexes[3] = cstidx + clUnitIdx + curr_width + 1;

						if (current->pool_type == 1) 
						{
							int j = 0;
							for (j = 0; j < 4; j++)
							{
								current->delta_weights[cfmapCtr] += lastlayer->error_deltas[player_idx] * current->neurons_output[sindexes[j]];
								current->error_deltas[sindexes[j]] =  (current->weights_matrix[cfmapCtr] * lastlayer->error_deltas[player_idx]) * current->dy_output[sindexes[j]];
							}

							current->delta_biases[cfmapCtr] += lastlayer->error_deltas[player_idx] * 1.0;
						}
						else if (current->pool_type == 2)
						{
							int gradientIdx = lastlayer->gradientMap[player_idx];
							//curent->delta_weights[cfmpCt]+=lastlayer->error_deltas[player_idx]*current->neurons_output[gradientIdx];
							current->error_deltas[gradientIdx] =  lastlayer->error_deltas[player_idx] * current->dy_output[gradientIdx];
						}
						else if (current->pool_type == 3)
						{
							int gradientIdx = lastlayer->gradientMap[player_idx];
							current->error_deltas[gradientIdx] =  lastlayer->error_deltas[player_idx] * current->dy_output[gradientIdx];
							//current->delta_biases[cfmapCtr] += lastlayer->error_deltas[player_idx] * 1.0;
						}
					}
				}
			}
					
			if (flag == true)
			{	
				lastlayer = current;	
				current = current->previous;
			}
		}
	}
		
	free(desired_vec);
}

//accumulate weight deltas
void average_deltas(struct nnlayer* headlayer)
{
	if (BATCH_SIZE > 1)
	{
		struct nnlayer* current = headlayer; 	
		struct nnlayer* next_to_current = current->next;
		bool_t flag = true;
		
		while (current != NULL && flag == true)
		{
			if (next_to_current->next == NULL)
				flag = false;
			
			if (next_to_current->subsampling == false && current->fkernel != 1)
			{
				int no_dweights = current->no_of_fmaps * next_to_current->no_of_fmaps * current->fkernel * current->fkernel;
				int cctr = 0;
				for (cctr = 0; cctr < no_dweights; cctr++)
				{
					current->delta_weights[cctr] = current->delta_weights[cctr]/BATCH_SIZE;
				}

				//biases
				for (cctr = 0; cctr < next_to_current->no_of_fmaps; cctr++)
				{
					current->delta_biases[cctr] = current->delta_biases[cctr]/BATCH_SIZE;
				}
			}
			else if (next_to_current->subsampling == false && current->fkernel == 1)
			{
				int curr_count = current->no_of_neurons;
				int next_curr_count = next_to_current->no_of_neurons;
				int cctr = 0;
				int ncctr = 0;
				for (cctr = 0; cctr < curr_count; cctr++)
				{
					for (ncctr = 0; ncctr < next_curr_count; ncctr++)
					{
						int idx = cctr * next_curr_count + ncctr;
						current->delta_weights[idx] = current->delta_weights[idx]/BATCH_SIZE;
					}
				}
	
				//biases
				for (cctr = 0; cctr < next_to_current->no_of_fmaps; cctr++)
				{
					current->delta_biases[cctr] = current->delta_biases[cctr]/BATCH_SIZE;
				}		
			}
			else if (next_to_current->subsampling == true)
			{
				// Subsampling layer 
				int count = current->no_of_fmaps;
				int counter = 0;
				for (counter = 0; counter < count; counter++)
				{
					current->delta_weights[counter] = current->delta_weights[counter]/BATCH_SIZE;
				}	

				//biases
				int cctr = 0;
				for (cctr = 0; cctr < next_to_current->no_of_fmaps; cctr++)
				{
					current->delta_biases[cctr] = current->delta_biases[cctr]/BATCH_SIZE;
				}		
			}
			
			if (flag == true)
			{
				current = next_to_current;
				next_to_current = current->next;
			}
		}
	}	
}

void h_update_weights(struct nnlayer* headlayer)
{
	struct nnlayer* current = headlayer; 	
	struct nnlayer* next_to_current = current->next;

	bool_t flag = true;
	while (current != NULL && flag == true)
	{
		if (next_to_current->next == NULL)
			flag = false;	

		int ndweights = 0;
		if (next_to_current->subsampling == false)
			ndweights = current->no_of_fmaps * next_to_current->no_of_fmaps * current->fkernel * current->fkernel;	
		else
			ndweights = current->no_of_fmaps;	

		int counter = 0;
		for (counter = 0; counter < ndweights; counter++)
		{
			current->weights_matrix[counter] -= LEARNING_RATE * current->delta_weights[counter];
		}

        for (counter = 0; counter < next_to_current->no_of_fmaps; counter++)
        {
            current->biases[counter] -= LEARNING_RATE * current->delta_biases[counter];
        }

		if (flag == true)
		{
			current = next_to_current;
			next_to_current = current->next;
		}
	}
}

void d_update_weights(struct nnlayer* headlayer)
{
	struct nnlayer* current = headlayer; 	
	struct nnlayer* next_to_current = current->next;

	bool_t flag = true;
	while (current != NULL && flag == true)
	{
		if (next_to_current->next == NULL)
			flag = false;	

        real_t* d_weights = current->d_weights;
        real_t* d_delta_weights = current->d_delta_weights;
        real_t* d_biases = current->d_biases;
        real_t* d_delta_biases = current->d_delta_biases;

        if (next_to_current->subsampling == true) 
        {
            int nBlocks = current->no_of_fmaps;
            int nThreads = 1; 
                        
            int sh_mem_size = sizeof(real_t);
            d_update_weights_kernel<<<nBlocks, nThreads, sh_mem_size>>>(d_weights, d_delta_weights, LEARNING_RATE);
            int nwSize = current->no_of_fmaps * sizeof(real_t);
            HANDLE_ERROR(cudaMemcpy(current->weights_matrix, current->d_weights, nwSize, cudaMemcpyDeviceToHost));

            d_update_biases_kernel<<< next_to_current->no_of_fmaps, 1 >>>(d_biases, d_delta_biases, LEARNING_RATE);
            HANDLE_ERROR(cudaMemcpy(current->biases, current->d_biases, nwSize, cudaMemcpyDeviceToHost));
        }
        else
        {
            dim3 nBlocks(current->no_of_fmaps, next_to_current->no_of_fmaps, 1);
            dim3 nThreads(current->fkernel, current->fkernel, 1);

            int sh_mem_size = 2 * current->fkernel * current->fkernel * sizeof(real_t);
            d_update_weights_kernel<<<nBlocks, nThreads, sh_mem_size>>>(d_weights, d_delta_weights, LEARNING_RATE);

            int nwSize = current->no_of_fmaps * next_to_current->no_of_fmaps * current->fkernel * current->fkernel * sizeof(real_t);    
            HANDLE_ERROR(cudaMemcpy(current->weights_matrix, current->d_weights, nwSize, cudaMemcpyDeviceToHost));

            d_update_biases_kernel<<< next_to_current->no_of_fmaps, 1 >>>(d_biases, d_delta_biases, LEARNING_RATE);
            int nbSize = next_to_current->no_of_fmaps * sizeof(real_t);    
            HANDLE_ERROR(cudaMemcpy(current->biases, current->d_biases, nbSize, cudaMemcpyDeviceToHost));
        }


		if (flag == true)
		{
			current = next_to_current;
			next_to_current = current->next;
		}
	}
}

void hd_reset_biases(cnnlayer_t* headlayer)
{
	cnnlayer_t* current = headlayer; 	
	cnnlayer_t* next_to_current = current->next;

	bool_t flag = true;
	while (current != NULL && flag == true)
	{
		if (next_to_current->next == NULL)
			flag = false;	

        int nbSize = next_to_current->no_of_fmaps * sizeof(real_t);    
        HANDLE_ERROR(cudaMemset(current->d_biases, 0, nbSize));
        memset(current->biases, 0, nbSize);

		if (flag == true)
		{
			current = next_to_current;
			next_to_current = current->next;
		}
	}
}


void d_reset_vectors(cnnlayer_t* headlayer)
{
	cnnlayer_t* current = headlayer; 	
	cnnlayer_t* next_to_current = current->next;

	bool_t flag = true;
	while (current != NULL && flag == true)
	{
		if (next_to_current->next == NULL)
			flag = false;	

        int fk = current->fkernel;
        if (next_to_current->subsampling == true) 
        {
            int nwSize = current->no_of_fmaps * sizeof(real_t);
            HANDLE_ERROR(cudaMemset(current->d_delta_weights, 0, nwSize));

            int nbSize = next_to_current->no_of_fmaps * sizeof(real_t);    
            HANDLE_ERROR(cudaMemset(current->d_delta_biases, 0, nbSize));
        }
        else
        {
            int nwSize = current->no_of_fmaps * next_to_current->no_of_fmaps * fk * fk * sizeof(real_t);    
            HANDLE_ERROR(cudaMemset(current->d_delta_weights, 0, nwSize));

            int nbSize = next_to_current->no_of_fmaps * sizeof(real_t);    
            HANDLE_ERROR(cudaMemset(current->d_delta_biases, 0, nbSize));
        }

        int noSize = current->no_of_fmaps * current->fmap_width * current->fmap_height * sizeof(real_t);    
        HANDLE_ERROR(cudaMemset(current->d_neurons_output, 0, noSize));
        HANDLE_ERROR(cudaMemset(current->d_error_deltas, 0, noSize));


		if (flag == true)
		{
			current = next_to_current;
			next_to_current = current->next;
		}
        else
        {
            //this is the very last layer
            int noSize = next_to_current->no_of_neurons * sizeof(real_t);    
            HANDLE_ERROR(cudaMemset(next_to_current->d_neurons_output, 0, noSize));
            HANDLE_ERROR(cudaMemset(next_to_current->d_error_deltas, 0, noSize));
        }
	}
}

void d_reset_output_vectors(cnnlayer_t* headlayer)
{
	cnnlayer_t* current = headlayer; 	
	cnnlayer_t* next_to_current = current->next;

	bool_t flag = true;
	while (current != NULL && flag == true)
	{
		if (next_to_current->next == NULL)
			flag = false;	

        int noSize = current->no_of_neurons * sizeof(real_t);    
        HANDLE_ERROR(cudaMemset(current->d_neurons_output, 0, noSize));

		if (flag == true)
		{
			current = next_to_current;
			next_to_current = current->next;
		}
        else
        {
            int noSize = next_to_current->no_of_neurons * sizeof(real_t);    
            HANDLE_ERROR(cudaMemset(next_to_current->d_neurons_output, 0, noSize));
        }
	}
}

int sign(real_t n1, real_t n2)
{
	if (n1 * n2 > 0)
		return 1;
	else if (n1 * n2 < 0)
		return -1;
	else 
		return 0; 
}

void reset_inputs_dweights_deltas(cnnlayer_t* headlayer)
{
	//printf("\nreset_dweight_deltas");
	cnnlayer_t* current = headlayer;
	cnnlayer_t* next_to_current = current->next;

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
 			weight_matrix_size = src_nfmaps; 
			biases_size = dst_nfmaps;
		}
		else
		{
 			weight_matrix_size = src_nfmaps * dst_nfmaps * fkernel * fkernel;
			biases_size = dst_nfmaps;
		}
	
		int counter = 0;
		for (counter = 0; counter < weight_matrix_size; counter++)
			current->delta_weights[counter] = 0.0;

		for (counter = 0; counter < biases_size; counter++)
			current->delta_biases[counter] = 0.0;

		//reset error deltas and neurons_input fields
		int reset_size = current->no_of_neurons * BATCH_SIZE;
		for (counter = 0; counter < reset_size; counter++)
		{
			current->error_deltas[counter] = 0.0;
			current->neurons_input[counter] = 0.0;
		}
	
		if (flag == true)
		{
			current = next_to_current;
			next_to_current = current->next;
		}
        // the last layer next_to_current does not collect +=,
        // neither neurons_input[] nor error_deltas, but GPU version needs 
        // it to be cleared
	}
}

real_t h_compute_missclassification_rate(cnnlayer_t *headlayer, dataset_t *samples)
{
    fprintf(stderr, "\n computing MCR, No. of samples: %d\n, Progress: ", samples->numVectors);
	int mcr = 0;
	int datactr = 0;
	for (datactr = 0; datactr < samples->numVectors; datactr++)
	{
		if (datactr % 1000 == 0)
			fprintf(stderr, ".");

		cnnlayer_t* current = headlayer;
		cnnlayer_t* next_to_current = current->next;
		
		bool_t flag = true;
		while (current != NULL)
		{
			int resetctr = 0;
			for (resetctr = 0; resetctr < current->no_of_neurons; resetctr++)
			{
				current->neurons_input[resetctr] = 0.0;
				current->neurons_output[resetctr] = 0.0;
			}			
			
			current = current->next;
		}

		current = headlayer;
		next_to_current = current->next;
		int inp_vec_size = current->no_of_neurons;
		int desired_label = samples->lables[datactr];

		int input_data_ctr = 0;
		for (input_data_ctr = 0; input_data_ctr < inp_vec_size; input_data_ctr++)
		{
			int inputIdx = datactr * inp_vec_size + input_data_ctr;
			current->neurons_input[input_data_ctr] = samples->data[inputIdx]; 
			current->neurons_output[input_data_ctr] = samples->data[inputIdx];
		}	

		flag = true;	
		while (current != NULL && flag == true)
		{	
			if (next_to_current->next == NULL)
				flag = false;

			if (next_to_current->subsampling == false && current->fkernel != 1)
			{
				//convolution layers	
				int src_fmaps = current->no_of_fmaps;
				int dst_fmaps = next_to_current->no_of_fmaps;
				int fkernel = current->fkernel;
				int bmargin = floor(fkernel/2);
				int imh = current->fmap_height;
				int imw = current->fmap_width;
	
				//for the first layer, output = input
		
				int sctr = 0, dctr = 0;
				for (dctr = 0; dctr < dst_fmaps; dctr++)
				{
					for (sctr = 0; sctr < src_fmaps; sctr++)
					{
						int weights_stidx = dctr * fkernel * fkernel * src_fmaps; 
						int st_idx = weights_stidx + sctr * fkernel * fkernel;
 
						real_t* filter = NULL;

                        filter = &(current->weights_matrix[st_idx]); 

						int fmap_stidx = sctr * imh * imw;

						//destination feature map 
						int next_imw = next_to_current->fmap_width;
						int next_imh = next_to_current->fmap_height;
						int dst_fmap_stidx = dctr * next_imh * next_imw;
						int dst_fmap_ctr = 0;

						int hctr = 0;
						int wctr = 0;

						for (hctr = 0; hctr < imh; hctr++)
						{
							for (wctr = 0; wctr < imw; wctr++)
							{
								if ((hctr >= bmargin && wctr >= bmargin) && (hctr < imh - bmargin && wctr < imw - bmargin))	
								{
									int cidx = fmap_stidx + hctr * imw + wctr;
									real_t sum = 0.0;
                                    int filterCtr = 0, convCtr1 = 0, convCtr2 = 0;
                                    for (convCtr1 = -1 * floor(current->fkernel/2); convCtr1 <= floor(current->fkernel/2); convCtr1++) 
                                    {
                                        for (convCtr2 = -1 * floor(current->fkernel/2); convCtr2 <= floor(current->fkernel/2); convCtr2++)
                                        {
                                            sum = sum + filter[filterCtr] * current->neurons_output[cidx + convCtr1 * imw + convCtr2];
                                            filterCtr++;
                                        }
                                    }
									 //save summation to destination feature map
									 int dst_idx = dst_fmap_stidx + dst_fmap_ctr; 
									 next_to_current->neurons_input[dst_idx] += sum; 
									
									//applying transfer function
									if (sctr == src_fmaps - 1)
									{
                                        next_to_current->neurons_input[dst_idx] += current->biases[dctr]; 
										real_t cn = next_to_current->neurons_input[dst_idx]; 
										next_to_current->neurons_input[dst_idx] = 0; 

										if (current->layer_type == 1)
											next_to_current->neurons_output[dst_idx] = sigmoid(cn); 
										else if (current->layer_type == 2)
										{
											next_to_current->neurons_output[dst_idx] = htangent(cn); 
										}
										else if (current->layer_type == 3)
											next_to_current->neurons_output[dst_idx] = reLUSoftPlus(cn);
									}

									dst_fmap_ctr++;
								}
							}
						}
					}
				}
			}
			else if (next_to_current->subsampling == false && current->fkernel == 1)
			{
		      	int src_layer_size = current->no_of_neurons;
				int dst_layer_size = next_to_current->no_of_neurons;

				int dcounter = 0;
				int scounter = 0;
				real_t sum = 0.0;

				for (dcounter = 0; dcounter < dst_layer_size; dcounter++)
				{
					sum = 0.0;
					for (scounter = 0; scounter < src_layer_size; scounter++)
					{
						real_t cweight = 0.0; 
                        cweight = current->weights_matrix[dcounter * src_layer_size + scounter];

						real_t xdata = 0;
                        xdata = current->neurons_output[scounter];

						sum += cweight * xdata; 
					}

					next_to_current->neurons_input[dcounter] = sum + current->biases[dcounter];

					if (next_to_current->layer_type == 1)
						next_to_current->neurons_output[dcounter] = sigmoid(next_to_current->neurons_input[dcounter]); 	
					else if (next_to_current->layer_type == 2) 
					{
						next_to_current->neurons_output[dcounter] = htangent(next_to_current->neurons_input[dcounter]);
					}
					else if (next_to_current->layer_type == 3)
						next_to_current->neurons_output[dcounter] = reLUSoftPlus(next_to_current->neurons_input[dcounter]); 
				}
			}
			else if (current->fkernel == 1 && next_to_current->subsampling == true)
			{
				// Subsampling goes here ... 
				int src_fmaps = current->no_of_fmaps;
				int imh = current->fmap_height;
				int imw = current->fmap_width;
				int next_imw = next_to_current->fmap_width;
				int next_imh = next_to_current->fmap_height;
				int dst_fmap_size = next_imh * next_imw;

				int src_fmap_ctr = 0;
				for (src_fmap_ctr = 0; src_fmap_ctr < src_fmaps; src_fmap_ctr++)
				{
					int dst_fmap_ctr = src_fmap_ctr;
					int fmap_stidx = src_fmap_ctr * imh * imw;
					int next_fmap_stidx = dst_fmap_ctr * dst_fmap_size;
					real_t cweight = current->weights_matrix[src_fmap_ctr];

					int wctr = 0, hctr = 0;
					for (hctr = 0; hctr < imh; hctr += 2)
					{
						for (wctr = 0; wctr < imw; wctr += 2)
						{
							int cidx = fmap_stidx + hctr * imw + wctr;

							int dhctr = hctr/2;
							int dwctr = wctr/2;
							int dst_pos = next_fmap_stidx + dhctr * next_imw + dwctr;			

							real_t p1, p2, p3, p4;
							p1 = current->neurons_output[cidx];
							p2 = current->neurons_output[cidx + 1];
							p3 = current->neurons_output[cidx + imw];
							p4 = current->neurons_output[cidx + imw + 1];
							//max = pool_max(p1, p2, p3, p4, 4);
							real_t sum = (p1 + p2 + p3 + p4);

							real_t pooled = 0;
							if (current->pool_type == 1)
							{
								pooled = sum/4;
								next_to_current->neurons_input[dst_pos] = current->biases[dst_fmap_ctr];
								next_to_current->neurons_input[dst_pos] += pooled * cweight;
							}						
							else if (current->pool_type == 2)
							{
								int idx = 0;
								pooled = pool_max(p1, p2, p3, p4, &idx, 4);
								next_to_current->neurons_input[dst_pos] = pooled;
							}						
							else if (current->pool_type == 3)
							{
								pooled = (p1 + p2 + p3 + p4)/4;
								next_to_current->neurons_input[dst_pos] = pooled;
							}
	
							if (next_to_current->layer_type == 1)
								next_to_current->neurons_output[dst_pos] = sigmoid(next_to_current->neurons_input[dst_pos]);
							if (next_to_current->layer_type == 2)
							{
								next_to_current->neurons_output[dst_pos] = htangent(next_to_current->neurons_input[dst_pos]);
							}
							if (next_to_current->layer_type == 3)
								next_to_current->neurons_output[dst_pos] = reLUSoftPlus(next_to_current->neurons_input[dst_pos]);
						}
					}
				}
			}
	
			if (flag == true)
			{
				current = next_to_current;
				next_to_current = current->next;
			}

			//we are at the last layer and we can compute miss classification rate over here		
			if (flag == false)
			{
				int mctr = 0;
				real_t max = next_to_current->neurons_output[0];
				int maxidx = 0;

				for (mctr = 0; mctr < samples->lenlable; mctr++)
				{
					if (next_to_current->neurons_output[mctr] > max)
					{
						max = next_to_current->neurons_output[mctr];
						maxidx = mctr;
					}
				}

				if(desired_label != maxidx)
					mcr++;
			}
		} 
	} 

	return ((real_t) mcr/(real_t)(samples->numVectors) * 100);	
}

real_t d_compute_missclassification_rate(cnnlayer_t *headlayer, dataset_t* samples)
{
    int d_mcr = 0;
	int sampleCtr = 0;
	for (sampleCtr = 0; sampleCtr < samples->numVectors; sampleCtr++)
	{
		cnnlayer_t* current = headlayer;
		cnnlayer_t* next_to_current = current->next;

        //This is needed as neurons_output accumulates input (+=)
        d_reset_output_vectors(headlayer);

		int inp_vec_size = current->no_of_neurons;
		int desired_label = samples->lables[sampleCtr];

		int input_data_ctr = 0;
		for (input_data_ctr = 0; input_data_ctr < inp_vec_size; input_data_ctr++)
		{
			int inputIdx = sampleCtr * inp_vec_size + input_data_ctr;
			current->neurons_input[input_data_ctr] = samples->data[inputIdx];
			current->neurons_output[input_data_ctr] = samples->data[inputIdx];
		}	

        int outSize = inp_vec_size * sizeof(real_t);
        HANDLE_ERROR(cudaMemcpy(current->d_neurons_output, current->neurons_output, outSize, cudaMemcpyHostToDevice));
        	
		bool_t flag = true;	
		while (current != NULL && flag == true)
		{	
			if (next_to_current->next == NULL)
				flag = false;

			int src_fmaps = current->no_of_fmaps;
			int dst_fmaps = next_to_current->no_of_fmaps;
			int fkernel = current->fkernel;
			int bmargin = floor(fkernel/2);
			int imh = current->fmap_height;
			int imw = current->fmap_width;
			int next_imw = next_to_current->fmap_width;
			int next_imh = next_to_current->fmap_height;

			if (next_to_current->subsampling == false && current->fkernel != 1)
			{
                real_t* d_output = next_to_current->d_neurons_output;
                real_t* d_input = current->d_neurons_output;
                real_t* d_kernel = current->d_weights;
                real_t* d_biases = current->d_biases;
               
                dim3 nBlocks(src_fmaps, dst_fmaps, 1); 
                dim3 nThreads(imw, imh, 1); 
                
                int sh_mem_size = imw * imh * sizeof(real_t) + fkernel * fkernel * sizeof(real_t);
                convolve_device_2D<<<nBlocks, nThreads, sh_mem_size>>>(d_output, d_input, d_kernel, fkernel * fkernel);
                compute_transfer_function<<<dst_fmaps, next_imw * next_imh >>>(d_output, d_biases, current->layer_type);
                cudaDeviceSynchronize();
			}
			else if (next_to_current->subsampling == false && current->fkernel == 1)
			{
			   	int src_layer_size = current->no_of_neurons;
				int dst_layer_size = next_to_current->no_of_neurons;

                real_t* d_input = current->d_neurons_output;
                real_t* d_output = next_to_current->d_neurons_output;
                real_t* d_weights = current->d_weights;
                real_t* d_biases = current->d_biases;

                dim3 nBlocks(dst_layer_size, 1, 1);
                dim3 nThreads(src_layer_size, 1, 1);

                int sh_mem_size = (2 * src_layer_size) * sizeof(real_t); 
                d_rear_DNN<<<nBlocks, nThreads, sh_mem_size>>>(d_output, d_input, d_weights);
                compute_transfer_function<<< dst_layer_size, 1 >>>(d_output, d_biases, current->layer_type);
                cudaDeviceSynchronize();
			}
			else if (next_to_current->subsampling == true)
			{
                // How to perform average pooling
				// Pattern Recognition and Machine Learning, By Christopher M. Bishop (P267)
				// ... Each subsampling unit might take inputs from a 2x2 unit region in the 
				// corresponding feature map and would compute the average of 
				// those inputs, multiplied by an adaptive weight with the addition of an adaptive bias
				// parameter, and then transformed using a sigmoidal non-linear activation function. 

                real_t* d_input = current->d_neurons_output;
                real_t* d_output = next_to_current->d_neurons_output;
                int* d_gradientMap = next_to_current->d_gradientMap;

                dim3 nBlocks(src_fmaps, 1, 1);
                dim3 nThreads(imw, imh, 1);

                int sh_mem_size = imw * imh * sizeof(real_t);
                d_subsampling<<<nBlocks, nThreads, sh_mem_size>>>(d_output, d_input, d_gradientMap, current->layer_type);
                cudaDeviceSynchronize();
			}
	
			if (flag == true)
			{
				current = next_to_current;
				next_to_current = current->next;
			}

            if (flag == false)
            {
                int noutSize = next_to_current->no_of_neurons * sizeof(real_t);
                HANDLE_ERROR(cudaMemcpy(next_to_current->neurons_output, next_to_current->d_neurons_output, noutSize, cudaMemcpyDeviceToHost));

				int mctr = 0;
				real_t max = next_to_current->neurons_output[0];
				int maxidx = 0;

				for (mctr = 0; mctr < samples->lenlable; mctr++)
				{
					if (next_to_current->neurons_output[mctr] > max)
					{
						max = next_to_current->neurons_output[mctr];
						maxidx = mctr;
					}
				}

				if(desired_label != maxidx)
					d_mcr++;
			}
		} 
	} 

	return ((real_t) d_mcr/(real_t)(samples->numVectors) * 100);	
}
