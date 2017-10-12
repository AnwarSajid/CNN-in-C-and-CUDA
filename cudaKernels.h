#ifndef __CUDA_KERNELS__
#define __CUDA_KERNELS__

#include "layer.h"

__global__ void compute_transfer_function(real_t* d_output, real_t* d_biases, int layer_type);

__global__ void convolve_device_2D(real_t* d_output, real_t* d_input, real_t* d_kernel, int kerSize);

__global__ void errorbp_convolution_layers2(real_t* d_output, real_t* d_lerr_deltas, real_t* d_cerr_deltas, real_t* d_weights, real_t* d_delta_weights, int kerSize);

__global__ void errorbp_convolution_update_biases(real_t* d_lerr_deltas, real_t* d_delta_biases);

__global__ void errorbp_convolution_layers(real_t* d_output, real_t* d_lerr_deltas, real_t* d_cerr_deltas, real_t* d_weights, real_t* d_delta_weights, int kerSize);

__global__ void d_errbp_subsampling(real_t* d_output, real_t* d_lerr_deltas, real_t* d_cerr_deltas, int* d_gradientMap, int layer_type);

__global__ void d_subsampling(real_t* d_output, real_t* d_input, int* d_gradientMap, int layer_type);

__global__ void d_rear_DNN(real_t* d_output, real_t* d_input, real_t* d_weights);

__global__ void d_update_weights_kernel2(real_t* d_weights, real_t* d_delta_weights, double d_LEARNING_RATE);

__global__ void d_update_weights_kernel(real_t* d_weights, real_t* d_delta_weights, double d_LEARNING_RATE);

__global__ void d_update_biases_kernel(real_t* d_biases, real_t* d_delta_biases, double d_LEARNING_RATE);

__global__ void d_rear_DNN_update_error_deltas(real_t* d_output, real_t* d_cerr_deltas, int layer_type);

__global__ void d_update_error_deltas(real_t* d_output, real_t* d_cerr_deltas, int layer_type);

__global__ void d_rear_DNN_errorbp(real_t* d_output, real_t* d_lerr_deltas, real_t* d_cerr_deltas, real_t* d_weights,real_t* d_delta_weights, real_t* d_delta_biases);

void copy_dweights_to_hweights(cnnlayer_t* headlayer);

void copy_hweights_to_dweights(cnnlayer_t* headlayer);

#endif
