#ifndef __ANN_H__
#define __ANN_H__

#include "layer.h"

#include "dataset.h"
#define MOMENTUM 	0.9
#define MSE_LIMIT 	0.000001

#ifdef __cplusplus
extern "C"
{
#endif

/* functions related to (CPU) */
int h_feed_forward(cnnlayer_t *headlayer, double *training_data, int *batch_indexes);
void h_compute_gradients_deltas(struct nnlayer *headlayer, int nouts,  unsigned char* desired_output, int* batch_indexes);
void h_update_weights(struct nnlayer* current);
real_t h_compute_missclassification_rate(cnnlayer_t *head, dataset_t *test_samples);
void average_deltas(struct nnlayer* headlayer);
real_t sigmoid(real_t x);
void reset_inputs_dweights_deltas(cnnlayer_t* headlayer);

real_t computeEntropy(real_t a11, real_t a12, real_t a13, real_t a14);
real_t stochastic_pooling(real_t a01, real_t a02, real_t a03, real_t a04, int* poolingIdx);
real_t htangent(real_t x);
real_t qhtangent(real_t x, int M, real_t cdelta);
real_t reLUSoftPlus(real_t x);
int sign(real_t n1, real_t n2);

/* functions related to device (GPU) */
int d_feed_forward(cnnlayer_t *headlayer, double *training_data, int *batch_indexes);
void d_compute_gradients_deltas(struct nnlayer *headlayer, int nouts,  unsigned char* desired_output, int* batch_indexes);
void d_reset_vectors(struct nnlayer* current);
void d_reset_output_vectors(cnnlayer_t* headlayer);
void d_update_weights(struct nnlayer* current);
real_t d_compute_missclassification_rate(cnnlayer_t *headlayer, dataset_t* samples);

real_t dreLUSoftPlus(real_t x);
real_t dhtangent(real_t x);

/* Common between CPU and GPU */
void train_cnn(cnnlayer_t* head, struct dataset* train_samples, struct dataset* test_samples);
void hd_reset_biases(cnnlayer_t* headlayer);

#ifdef __cplusplus
}
#endif


#endif
