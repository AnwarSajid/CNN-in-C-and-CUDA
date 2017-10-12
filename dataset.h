#ifndef __DATASET_H__
#define __DATASET_H__


//typedef int bool_t;

#define true 1
#define false 0

#include "layer.h"

typedef struct dataset dataset_t;

struct dataset
{
	int numVectors;
	int lenVector;
	double* data;	
	unsigned char* lables;	
	int lenlable;	
};

#ifdef __cplusplus
extern "C"
{
#endif
    struct dataset* create_data_container(int numVectors, int lenVector, int lenlable);
    void free_data_container(dataset_t* dataVector);
    void generate_lables_eye_closure(dataset_t* training_samples, dataset_t* test_samples);
    int argmax(double p1, double p2, double p3, double p4, int count);
    double pool_max(double p1, double p2, double p3, double p4, int* idx, int count);
    void save_trained_network_weights(cnnlayer_t* headlayer, char* filename); 
    void save_trained_network_qweights(cnnlayer_t* headlayer, char* filename); 
#ifdef __cplusplus
}
#endif

#endif
