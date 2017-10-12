#ifndef __READ_ESD__
#define __READ_ESD__

#include "dataset.h"

int read_data_files(dataset_t* input_samples, dataset_t* target_samples);

int read_training_samples(dataset_t* input_samples);

int read_test_samples(dataset_t* target_samples);


#endif
