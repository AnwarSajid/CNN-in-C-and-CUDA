#ifndef __READ_CIFAR_INC__
#define __READ_CIFAR_INC__

#include "dataset.h"

int readCIFAR(struct dataset* train_samples, struct dataset* test_samples);

int readCIFAR_TrainingFiles(struct dataset* train_samples);

int readCIFAR_TestFiles(struct dataset* test_samples);

#endif
