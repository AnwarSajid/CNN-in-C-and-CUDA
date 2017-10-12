#ifndef __READ_MNIST__
#define __READ_MNIST__

#include "dataset.h"

int readMNISTDB(struct dataset* train_samples, struct dataset* test_samples);

int readTrainingFiles(struct dataset* train_samples);

int readTestFiles(struct dataset* test_samples);

int imageLabelOrg(int imageIdx, int testTrainFlag);

int imageLabelGen(unsigned char *inputVector, float *weightVector, float *biasesVector);



#endif
