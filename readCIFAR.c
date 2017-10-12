//File Headers provided at yann.lecun.com/exdb/mnist/
//Sajid Anwar July 01, 2013
 
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"
#include "readCIFAR.h"
#include "dataset.h"

//#define __lreadDebug //lable reading debug messages 
//#define __freadDebug //file reading debug messages

int readCIFAR(struct dataset* train_samples, struct dataset* test_samples)
{
	//reading Training files
	int rTrain = readCIFAR_TrainingFiles(train_samples);
	if (rTrain == -1)	
		fprintf(stderr, "\n\n Returned with error while reading training files");	

	//reading Test files
	int rTest = readCIFAR_TestFiles(test_samples);
	if (rTest == -1)
		fprintf(stderr, "\n\n Returned with error while reading test files");	

	return 0;
}

int readCIFAR_TestFiles(struct dataset* test_samples)
{
	char *test_set_file = "../CNN-CIFAR10/CIFAR/cifar-10-batches-bin/test_batch.bin";  	
	FILE *fpTestImagesLables = NULL;
   	fpTestImagesLables = fopen(test_set_file,"rb");

 	if (fpTestImagesLables == NULL)
	{
		printf("\nCan't read %s file", test_set_file);
		return -1;
	}
	else
	{
		printf("\nSuccessfully read %s file", test_set_file);

		unsigned char tempA[3072];			
		int counter = 0;
		for (counter = 0; counter < 10000; counter++)
		{
			int lv = test_samples->lenVector;
			int lableIdx = counter;
			int sampleIdx = counter * lv;

			int sizer = fread(&test_samples->lables[lableIdx], sizeof(unsigned char), 1, fpTestImagesLables);
			sizer =	fread(tempA, sizeof(unsigned char), 3072, fpTestImagesLables);

			if (sizer != 3072)
			{
				fprintf(stderr, "\nproblem in reading the file");
				return -1;
			}

			// allocate tempA to train_samples->data
			int i = 0;
			for (i = 0; i < 3072; i++)
			{
				test_samples->data[sampleIdx + i] = (255.0 - tempA[i])/128.0 - 1.0;
			}
		}
		fclose(fpTestImagesLables);
	}	 	 
	
	return 0;
}

int readCIFAR_TrainingFiles(struct dataset* train_samples)
{
	char *training_files[5] = {"../CNN-CIFAR10/CIFAR/cifar-10-batches-bin/data_batch_1.bin",  	
								"../CNN-CIFAR10/CIFAR/cifar-10-batches-bin/data_batch_2.bin",  	
								"../CNN-CIFAR10/CIFAR/cifar-10-batches-bin/data_batch_3.bin",  	
								"../CNN-CIFAR10/CIFAR/cifar-10-batches-bin/data_batch_4.bin",  	
								"../CNN-CIFAR10/CIFAR/cifar-10-batches-bin/data_batch_5.bin"};  	

	unsigned char tempA[3072];
	int fileCounter = 0;
	for (fileCounter = 0; fileCounter < 5; fileCounter++)
	{
		char *TrainImagesLables = training_files[fileCounter]; 
   		FILE *fpTrainImagesLables = NULL;
   		fpTrainImagesLables = fopen(TrainImagesLables,"rb");

		if (fpTrainImagesLables == NULL)
		{
			printf("\nCan't read %s file", TrainImagesLables);
			return -1;
		}
		else
		{
			printf("\nSuccessfully read %s file", TrainImagesLables);
			
			int counter = 0;
			for (counter = 0; counter < 10000; counter++)
			{
				int lv = train_samples->lenVector;
				int lableIdx = fileCounter * 10000 + counter;
				int sampleIdx = fileCounter * (10000 * lv) + counter * lv;


				// allocate for double ...

				int sizer = fread(&train_samples->lables[lableIdx], 1, 1, fpTrainImagesLables);
				sizer = fread(tempA, 1, 3072, fpTrainImagesLables);
				if (sizer != 3072)
				{
					fprintf(stderr, "\n Problem in reading the file");
					return -1;
				}
				// allocate tempA to train_samples->data
				int i = 0;
				for (i = 0; i < 3072; i++)
				{
					train_samples->data[sampleIdx + i] = (255.0 - tempA[i])/128.0 - 1.0;
				}
			}
			fclose(fpTrainImagesLables);
		}
	}

	return 0;
}
