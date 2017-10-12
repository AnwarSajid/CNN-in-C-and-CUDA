//read eye samples (open and closed, ZJU dataset)
 
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"
#include "read_ESD_samples.h"
#include "dataset.h"


int read_data_files(dataset_t* train_samples, dataset_t* test_samples)
{
	//reading input training samples
	int rTrain = read_training_samples(train_samples);
	if (rTrain == -1)	
		fprintf(stderr, "\n\n Returned with error while reading training files");	

	//reading Target samples
	int rTest = read_test_samples(test_samples);
	if (rTest == -1)
		fprintf(stderr, "\n\n Returned with error while reading target files");	

	return 0;
}

int read_test_samples(dataset_t* test_samples)
{
	char *filename_test_open = "ESD/open_test"; 
	char *filename_test_closed = "ESD/closed_test";
 
	FILE *fp_test_open = NULL, *fp_test_closed = NULL;
	fp_test_open = fopen(filename_test_open,"rb");
	fp_test_closed = fopen(filename_test_closed,"rb");

	if (fp_test_open != NULL && fp_test_closed != NULL) 
	{
		fprintf(stderr, "\n\n ********* Test samples (Open) *****************************");
		fseek(fp_test_open, 0, SEEK_END);
		long fSizeI = ftell(fp_test_open);
		fprintf(stderr, "\n Image File Size: %ld\n", fSizeI);
		rewind(fp_test_open);

		unsigned char *bufferImages = (unsigned char *) malloc(fSizeI * sizeof(unsigned char)); 
		size_t readSizeI = fread(bufferImages, sizeof(char), fSizeI, fp_test_open);
		if (readSizeI != fSizeI)
		{
			fprintf(stderr, "\n\n File has NOT been read properly ");
			fclose(fp_test_open);
			free(bufferImages);
			return -1;
		}

		//allocate to the test_sample->data
		int counter = 0;
		for (counter = 0; counter < fSizeI; counter++)
		{
			test_samples->data[counter] = (255.0 - bufferImages[counter])/128.0 - 1.0;			
		}
		free(bufferImages);
		
		int lIdx = fSizeI;
		fprintf(stderr, " Reading Open eyes samples:\n");
		fprintf(stderr, " no. of Images: %ld Image (row, col) size: 24 x 24", fSizeI/(24*24));
		fclose(fp_test_open);

		fprintf(stderr, "\n\n ********* Test samples (closed) *****************************");
		fseek(fp_test_closed, 0, SEEK_END);
		fSizeI = ftell(fp_test_closed);
		fprintf(stderr, "\n Image File Size: %ld\n", fSizeI);
		rewind(fp_test_closed);
		
		bufferImages = (unsigned char *) malloc(fSizeI * sizeof(unsigned char)); 
		readSizeI = fread(bufferImages, sizeof(char), fSizeI, fp_test_closed);
		if (readSizeI != fSizeI)
		{
			fprintf(stderr, "\n\n File has NOT been read properly ");
			fclose(fp_test_closed);
			free(bufferImages);
			return -1;
		}

		//allocate to the test_sample->data
		for (counter = 0; counter < fSizeI; counter++)
		{
			test_samples->data[lIdx + counter] = (255.0 - bufferImages[counter])/128.0 - 1.0;			
		}

		free(bufferImages);
		
		fprintf(stderr, " Reading Closed eyes samples:\n");
		fprintf(stderr, " no. of Images: %ld Image (row, col) size: 24 x 24", fSizeI/(24*24));
		fclose(fp_test_closed);
	}
	else
	{
		fprintf(stderr, "\n\n File has NOT been successfully opened");
	}

	return 0;
}

int read_training_samples(dataset_t* training_samples)
{
	char *filename_train_open = "ESD/open_train"; 
	char *filename_train_closed = "ESD/closed_train";
 
	FILE *fp_train_open = NULL, *fp_train_closed = NULL;
	fp_train_open = fopen(filename_train_open,"rb");
	fp_train_closed = fopen(filename_train_closed,"rb");


	if (fp_train_open != NULL && fp_train_closed != NULL) 
	{
		fprintf(stderr, "\n\n ********* Training samples (Open) *****************************");
		fseek(fp_train_open, 0, SEEK_END);
		long fSizeI = ftell(fp_train_open);
		fprintf(stderr, "\n Image File Size: %ld\n", fSizeI);
		rewind(fp_train_open);

		unsigned char *bufferImages = (unsigned char*) malloc(sizeof(unsigned char) * fSizeI);
		size_t readSizeI = fread(bufferImages, sizeof(char), fSizeI, fp_train_open);
		if (readSizeI != fSizeI)
		{
			fprintf(stderr, "\n\n File has NOT been read properly ");
			fclose(fp_train_open);
			free(bufferImages);
			return -1;
		}

		int counter = 0;
		for (counter = 0; counter < fSizeI; counter++)
		{
			training_samples->data[counter] = (255.0 - bufferImages[counter])/128.0 - 1.0;
		}
		free(bufferImages);

		int lIdx = fSizeI;
		fprintf(stderr, " Reading Open eyes samples:\n");
		fprintf(stderr, " no. of Images: %ld Image (row, col) size: 24 x 24", fSizeI/(24*24));
		//fprintf(stderr, " no. of Images: %ld Image (row, col) size: 40 x 100 x 3", fSizeI/(40*100*3));
		fclose(fp_train_open);

		fprintf(stderr, "\n\n ********* Training samples (Closed) *****************************");
		fseek(fp_train_closed, 0, SEEK_END);
		fSizeI = ftell(fp_train_closed);
		fprintf(stderr, "\n Image File Size: %ld\n", fSizeI);
		rewind(fp_train_closed);

		bufferImages = (unsigned char*) malloc(sizeof(unsigned char) * fSizeI);
		readSizeI = fread(bufferImages, sizeof(char), fSizeI, fp_train_closed);

		if (readSizeI != fSizeI)
		{
			fprintf(stderr, "\n\n File has NOT been read properly ");
			fclose(fp_train_closed);
			free(bufferImages);
			return -1;
		}

		for (counter = 0; counter < fSizeI; counter++)
		{
			training_samples->data[lIdx + counter] = (255.0 - bufferImages[counter])/128.0 - 1.0;
		}
		free(bufferImages);

		fprintf(stderr, " Reading Closed eyes samples:\n");
		fprintf(stderr, " no. of Images: %ld Image (row, col) size: 24 x 24", fSizeI/(24*24));
		fclose(fp_train_closed);
	}
	else
	{
		fprintf(stderr, "\n\n File has NOT been successfully opened");
	}

	return 0;
}
