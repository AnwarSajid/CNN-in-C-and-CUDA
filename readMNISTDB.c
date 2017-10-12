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

//Reading MNIST Image DataBase Files
//File Headers provided at yann.lecun.com/exdb/mnist/
//Sajid Anwar July 01, 2013
 
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"
#include "readMNISTDB.h"
#include "dataset.h"

int inputVectorSize = 1024;

//#define __lreadDebug //lable reading debug messages 
//#define __freadDebug //file reading debug messages

int readMNISTDB(struct dataset* train_samples, struct dataset* test_samples)
{
	//reading Training files
	inputVectorSize = train_samples->lenVector;
	
	int rTrain = readTrainingFiles(train_samples);
	if (rTrain == -1)	
		fprintf(stderr, "\n\n Returned with error while reading training files");	

	//reading Test files
	int rTest = readTestFiles(test_samples);
	if (rTest == -1)
		fprintf(stderr, "\n\n Returned with error while reading test files");	

	return 0;
}

int readTestFiles(struct dataset* test_samples)
{
	char *filenameTestImages = "../Cnnfull2/MNIST/t10k-images-idx3-ubyte-resized"; //32 by 32
	char *filenameTestLables = "../Cnnfull2/MNIST/t10k-labels-idx1-ubyte";

	FILE *fpTestImages = NULL;
	FILE *fpTestLables = NULL;

   fpTestImages = fopen(filenameTestImages,"rb");
   fpTestLables = fopen(filenameTestLables,"rb");

   unsigned char *bufferImages;
   unsigned char *bufferLables;

   if (fpTestImages != NULL && fpTestLables != NULL)
   {
      fprintf(stderr, "\n\n ********* Test Files *****************************");
      fseek(fpTestImages, 0, SEEK_END);
      long fSizeI = ftell(fpTestImages);
      fprintf(stderr, "\n Image File Size: %ld\n", fSizeI);
      rewind(fpTestImages);
      bufferImages = (unsigned char*) malloc(fSizeI);
      size_t readSizeI = fread(bufferImages, sizeof(char), fSizeI, fpTestImages);

		//Test label file
      fseek(fpTestLables, 0, SEEK_END);
      long fSizeL = ftell(fpTestLables);
      fprintf(stderr, " Lable File Size: %ld\n", fSizeL);
      rewind(fpTestLables);
      bufferLables = (unsigned char*) malloc(fSizeL);
      size_t readSizeL = fread(bufferLables, sizeof(char), fSizeL, fpTestLables);

      if (readSizeI != fSizeI && readSizeL != fSizeL)
      {
         fprintf(stderr, "\n\n File has NOT been read properly ");
         fclose(fpTestImages);
         fclose(fpTestLables);
         return -1;
      }

      int counter = 0;
      int magicNumber = 0;
      int noImages = 0;
      int noRows = 0, noCols = 0;

		//little endian format (not the default one, because I resized in Matlab)
		bool_t little_endian = false;
		if (little_endian == true)
		{ 
      	magicNumber = (bufferImages[0]) | bufferImages[1] << 8 | bufferImages[2] << 16 | bufferImages[3] << 24;
      	noImages = bufferImages[4] | bufferImages[5] << 8 | bufferImages[6] << 16 | bufferImages[7] << 24;
      	noRows = bufferImages[8] | bufferImages[9] << 8 | bufferImages[10] << 16 | bufferImages[11] << 24;
      	noCols = bufferImages[12] | bufferImages[13] << 8 | bufferImages[14] << 16 | bufferImages[15] << 24;
		}
		else
		{
		 	magicNumber = (bufferImages[3]) | bufferImages[2] << 8 | bufferImages[1] << 16 | bufferImages[0] << 24;
      	noImages = bufferImages[7] | bufferImages[6] << 8 | bufferImages[5] << 16 | bufferImages[4] << 24;
      	noRows = bufferImages[11] | bufferImages[10] << 8 | bufferImages[9] << 16 | bufferImages[8] << 24;
      	noCols = bufferImages[15] | bufferImages[14] << 8 | bufferImages[13] << 16 | bufferImages[12] << 24;
		}

	   fprintf(stderr, " Reading Test Images:\n");
	   fprintf(stderr, " MagicNumber: 0x%08x, no. of Images: %d Image (row, col) size: %d, %d resized to 32x32\n", magicNumber, noImages, noRows, noCols);

		//default big endian format
      magicNumber = (bufferLables[3]) | bufferLables[2] << 8 | bufferLables[1] << 16 | bufferLables[0] << 24;
      noImages = bufferLables[7] | bufferLables[6] << 8 | bufferLables[5] << 16 | bufferLables[4] << 24;
	   fprintf(stderr, " Reading Test set Labels:\n");
	   fprintf(stderr, " MagicNumber: 0x%08x, no. of Image Lables: %d \n\n", magicNumber, noImages);

		int labelCounterF = 8;
		int labelCounterD = 0;
      int imgCounter = -1;
      for (counter = 16; counter < fSizeI; counter = counter + inputVectorSize)
      {
         imgCounter = imgCounter + 1;
         int ctr = 0;
			int bimgctr = counter;
         for (ctr = 0; ctr < inputVectorSize; ctr++)
         {
            test_samples->data[imgCounter * inputVectorSize + ctr] = (255.0 - bufferImages[bimgctr])/128.0 - 1.0;
				bimgctr = bimgctr + 1;
         }

			int currlabel = bufferLables[labelCounterF];
         test_samples->lables[labelCounterD] = currlabel;

#ifdef __freadDebug
         fprintf(stderr, "\n labelcounter(d,f): %d, %d, Train60kLables: %d", labelCounterD, labelCounterF, Test10kLables[labelCounterD]);
#endif
         labelCounterF++;
         labelCounterD++;
      }

      fclose(fpTestImages);
      fclose(fpTestLables);
      free(bufferImages);
      free(bufferLables);
   }
   else
   {
      fprintf(stderr, "\n\n File has NOT been successfully opened");
   }
	
	return 0;
}

int readTrainingFiles(struct dataset* train_samples)
{
   char *filenameTrainImages = "../Cnnfull2/MNIST/train-images-idx3-ubyte-resized";
   char *filenameTrainLables = "../Cnnfull2/MNIST/train-label-idx1-ubyte";

   FILE *fpTrainImages = NULL;
   FILE *fpTrainLables = NULL;

   fpTrainImages = fopen(filenameTrainImages,"rb");
   fpTrainLables = fopen(filenameTrainLables,"rb");

   unsigned char *bufferImages;
   unsigned char *bufferLables;

   if (fpTrainImages != NULL && fpTrainLables != NULL)
   {
      fprintf(stderr, "\n\n ************  Training Files **********************");
      fseek(fpTrainImages, 0, SEEK_END);
      long fSizeI = ftell(fpTrainImages);
      fprintf(stderr, "\n Image File Size: %ld\n", fSizeI);
      rewind(fpTrainImages);
      bufferImages = (unsigned char*) malloc(fSizeI);
      size_t readSizeI = fread(bufferImages, sizeof(char), fSizeI, fpTrainImages);

		fseek(fpTrainLables, 0, SEEK_END);
      long fSizeL = ftell(fpTrainLables);
      fprintf(stderr, " Label File Size: %ld\n", fSizeL);
      rewind(fpTrainLables);
      bufferLables = (unsigned char*) malloc(fSizeL);
      size_t readSizeL = fread(bufferLables, sizeof(char), fSizeL, fpTrainLables);


		if (readSizeI != fSizeI && readSizeL != fSizeL)
      {
         fprintf(stderr, "\n\n File has NOT been read properly ");
         fclose(fpTrainImages);
         fclose(fpTrainLables);
         return -1;
      }

      int counter = 0;
      int magicNumber = 0;
      int noImages = 0;
      int noRows = 0, noCols = 0;

		bool_t little_endian = false;

		if (little_endian == true)
		{
      	magicNumber = (bufferImages[0]) | bufferImages[1] << 8 | bufferImages[2] << 16 | bufferImages[3] << 24;
      	noImages = bufferImages[4] | bufferImages[5] << 8 | bufferImages[6] << 16 | bufferImages[7] << 24;
      	noRows = bufferImages[8] | bufferImages[9] << 8 | bufferImages[10] << 16 | bufferImages[11] << 24;
      	noCols = bufferImages[12] | bufferImages[13] << 8 | bufferImages[14] << 16 | bufferImages[15] << 24;
		}
		else
		{
      	magicNumber = (bufferImages[3]) | bufferImages[2] << 8 | bufferImages[1] << 16 | bufferImages[0] << 24;
      	noImages = bufferImages[7] | bufferImages[6] << 8 | bufferImages[5] << 16 | bufferImages[4] << 24;
      	noRows = bufferImages[11] | bufferImages[10] << 8 | bufferImages[9] << 16 | bufferImages[8] << 24;
      	noCols = bufferImages[15] | bufferImages[14] << 8 | bufferImages[13] << 16 | bufferImages[12] << 24;
		}

      fprintf(stderr, " Reading Train Images:");
      fprintf(stderr, " MagicNumber: 0x%08x, no. of Images: %d Image (row, col) size: %d, %d resized to 32x32", magicNumber, noImages, noRows, noCols);

      magicNumber = (bufferLables[3]) | bufferLables[2] << 8 | bufferLables[1] << 16 | bufferLables[0] << 24;
      noImages = bufferLables[7] | bufferLables[6] << 8 | bufferLables[5] << 16 | bufferLables[4] << 24;
      fprintf(stderr, "\n Reading Train set Labels: ");
      fprintf(stderr, " MagicNumber: 0x%08x, no. of Image Lables: %d \n", magicNumber, noImages);


      int labelCounterF = 8;
      int labelCounterD = 0;
      int imgCounter = -1;
      for (counter = 16; counter < fSizeI; counter = counter + inputVectorSize)
      {
         imgCounter = imgCounter + 1;
         int ctr = counter;
			int bimgctr = counter;
         for (ctr = 0; ctr < inputVectorSize; ctr++)
         {
            train_samples->data[imgCounter * inputVectorSize + ctr] = (255.0 - bufferImages[bimgctr])/128.0 - 1.0;
				bimgctr = bimgctr + 1;
         }

			int currlabel = bufferLables[labelCounterF];
			train_samples->lables[labelCounterD] = currlabel;

#ifdef __freadDebug
			fprintf(stderr, "\n labelcounter(d,f): %d, %d, Train60kLables: %d", labelCounterD, labelCounterF, Train60kLables[labelCounterD]);
#endif
			labelCounterF++;
			labelCounterD++;
      }
		
      fclose(fpTrainImages);
      fclose(fpTrainLables);
      free(bufferImages);
      free(bufferLables);
   }
   else
   {
      fprintf(stderr, "\n\n File has NOT been successfully opened");
   }

	return 0;
}
/*
int imageLabelOrg(int imageIdx, int testTrainFlag)
{
	if (testTrainFlag == 1)
	{
		// fetch label for corresponding test images
		if (imageIdx >= 0 && imageIdx < 10000)
			return Test10kLables[imageIdx];
	}
	else
	{
		if (imageIdx >= 0 && imageIdx < 60000)
			return Train60kLables[imageIdx];
	}
	
	return 0;
}

int imageLabelGen(unsigned char *inputVector, float *weightVector, float *biasesVector)
{
	//Generated labels
	int genLable = rand() % 10;
	return genLable;	
}
*/
