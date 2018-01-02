#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

#define HISTOGRAM_SIZE 256

#define cudaErrorCheck()                                                                      \
{                                                                                             \
	cudaError_t error = cudaGetLastError();                                                   \
	if (error != cudaSuccess)                                                                 \
	{                                                                                         \
		printf("\033[1;31m");                                                                 \
		printf("Cuda Error Found %s:%d:  '%s'\n", __FILE__, __LINE__, cudaGetErrorString(error));          \
		printf("\033[0m");                                                                  \
		freeMemory(d_lut, d_outImage);                                       \
	}                                                                                         \
}

__constant__ int d_hist[HISTOGRAM_SIZE];

__global__ void prefix_sum_cdf(int *lut, int img_size, int min)
{
	__shared__ int s_cdf[512]; //double buffer: 2 rows of 256 indices each
	int imageSize = img_size;
	int d_min = min;
	int idx = threadIdx.x;
	int output = 0, input = 1;
	int result;
	int d = 1; //offset

	//Load hist into shared memory
	s_cdf[idx] = d_hist[idx];
	__syncthreads();

	//calculate cdf using prefix sum technic
	for ( ; d < HISTOGRAM_SIZE; d = d << 1)
	{
		output = 1 - output;
		input = 1 - input;

		if (idx >= d)
		{
			s_cdf[output*HISTOGRAM_SIZE + idx] = s_cdf[input*HISTOGRAM_SIZE + idx] + s_cdf[input*HISTOGRAM_SIZE + idx - d];
		}
		else
		{
			s_cdf[output*HISTOGRAM_SIZE + idx] = s_cdf[input*HISTOGRAM_SIZE + idx];
		}
		__syncthreads();
	}
	//calculate the lut
	result = (int)((((float)s_cdf[output*HISTOGRAM_SIZE + idx] - d_min)*255) / (imageSize - d_min) + 0.5);
	
	if (result < 0) result = 0;
	if (result > 255) result = 255;

	lut[idx] = result;
	
}

void freeMemory(int *d_lut, unsigned char *outputImage)
{
	cudaError_t err;

	if (outputImage != NULL)
	{
		err = cudaFree(outputImage);
		if (err != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error during cudaFree (outputImage):  %s\n", cudaGetErrorString(err));
			printf("\033[0m");
		}
	}

	if (d_lut != NULL)
	{
		err = cudaFree(d_lut);
		if (err != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error during cudaFree (d_lut):  %s\n", cudaGetErrorString(err));
			printf("\033[0m");
		}
	}

	printf("Reset Device\n");
	err = cudaDeviceReset();
	if (err != cudaSuccess){
		printf("\033[1;31m");
		printf("Error during cudaDeviceReset:  %s\n", cudaGetErrorString(err));
		printf("\033[0m");
	}
}

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
	//counting variable
		//histogram
		struct timespec histogram_start, histogram_stop;
		double histogram_time;
		//struct timespec hist_equal_start, hist_equal_stop;
		double hist_equal_time;

		//histogram_equalization
		struct timespec hist_equal_malloc_start, hist_equal_malloc_stop;
		struct timespec hist_equal_HtoD_start, hist_equal_HtoD_stop;
		struct timespec hist_equal_DtoH_start, hist_equal_DtoH_stop;
		struct timespec hist_equal_findMin_start, hist_equal_findMin_stop;
		struct timespec hist_equal_result_start, hist_equal_result_stop;
		double hist_equal_malloc_time, hist_equal_HtoD_time, hist_equal_DtoH_time, hist_equal_findMin_time, hist_equal_result_time;
		cudaEvent_t prefix_start, prefix_stop;
		float prefix_elapsed;
	//__|

    PGM_IMG result;
	int h_hist[HISTOGRAM_SIZE];
	int h_lut[HISTOGRAM_SIZE];
	int min = 0;
	int i = 0;
	int imageSize = img_in.w * img_in.h;
	
	//device variables
	cudaError_t error;
	int *d_lut = NULL;
	unsigned char *d_outImage = NULL;
    
    result.w = img_in.w;
    result.h = img_in.h;
	result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
	
	//Allocate device (GPU) memory for prefix_sum kernel
		clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_malloc_start);
		error = cudaMalloc((void **)&d_lut, HISTOGRAM_SIZE * sizeof(int));
			if (error != cudaSuccess)
			{
				printf("\033[1;31m");
				printf("Error allocating memory on host for d_lut:   %s\n", cudaGetErrorString(error));
				printf("\033[0m");
				freeMemory(d_lut, d_outImage);
				return (result);
			}

		clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_malloc_stop);
	//__|

	clock_gettime(CLOCK_MONOTONIC_RAW, &histogram_start);
	histogram(h_hist, img_in.img, imageSize, HISTOGRAM_SIZE);
	clock_gettime(CLOCK_MONOTONIC_RAW, &histogram_stop);

	printf("Done with histogram\n");

	dim3 prefix_grid(1, 1);  //We can use only one block, because we only need 256 threads << 1024 threads per block.
	dim3 prefix_block(HISTOGRAM_SIZE); //1D block geometry

	//MemCpy h_hist to __constant__ memory (Host to Device)
		clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_HtoD_start);
		error = cudaMemcpyToSymbol(d_hist, h_hist, HISTOGRAM_SIZE * sizeof(int));
		if (error != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error during cudaMemcpyToSymbol of h_hist to d_hist:  %s\n", cudaGetErrorString(error));
			printf("\033[0m");
			freeMemory(d_lut, d_outImage);
			return (result);
		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_HtoD_stop);
	//__|

	clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_findMin_start);
	while(min == 0){
        min = h_hist[i++];
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_findMin_stop);

	printf("Done with findMin\n");

	cudaEventCreate(&prefix_start);
	cudaEventCreate(&prefix_stop);

	cudaEventRecord(prefix_start, 0);
	//Prefix SUM CDF kernel
		prefix_sum_cdf <<< prefix_grid, prefix_block >>> (d_lut, imageSize, min);
	//__|
	cudaEventRecord(prefix_stop, 0);

	printf("Done with prefix_sum\n");

	//MemCpy Device to Host
		clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_DtoH_start);
		error = cudaMemcpy(h_lut, d_lut, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error during cudaMemcpy of d_lut to h_lut:  %s\n", cudaGetErrorString(error));
			printf("\033[0m");
			freeMemory(d_lut, d_outImage);
			return (result);
		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_DtoH_stop);
	//__|
	
	clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_result_start);
	/* Get the result image */
    for(i = 0; i < imageSize; i ++){
        result.img[i] = (unsigned char)h_lut[img_in.img[i]];        
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_result_stop);

	//Error Checking
	cudaErrorCheck();

	//clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_start);
	//histogram_equalization(result.img,img_in.img, h_hist,result.w*result.h, HISTOGRAM_SIZE);
	//clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_stop);

	//Calculate total time
		//histogram
			histogram_time = (double)(histogram_stop.tv_nsec - histogram_start.tv_nsec) / 1000000000.0 +
				(double)(histogram_stop.tv_sec - histogram_start.tv_sec);
			printf("\033[1;34m"); //Bold blue
			printf("histogram = ");
			printf("\033[1;32m"); //Bold green
			printf("%10g seconds\n", histogram_time);
		//histogram_equalization_cpu
			// hist_equal_time = (double)(hist_equal_stop.tv_nsec - hist_equal_start.tv_nsec) / 1000000000.0 +
			// 	(double)(hist_equal_stop.tv_sec - hist_equal_start.tv_sec);
			// printf("\033[1;34m"); //Bold blue
			// printf("histogram_equalization (CPU) = ");
			// printf("\033[1;35m"); //Bold magenta
			// printf("%10g seconds\n", hist_equal_time);
		//histogram_equalization_gpu
			hist_equal_malloc_time = (double)(hist_equal_malloc_stop.tv_nsec - hist_equal_malloc_start.tv_nsec) / 1000000000.0 +
				(double)(hist_equal_malloc_stop.tv_sec - hist_equal_malloc_start.tv_sec);
			printf("\033[1;34m"); //Bold blue
			printf("prefix_sum malloc = ");
			printf("\033[1;35m"); //Bold magenta
			printf("%10g seconds\n", hist_equal_malloc_time);

			hist_equal_HtoD_time = (double)(hist_equal_HtoD_stop.tv_nsec - hist_equal_HtoD_start.tv_nsec) / 1000000000.0 +
				(double)(hist_equal_HtoD_stop.tv_sec - hist_equal_HtoD_start.tv_sec);
			printf("\033[1;34m"); //Bold blue
			printf("prefix sum MemCpy HtoD = ");
			printf("\033[1;35m"); //Bold magenta
			printf("%10g seconds\n", hist_equal_HtoD_time);

			cudaEventSynchronize(prefix_stop);
			cudaEventElapsedTime(&prefix_elapsed, prefix_start, prefix_stop);
			printf("\033[1;34m");
			printf("prefix_sum_cdf = ");
			printf("\033[1;35m");
			printf("%g ms\n", prefix_elapsed);

			hist_equal_DtoH_time = (double)(hist_equal_DtoH_stop.tv_nsec - hist_equal_DtoH_start.tv_nsec) / 1000000000.0 +
				(double)(hist_equal_DtoH_stop.tv_sec - hist_equal_DtoH_start.tv_sec);
			printf("\033[1;34m"); //Bold blue
			printf("prefix sum MemCpy DtoH = ");
			printf("\033[1;35m"); //Bold magenta
			printf("%10g seconds\n", hist_equal_DtoH_time);

			hist_equal_findMin_time = (double)(hist_equal_findMin_stop.tv_nsec - hist_equal_findMin_start.tv_nsec) / 1000000000.0 +
				(double)(hist_equal_findMin_stop.tv_sec - hist_equal_findMin_start.tv_sec);
			printf("\033[1;34m"); //Bold blue
			printf("findMin = ");
			printf("\033[1;35m"); //Bold magenta
			printf("%10g seconds\n", hist_equal_findMin_time);

			hist_equal_result_time = (double)(hist_equal_result_stop.tv_nsec - hist_equal_result_start.tv_nsec) / 1000000000.0 +
				(double)(hist_equal_result_stop.tv_sec - hist_equal_result_start.tv_sec);
			printf("\033[1;34m"); //Bold blue
			printf("result = ");
			printf("\033[1;35m"); //Bold magenta
			printf("%10g seconds\n", hist_equal_result_time);

			hist_equal_time = hist_equal_HtoD_time*1000 + prefix_elapsed + hist_equal_DtoH_time*1000 + hist_equal_findMin_time*1000 + hist_equal_result_time*1000;
			printf("\033[1;34m"); //Bold blue
			printf("Total time = ");
			printf("\033[1;35m"); //Bold magenta
			printf("%10g ms\n", hist_equal_time);
			
		//Total Time
			printf("\033[1;34m"); //Bold blue
			printf("Total Time = ");
			printf("\033[1;33m"); //Bold Yellow
			printf("%10g ms\n", histogram_time*1000 + hist_equal_time);
		printf("\033[0m");
	//__|
	printf("\033[0m");

	freeMemory(d_lut, d_outImage);
	
    return result;
}
