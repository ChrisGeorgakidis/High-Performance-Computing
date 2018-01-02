#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

#define HISTOGRAM_SIZE 256
#define BLOCK_SIZE 32

#define cudaErrorCheck()                                                                      \
{                                                                                             \
	cudaError_t error = cudaGetLastError();                                                   \
	if (error != cudaSuccess)                                                                 \
	{                                                                                         \
		printf("\033[1;31m");                                                                 \
		printf("Cuda Error Found %s:%d:  '%s'\n", __FILE__, __LINE__, cudaGetErrorString(error));          \
		printf("\033[0m");                                                                  \
		freeMemory(h_hist, d_lut, d_outImage, d_inImage);                                       \
		exit(1);																			\
	}                                                                                         \
}

__constant__ int d_hist[HISTOGRAM_SIZE];
__constant__ int d_conLut[HISTOGRAM_SIZE];

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

	//calculate lut
	result = (int)((((float)s_cdf[output*HISTOGRAM_SIZE + idx] - d_min)*255) / (imageSize - d_min) + 0.5);
	
	if (result < 0) result = 0;
	if (result > 255) result = 255;

	lut[idx] = result;
	
}

__global__ void resultCalc(unsigned char *result, unsigned char *inImage, int imageSize)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//int idx_y = blockIdx.y*blockDim.y + threadIdx.y;
	//int idx = idx_y*gridWidth + idx_x;

	// if ((idx_x < width) && (idx_y < height) && (idx < width*height))
	// 	result[idx] = (unsigned char)d_conLut[inImage[idx]];
		
	if (idx < imageSize)
		result[idx] = (unsigned char)d_conLut[inImage[idx]];
}

void freeMemory(int *h_hist, int *d_lut, unsigned char *outputImage, unsigned char *inputImage)
{
	cudaError_t err;

	if (h_hist != NULL)
	{
		err = cudaFreeHost(h_hist);
    	if (err != cudaSuccess) {
        	printf("Error during cudaFreeHost (h_hist): %s\n", cudaGetErrorString(err));
    	}
	}

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
}

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
	//counting variable
		//cpu
			//histogram
			struct timespec histogram_start, histogram_stop;
			double histogram_time;
			//histogram_equalization
			//struct timespec hist_equal_start, hist_equal_stop;
			double hist_equal_time;
			//resultCalc
			//struct timespec hist_equal_result_start, hist_equal_result_stop;
			double hist_equal_result_time;

		//histogram_equalization
			//prefix_sum
			struct timespec prefix_sum_malloc_start, prefix_sum_malloc_stop;
			struct timespec prefix_sum_HtoD_start, prefix_sum_HtoD_stop;
			struct timespec prefix_sum_DtoH_start, prefix_sum_DtoH_stop;
			double prefix_sum_malloc_time, prefix_sum_HtoD_time, prefix_sum_DtoH_time;
			cudaEvent_t prefix_start, prefix_stop;
			float prefix_elapsed;

			//findMin
			struct timespec hist_equal_findMin_start, hist_equal_findMin_stop;
			double hist_equal_findMin_time;

			//resultCalc
			struct timespec resultCalc_malloc_start, resultCalc_malloc_stop;
			struct timespec resultCalc_HtoD_start, resultCalc_HtoD_stop;
			struct timespec resultCalc_DtoH_start, resultCalc_DtoH_stop;
			double resultCalc_malloc_time, resultCalc_HtoD_time, resultCalc_DtoH_time;
			cudaEvent_t resultCalc_start, resultCalc_stop;
			float resultCalc_elapsed;	
	//__|

	PGM_IMG result;
	//It will be pinned memory so i can use it in stream
	int *h_hist = NULL;
	int min = 0;
	int i = 0;
	int imageSize = img_in.w * img_in.h;
	
	//device variables
	cudaError_t error;
	int *d_lut = NULL;
	unsigned char *d_outImage = NULL, *d_inImage = NULL;

	//create the stream
	cudaStream_t stream;
	error = cudaStreamCreate(&stream);
	if (error != cudaSuccess)
	{
		printf("\033[1;31m");
		printf("Error creating stream:   %s\n", cudaGetErrorString(error));
		printf("\033[0m");
		freeMemory(h_hist, d_lut, d_outImage, d_inImage);
		exit(1);
	}
    
    result.w = img_in.w;
	result.h = img_in.h;
	
	//Allocate pinned memory needed for the stream
		error = cudaMallocHost((void **)&result.img, result.w*result.h*sizeof(unsigned char));
		if (error != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error allocating pinned memory on host for result.img:   %s\n", cudaGetErrorString(error));
			printf("\033[0m");
			freeMemory(h_hist, d_lut, d_outImage, d_inImage);
			exit(1);
		}

		error = cudaMallocHost((void **)&h_hist, HISTOGRAM_SIZE*sizeof(int));
		if (error != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error allocating pinned memory on host for h_hist:   %s\n", cudaGetErrorString(error));
			printf("\033[0m");
			freeMemory(h_hist, d_lut, d_outImage, d_inImage);
			exit(1);
		}
	//__|


	//Allocate device (GPU) memory for prefix_sum kernel
		clock_gettime(CLOCK_MONOTONIC_RAW, &prefix_sum_malloc_start);
		error = cudaMalloc((void **)&d_lut, HISTOGRAM_SIZE * sizeof(int));
		if (error != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error allocating memory on host for d_lut:   %s\n", cudaGetErrorString(error));
			printf("\033[0m");
			freeMemory(h_hist, d_lut, d_outImage, d_inImage);
			exit(1);
		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &prefix_sum_malloc_stop);
	//__|

	
	//Allocate device (GPU) memory for resultCalc kernel
		clock_gettime(CLOCK_MONOTONIC_RAW, &resultCalc_malloc_start);
		error = cudaMalloc((void **)&d_inImage, imageSize * sizeof(unsigned char));
		if (error != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error allocating memory on host for d_inImage:   %s\n", cudaGetErrorString(error));
			printf("\033[0m");
			freeMemory(h_hist, d_lut, d_outImage, d_inImage);
			exit(1);
		}

		error = cudaMalloc((void **)&d_outImage, imageSize * sizeof(unsigned char));
		if (error != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error allocating memory on host for d_outImage:   %s\n", cudaGetErrorString(error));
			printf("\033[0m");
			freeMemory(h_hist, d_lut, d_outImage, d_inImage);
			exit(1);
		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &resultCalc_malloc_stop);
	//__|
	

	clock_gettime(CLOCK_MONOTONIC_RAW, &histogram_start);
	histogram(h_hist, img_in.img, imageSize, HISTOGRAM_SIZE);
	clock_gettime(CLOCK_MONOTONIC_RAW, &histogram_stop);

	

	//Geometry for prefix_sum_cdf kernel
	dim3 prefix_grid(1, 1);  //We can use only one block, because we only need 256 threads << 1024 threads per block.
	dim3 prefix_block(HISTOGRAM_SIZE); //1D block geometry

	//Geometry for resultCalc kernel
	// int mod_width = ((img_in.w)%BLOCK_SIZE == 0)?0:1;
	// int mod_height = ((img_in.h)%BLOCK_SIZE == 0)?0:1;
	// int grid_width = (img_in.w)/BLOCK_SIZE + mod_width;
	// int grid_height = (img_in.h)/BLOCK_SIZE + mod_height;
	// printf("grid_width = %d		grid_height = %d\n", grid_width, grid_height);
	// dim3 hist_grid(grid_width, grid_height);
	// dim3 hist_block(BLOCK_SIZE, BLOCK_SIZE);
	
	int mod = (imageSize%1024 == 0)? 0 : 1;
	dim3 hist_grid((int)(imageSize/1024 + mod));
	dim3 hist_block(1024);

	//MemCpyAsync h_hist to __constant__ memory (Host to Device)
		clock_gettime(CLOCK_MONOTONIC_RAW, &prefix_sum_HtoD_start);
		error = cudaMemcpyToSymbolAsync(d_hist, h_hist, HISTOGRAM_SIZE*sizeof(int), 0, cudaMemcpyHostToDevice, stream);
		if (error != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error during cudaMemcpyToSymbolAsync of h_hist to d_hist:  %s\n", cudaGetErrorString(error));
			printf("\033[0m");
			freeMemory(h_hist, d_lut, d_outImage, d_inImage);
			exit(1);
		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &prefix_sum_HtoD_stop);
	//__|

	clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_findMin_start);
	while(min == 0){
        min = h_hist[i++];
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_findMin_stop);

	

	cudaEventCreate(&prefix_start);
	cudaEventCreate(&prefix_stop);

	cudaEventRecord(prefix_start, 0);
	//Prefix SUM CDF kernel
		prefix_sum_cdf <<< prefix_grid, prefix_block, 0, stream >>> (d_lut, imageSize, min);
	//__|
	cudaEventRecord(prefix_stop, 0);

	//Error Checking
	cudaErrorCheck();

	//MemCpyToSymbolAsync Device to Device so that it will be ready for the net kernel launch
		clock_gettime(CLOCK_MONOTONIC_RAW, &prefix_sum_DtoH_start);
		error = cudaMemcpyToSymbolAsync(d_conLut, d_lut, HISTOGRAM_SIZE*sizeof(int), 0, cudaMemcpyDeviceToDevice, stream);
		if (error != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error during cudaMemcpyToSymbolAsync of d_lut to d_conLut:  %s\n", cudaGetErrorString(error));
			printf("\033[0m");
			freeMemory(h_hist, d_lut, d_outImage, d_inImage);
			exit(1);
		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &prefix_sum_DtoH_stop);
	//__|

	//MemCpy Host to Device for resultCalc
		clock_gettime(CLOCK_MONOTONIC_RAW, &resultCalc_HtoD_start);
		error = cudaMemcpyAsync(d_inImage, img_in.img, imageSize*sizeof(unsigned char), cudaMemcpyHostToDevice, stream);
		if (error != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error during cudaMemcpyAsync of img_in.img to d_inImage:  %s\n", cudaGetErrorString(error));
			printf("\033[0m");
			freeMemory(h_hist, d_lut, d_outImage, d_inImage);
			exit(1);

		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &resultCalc_HtoD_stop);
	//__|	

	cudaEventCreate(&resultCalc_start);
	cudaEventCreate(&resultCalc_stop);

	cudaEventRecord(resultCalc_start, 0);
	//Result Calculation in GPU
		resultCalc <<< hist_grid, hist_block, 0, stream >>> (d_outImage, d_inImage, imageSize);
	//__|
	cudaEventRecord(resultCalc_stop, 0);

	//Error Checking
	cudaErrorCheck();

	clock_gettime(CLOCK_MONOTONIC_RAW, &resultCalc_DtoH_start);
	//MemCpy Device to Host
		error = cudaMemcpyAsync(result.img, d_outImage, imageSize*sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
		if (error != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error during cudaMemcpyAsync of d_outImage to result.img:  %s\n", cudaGetErrorString(error));
			printf("\033[0m");
			freeMemory(h_hist, d_lut, d_outImage, d_inImage);
			exit(1);
		}
	//__|
	clock_gettime(CLOCK_MONOTONIC_RAW, &resultCalc_DtoH_stop);
	
	
	// clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_result_start);
	// /* Get the result image */
    // for(i = 0; i < imageSize; i ++){
    //     result.img[i] = (unsigned char)h_lut[img_in.img[i]];        
	// }
	// clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_result_stop);

	//Error Checking
	cudaErrorCheck();

	// clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_start);
	// histogram_equalization(result.img,img_in.img, h_hist,result.w*result.h, HISTOGRAM_SIZE);
	// clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_stop);
	

	error = cudaStreamDestroy(stream);
	if (error != cudaSuccess)
	{
		printf("\033[1;31m");
		printf("Error during cudaStreamDestroy:  %s\n", cudaGetErrorString(error));
		printf("\033[0m");
		freeMemory(h_hist, d_lut, d_outImage, d_inImage);
		exit(1);
	}

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
			prefix_sum_malloc_time = (double)(prefix_sum_malloc_stop.tv_nsec - prefix_sum_malloc_start.tv_nsec) / 1000000000.0 +
				(double)(prefix_sum_malloc_stop.tv_sec - prefix_sum_malloc_start.tv_sec);
			printf("\033[1;34m"); //Bold blue
			printf("prefix_sum malloc = ");
			printf("\033[1;35m"); //Bold magenta
			printf("%10g seconds\n", prefix_sum_malloc_time);

			prefix_sum_HtoD_time = (double)(prefix_sum_HtoD_stop.tv_nsec - prefix_sum_HtoD_start.tv_nsec) / 1000000000.0 +
				(double)(prefix_sum_HtoD_stop.tv_sec - prefix_sum_HtoD_start.tv_sec);
			printf("\033[1;34m"); //Bold blue
			printf("prefix sum MemCpy HtoD = ");
			printf("\033[1;35m"); //Bold magenta
			printf("%10g seconds\n", prefix_sum_HtoD_time);

			cudaEventSynchronize(prefix_stop);
			cudaEventElapsedTime(&prefix_elapsed, prefix_start, prefix_stop);
			printf("\033[1;34m");
			printf("prefix_sum_cdf = ");
			printf("\033[1;35m");
			printf("%g ms\n", prefix_elapsed);

			prefix_sum_DtoH_time = (double)(prefix_sum_DtoH_stop.tv_nsec - prefix_sum_DtoH_start.tv_nsec) / 1000000000.0 +
				(double)(prefix_sum_DtoH_stop.tv_sec - prefix_sum_DtoH_start.tv_sec);
			printf("\033[1;34m"); //Bold blue
			printf("prefix sum MemCpy DtoH = ");
			printf("\033[1;35m"); //Bold magenta
			printf("%10g seconds\n", prefix_sum_DtoH_time);

			hist_equal_findMin_time = (double)(hist_equal_findMin_stop.tv_nsec - hist_equal_findMin_start.tv_nsec) / 1000000000.0 +
				(double)(hist_equal_findMin_stop.tv_sec - hist_equal_findMin_start.tv_sec);
			printf("\033[1;34m"); //Bold blue
			printf("findMin = ");
			printf("\033[1;35m"); //Bold magenta
			printf("%10g seconds\n", hist_equal_findMin_time);

			// hist_equal_result_time = (double)(hist_equal_result_stop.tv_nsec - hist_equal_result_start.tv_nsec) / 1000000000.0 +
			// 	(double)(hist_equal_result_stop.tv_sec - hist_equal_result_start.tv_sec);
			// printf("\033[1;34m"); //Bold blue
			// printf("result = ");
			// printf("\033[1;35m"); //Bold magenta
			// printf("%10g seconds\n", hist_equal_result_time);

			//resultCalc gpu
			resultCalc_malloc_time = (double)(resultCalc_malloc_stop.tv_nsec - resultCalc_malloc_start.tv_nsec) / 1000000000.0 +
				(double)(resultCalc_malloc_stop.tv_sec - resultCalc_malloc_start.tv_sec);
			printf("\033[1;34m"); //Bold blue
			printf("resultCalc malloc = ");
			printf("\033[1;35m"); //Bold magenta
			printf("%10g seconds\n", resultCalc_malloc_time);

			resultCalc_HtoD_time = (double)(resultCalc_HtoD_stop.tv_nsec - resultCalc_HtoD_start.tv_nsec) / 1000000000.0 +
				(double)(resultCalc_HtoD_stop.tv_sec - resultCalc_HtoD_start.tv_sec);
			printf("\033[1;34m"); //Bold blue
			printf("resultCalc MemCpy HtoD = ");
			printf("\033[1;35m"); //Bold magenta
			printf("%10g seconds\n", resultCalc_HtoD_time);

			cudaEventSynchronize(resultCalc_stop);
			cudaEventElapsedTime(&resultCalc_elapsed, resultCalc_start, resultCalc_stop);
			printf("\033[1;34m");
			printf("resultCalc = ");
			printf("\033[1;35m");
			printf("%g ms\n", resultCalc_elapsed);

			resultCalc_DtoH_time = (double)(resultCalc_DtoH_stop.tv_nsec - resultCalc_DtoH_start.tv_nsec) / 1000000000.0 +
				(double)(resultCalc_DtoH_stop.tv_sec - resultCalc_DtoH_start.tv_sec);
			printf("\033[1;34m"); //Bold blue
			printf("resultCalc MemCpy DtoH = ");
			printf("\033[1;35m"); //Bold magenta
			printf("%10g seconds\n", resultCalc_DtoH_time);

			hist_equal_result_time = resultCalc_HtoD_time*1000 + resultCalc_elapsed + resultCalc_DtoH_time*1000;
			printf("\033[1;34m"); //Bold blue
			printf("Total resultCalc time = ");
			printf("\033[1;35m"); //Bold magenta
			printf("%10g ms\n", hist_equal_result_time);


			hist_equal_time = prefix_sum_HtoD_time*1000 + prefix_elapsed + prefix_sum_DtoH_time*1000 + hist_equal_findMin_time*1000 + hist_equal_result_time;
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

	freeMemory(h_hist, d_lut, d_outImage, d_inImage);
	
    return result;
}
