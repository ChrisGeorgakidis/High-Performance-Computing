#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
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
		freeMemory(inputImage, g_hist, d_lut, d_cdf, outImage);                                       \
	}                                                                                         \
}

__constant__ int d_hist[HISTOGRAM_SIZE];
__constant__ int lut_con[HISTOGRAM_SIZE];

//TODO: Το παρακάτω δεν δουλεύει σωστά! Να διορθωθεί...
__global__ void histogram_gpu_image(int *hist, unsigned char *inImage, int imageW, int imageH, int gridWidth)
{
	int idx_x = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y*blockDim.y +threadIdx.y;
	//int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idx = idx_y*gridWidth + idx_x;
	int bl_idx = inImage[idx]; 
	//int imageSize = imageW*imageH;
	__shared__ int s_hist[HISTOGRAM_SIZE];
	
	//Initialise shared memory. Since shared memory is private for each
	//block only the threads of each block that their indices are < HISTOGRAM_SIZE
	//need to initialise the shared memory.
	if (bl_idx < HISTOGRAM_SIZE) //bl_idx < 256
	{
		s_hist[bl_idx] = 0;
	}
	__syncthreads(); //wait until all threads are done with initialisation

	//Generate the histogram
	//TODO: illegal memory access 
	if (idx < imageW*imageH) //because there are extra threads since the image size is not only ^2
	{
		//The addition must be atomic because multiple threads
		//may want to increase by one the same index of s_hist.
		atomicAdd(&s_hist[bl_idx], 1);   
	}

	__syncthreads(); //wait until all threeads are done with increament

	//Now copy the result back to global memory g_hist.
	if (bl_idx < HISTOGRAM_SIZE)
	{
		//This addition must be atomic because bl_idx is unique only inside the block.
		//So threads with the same bl_idx, will go to add their result to the same idx of g_hist at the same time.
		atomicAdd(&hist[bl_idx], s_hist[bl_idx]);
	}
}

__global__ void histogram_gpu_hist(int *hist, unsigned char *inImage, int imageSize)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i = 0;

	__shared__ int s_hist[HISTOGRAM_SIZE];
	
	s_hist[idx] = 0;

	#pragma unroll
	for(; i < imageSize; i++)
	{
		if (idx == inImage[i])
			s_hist[idx]++;
	}

	hist[idx] = s_hist[idx];
}

__global__ void prefix_sum_cdf(int *lut, int *cdf, int img_size, int min)
{
	__shared__ int s_cdf[512]; //double buffer: 2 rows of 256 indices each
	int imageSize = img_size;
	int d_min = min;
	int idx = threadIdx.x;
	int output = 0, input = 1;
	int d = 1; //offset

	//Load hist into shared memory
	s_cdf[idx] = d_hist[idx];
	__syncthreads();

	//calculate cdf using prefix sum technic
	for ( ; d < 256; d = d << 1)
	{
		output = 1 - output;
		input = 1 - input;

		if (idx >= d)
		{
			s_cdf[output*256 + idx] = s_cdf[input*256 + idx] + s_cdf[input*256 + idx - d];
		}
		else
		{
			s_cdf[output*256 + idx] = s_cdf[input*256 + idx];
		}
		__syncthreads();
	}
	//write shared memory result into global memory output
	cdf[idx] = s_cdf[output*256 + idx];
	lut[idx] = (int)((((float)s_cdf[output*256 + idx] - d_min)*255) / (imageSize - d_min) + 0.5);
	
	if (lut[idx] < 0) lut[idx] = 0;
	if (lut[idx] > 255) lut[idx] = 255;
	
}

__global__ void resultCalc(unsigned char *result, unsigned char *inImage, int gridWidth)
{
	int idx_x = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y*blockDim.y +threadIdx.y;

	result[idx_y*gridWidth + idx_x] = lut_con[inImage[idx_y*gridWidth + idx_x]];
}

void freeMemory(unsigned char *inputImage, int *g_hist, int *d_lut, int *d_cdf, unsigned char *outputImage)
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

	if (inputImage != NULL)
	{
		err = cudaFree(inputImage);
		if (err != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error during cudaFree (inputImage):  %s\n", cudaGetErrorString(err));
			printf("\033[0m");
		}
	}

	if (g_hist != NULL)
	{
		err = cudaFree(g_hist);
		if (err != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error during cudaFree (g_hist):  %s\n", cudaGetErrorString(err));
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

	if (d_cdf != NULL)
	{
		err = cudaFree(d_cdf);
		if (err != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error during cudaFree (d_cdf):  %s\n", cudaGetErrorString(err));
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
	//counting time variables
		//malloc timing variables
			struct timespec histogram_time_malloc1, histogram_time_malloc2;
			struct timespec prefix_time_malloc1, prefix_time_malloc2;
			struct timespec resultCalc_time_malloc1, resultCalc_time_malloc2;
		//~~~~~~~//
		//kernel & cpu execution timing variables
			cudaEvent_t histogram_image_start, histogram_image_stop;
			cudaEvent_t histogram_hist_start, histogram_hist_stop;
			cudaEvent_t prefix_start, prefix_stop;
			cudaEvent_t resultCalc_start, resultCalc_stop;
			float histogram_elapsed, prefix_elapsed, resultCalc_elapsed;
			struct timespec minFind_t1, minFind_t2;
			struct timespec histogram_cpu_start, histogram_cpu_stop;
			struct timespec histogram_equalization_start, histogram_equalization_stop;
			struct timespec clipping_start, clipping_stop;
		//~~~~~~~//
		//MemCpy timing variables
			struct timespec histogram_time_HtoD1, histogram_time_HtoD2;
			struct timespec histogram_time_DtoH1, histogram_time_DtoH2;
			struct timespec prefix_time_HtoD1, prefix_time_HtoD2;
			struct timespec prefix_time_DtoH1, prefix_time_DtoH2;
			struct timespec resultCalc_time_HtoD1, resultCalc_time_HtoD2;
			struct timespec resultCalc_time_DtoH1, resultCalc_time_DtoH2;
		//~~~~~~~//
	//__|
	
	//host variables
		PGM_IMG result;
		int h_hist[256];
		int *h_lut = (int *)malloc(256 * sizeof(int));
	//__|
	
	//device variables
		cudaError_t error;
		int *g_hist = NULL, *d_lut = NULL, *d_cdf = NULL;
		unsigned char *inputImage = NULL;
		unsigned char *outImage = NULL;
	//__|
	
	//constant variables
		int imageSize = img_in.w * img_in.h;
		int min = 0;
		int i = 0;
	//__|

	result.w = img_in.w;
	result.h = img_in.h;
	result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
	
	//Memory Allocations
		clock_gettime(CLOCK_MONOTONIC_RAW, &histogram_time_malloc1);
		//Allocate device (GPU) memory for histogram_gpu kernel//
			error = cudaMalloc((void **)&inputImage, imageSize * sizeof(unsigned char));
			if (error != cudaSuccess)
			{
				printf("\033[1;31m");
				printf("Error allocating memory on host for inputImage:   %s\n", cudaGetErrorString(error));
				printf("\033[0m");
				freeMemory(inputImage, g_hist, d_lut, d_cdf, outImage);
				return (result);
			}

			error = cudaMalloc((void **)&g_hist, HISTOGRAM_SIZE * sizeof(int));
			if (error != cudaSuccess)
			{
				printf("\033[1;31m");
				printf("Error allocating memory on host for g_hist:   %s\n", cudaGetErrorString(error));
				printf("\033[0m");
				freeMemory(inputImage, g_hist, d_lut, d_cdf, outImage);
				return (result);
			}
			//Init g_hist with zeros
			cudaMemset(g_hist, 0.0, HISTOGRAM_SIZE *sizeof(int)); 
		//-----------------------------------------------------//
		clock_gettime(CLOCK_MONOTONIC_RAW, &histogram_time_malloc2);

		clock_gettime(CLOCK_MONOTONIC_RAW, &prefix_time_malloc1);
		//Allocate device (GPU) memory for prefix_sum_cdf kernel//
			error = cudaMalloc((void **)&d_lut, HISTOGRAM_SIZE * sizeof(int));
			if (error != cudaSuccess)
			{
				printf("\033[1;31m");
				printf("Error allocating memory on host for d_lut:   %s\n", cudaGetErrorString(error));
				printf("\033[0m");
				freeMemory(inputImage, g_hist, d_lut, d_cdf, outImage);
				return (result);
			}

			error = cudaMalloc((void **)&d_cdf, HISTOGRAM_SIZE * sizeof(int));
			if (error != cudaSuccess)
			{
				printf("\033[1;31m");
				printf("Error allocating memory on host for d_cdf:   %s\n", cudaGetErrorString(error));
				printf("\033[0m");
				freeMemory(inputImage, g_hist, d_lut, d_cdf, outImage);
				return (result);
			}
		//------------------------------------------------------//
		clock_gettime(CLOCK_MONOTONIC_RAW, &prefix_time_malloc2);

		clock_gettime(CLOCK_MONOTONIC_RAW, &resultCalc_time_malloc1);
		//Allocate device (GPU) memory for resultCalc kernel//
			error = cudaMalloc((void **)&outImage, imageSize * sizeof(unsigned char));
			if (error != cudaSuccess)
			{
				printf("\033[1;31m");
				printf("Error allocating memory on host for outImage:   %s\n", cudaGetErrorString(error));
				printf("\033[0m");
				freeMemory(inputImage, g_hist, d_lut, d_cdf, outImage);
				return (result);
			}
		//--------------------------------------------------//
		clock_gettime(CLOCK_MONOTONIC_RAW, &resultCalc_time_malloc2);
	//__|
	
	clock_gettime(CLOCK_MONOTONIC_RAW, &histogram_time_HtoD1);
	//MemCpy Host to Device
		error = cudaMemcpy(inputImage, img_in.img, imageSize*sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error during cudaMemcpy of img_in.img to inputImage:  %s\n", cudaGetErrorString(error));
			printf("\033[0m");
			freeMemory(inputImage, g_hist, d_lut, d_cdf, outImage);
			return (result);
	
		}
	//__|
	clock_gettime(CLOCK_MONOTONIC_RAW, &histogram_time_HtoD2);

	
	//Initialise geometries for kernels
	
		//-->Geometry for histogram_gpu kernel
		int grid_width = (img_in.w - 1)/BLOCK_SIZE + 1; //
		int grid_height = (img_in.h - 1)/BLOCK_SIZE + 1;
		dim3 hist_grid(grid_width, grid_height);
		dim3 hist_block(BLOCK_SIZE, BLOCK_SIZE);

		//-->Geometry for prefix_sum_cdf kernel
		dim3 prefix_grid(1, 1);  //We can use only one block, because we only need 256 threads << 1024 threads per block.
		dim3 prefix_block(HISTOGRAM_SIZE); //1D block geometry
	
	//__|

	cudaEventCreate(&histogram_image_start);
	cudaEventCreate(&histogram_image_stop);
	cudaEventCreate(&histogram_hist_start);
	cudaEventCreate(&histogram_hist_stop);

	cudaEventRecord(histogram_image_start, 0);
	//Image implementation of histogram calculation
		histogram_gpu_image <<< hist_grid, hist_block >>> (g_hist, inputImage, img_in.w, img_in.h, grid_width);
	//__|
	cudaEventRecord(histogram_image_stop, 0);
	
	cudaEventSynchronize(histogram_image_stop);

	cudaEventRecord(histogram_hist_start, 0);
	//Histogram implementation of histogram calculation
		histogram_gpu_hist <<< prefix_grid, prefix_block >>> (g_hist, inputImage, imageSize);
	//__|
	cudaEventRecord(histogram_hist_stop, 0);
	
	cudaEventSynchronize(histogram_hist_stop); 

	//Error Checking
	cudaErrorCheck();
	
	//Now h_hist has the correct values, so it can be copied to constant memory
	clock_gettime(CLOCK_MONOTONIC_RAW, &histogram_time_DtoH1);
	//MemCpy Device to Host
		error = cudaMemcpy(h_hist, g_hist, HISTOGRAM_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error during cudaMemcpy of g_hist to h_hist:  %s\n", cudaGetErrorString(error));
			printf("\033[0m");
			freeMemory(inputImage, g_hist, d_lut, d_cdf, outImage);
			return (result);
		}
	//__|
	clock_gettime(CLOCK_MONOTONIC_RAW, &histogram_time_DtoH2);

	clock_gettime(CLOCK_MONOTONIC_RAW, &histogram_cpu_start);
	//CPU Execution
	histogram(h_hist, img_in.img, img_in.h * img_in.w, HISTOGRAM_SIZE);
	clock_gettime(CLOCK_MONOTONIC_RAW, &histogram_cpu_stop);

	clock_gettime(CLOCK_MONOTONIC_RAW, &minFind_t1);
	//Find first non-zero value in h_hist
		while(min == 0){
			min = h_hist[i++];
		}
	//__|
	clock_gettime(CLOCK_MONOTONIC_RAW, &minFind_t2);

	clock_gettime(CLOCK_MONOTONIC_RAW, &prefix_time_HtoD1);
	//MemCpy to __constant__ memory (Host to Device)
		error = cudaMemcpyToSymbol(d_hist, h_hist, HISTOGRAM_SIZE * sizeof(int));
		if (error != cudaSuccess)
		{
			printf("\033[1;31m");      
			printf("Error during cudaMemcpyToSymbol of h_hist to d_hist:  %s\n", cudaGetErrorString(error));
			printf("\033[0m");
			freeMemory(inputImage, g_hist, d_lut, d_cdf, outImage);
			return (result);
		}
	//__|
	clock_gettime(CLOCK_MONOTONIC_RAW, &prefix_time_HtoD2);

	cudaEventCreate(&prefix_start);
	cudaEventCreate(&prefix_stop);
	
	cudaEventRecord(prefix_start, 0);
	//Prefix SUM CDF kernel
		prefix_sum_cdf <<< prefix_grid, prefix_block >>> (d_lut, d_cdf, imageSize, min);
	//__|
	cudaEventRecord(prefix_stop, 0);
	
	//Error Checking
	cudaErrorCheck();

	clock_gettime(CLOCK_MONOTONIC_RAW, &prefix_time_DtoH1);
	//MemCpy Device to Host
		error = cudaMemcpy(h_lut, d_lut, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error during cudaMemcpy of d_lut to h_lut:  %s\n", cudaGetErrorString(error));
			printf("\033[0m");
			freeMemory(inputImage, g_hist, d_lut, d_cdf, outImage);
			return (result);
		}
	//__|
	clock_gettime(CLOCK_MONOTONIC_RAW, &prefix_time_DtoH2);
	
	clock_gettime(CLOCK_MONOTONIC_RAW, &resultCalc_time_HtoD1);
	//MemCpy lut to __constant__ memory (Host to Device)
		error = cudaMemcpyToSymbol(lut_con, h_lut, HISTOGRAM_SIZE * sizeof(int));
		if (error != cudaSuccess)
		{
			printf("\033[1;31m");      
			printf("Error during cudaMemcpyToSymbol of h_lut to lut_con:  %s\n", cudaGetErrorString(error));
			printf("\033[0m");
			freeMemory(inputImage, g_hist, d_lut, d_cdf, outImage);
			return (result);
		}
	//__|
	clock_gettime(CLOCK_MONOTONIC_RAW, &resultCalc_time_HtoD2);

	cudaEventCreate(&resultCalc_start);
	cudaEventCreate(&resultCalc_stop);

	cudaEventRecord(resultCalc_start, 0);
	//Result Calculation in GPU
		resultCalc <<< hist_grid, hist_block >>> (outImage, inputImage, grid_width);
	//__|
	cudaEventRecord(resultCalc_stop, 0);

	//Error Checking
	cudaErrorCheck();

	clock_gettime(CLOCK_MONOTONIC_RAW, &resultCalc_time_DtoH1);
	//MemCpy Device to Host
		error = cudaMemcpy(result.img, outImage, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("\033[1;31m");
			printf("Error during cudaMemcpy of outImage to result.img:  %s\n", cudaGetErrorString(error));
			printf("\033[0m");
			freeMemory(inputImage, g_hist, d_lut, d_cdf, outImage);
			return (result);
		}
	//__|
	clock_gettime(CLOCK_MONOTONIC_RAW, &resultCalc_time_DtoH2);

	clock_gettime(CLOCK_MONOTONIC_RAW, &clipping_start);
	#pragma unroll
	for (i = 0; i < imageSize; i++)
	{
		result.img[i] = (unsigned char)h_lut[img_in.img[i]];
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &clipping_stop);

	clock_gettime(CLOCK_MONOTONIC_RAW, &histogram_equalization_start);
	//CPU Execution
	histogram_equalization(result.img,img_in.img,h_hist,result.w*result.h, HISTOGRAM_SIZE);
	clock_gettime(CLOCK_MONOTONIC_RAW, &histogram_equalization_stop);

	//Calculate the durations of memory allocations, kernel executions & MemCpy
		//Time for Memory Allocations
			printf("Memory Allocation Times\n");
			//histogram_gpu
			printf("\033[1;34m");
			printf("histogram_gpu = ");
			printf("\033[1;32m");
			printf("%10g seconds\n", (double)(histogram_time_malloc2.tv_nsec - histogram_time_malloc1.tv_nsec) / 1000000000.0 +
				(double)(histogram_time_malloc2.tv_sec - histogram_time_malloc1.tv_sec));
			//prefix_sum_cdf
			printf("\033[1;34m");
			printf("prefix_sum_cdf = ");
			printf("\033[1;35m");
			printf("%10g seconds\n", (double)(prefix_time_malloc2.tv_nsec - prefix_time_malloc1.tv_nsec) / 1000000000.0 +
				(double)(prefix_time_malloc2.tv_sec - prefix_time_malloc1.tv_sec));
			//resultCalc
			printf("\033[1;34m");
			printf("resultCalc = ");
			printf("\033[0m");
			printf("%10g seconds\n", (double)(resultCalc_time_malloc2.tv_nsec - resultCalc_time_malloc1.tv_nsec) / 1000000000.0 +
				(double)(resultCalc_time_malloc2.tv_sec - resultCalc_time_malloc1.tv_sec));
			//Total
			printf("\033[1;34m");
			printf("Total memory allocation time = ");
			printf("\033[1;33m");
			printf("%10g seconds\n", 
					(double)(histogram_time_malloc2.tv_nsec - histogram_time_malloc1.tv_nsec) / 1000000000.0 +
					(double)(histogram_time_malloc2.tv_sec - histogram_time_malloc1.tv_sec) + 
					(double)(prefix_time_malloc2.tv_nsec - prefix_time_malloc1.tv_nsec) / 1000000000.0 +
					(double)(prefix_time_malloc2.tv_sec - prefix_time_malloc1.tv_sec));
			printf("\033[0m");
		//__|
		
		//Time for kernel executions
			printf("Kernel Execution Times\n");
			//histogram_gpu_image
			cudaEventElapsedTime(&histogram_elapsed, histogram_image_start, histogram_image_stop);
			printf("\033[1;34m");
			printf("histogram_gpu_image = ");
			printf("\033[1;32m");
			printf("%g ms\n", histogram_elapsed);
			//histogram_gpu_hist
			cudaEventElapsedTime(&histogram_elapsed, histogram_hist_start, histogram_hist_stop);
			printf("\033[1;34m");
			printf("histogram_gpu_hist = ");
			printf("\033[1;36m");
			printf("%g ms\n", histogram_elapsed);
			//histogram_cpu
			printf("\033[1;34m");
			printf("histogram_cpu = ");
			printf("\033[1;31m");
			printf("%10g seconds\n", (double)(histogram_cpu_stop.tv_nsec - histogram_cpu_start.tv_nsec) / 1000000000.0 +
				(double)(histogram_cpu_stop.tv_sec - histogram_cpu_start.tv_sec));
			//minFind
			printf("\033[1;34m");
			printf("minFind = ");
			printf("\033[1;31m");
			printf("%10g seconds\n", (double)(minFind_t2.tv_nsec - minFind_t1.tv_nsec) / 1000000000.0 +
				(double)(minFind_t2.tv_sec - minFind_t1.tv_sec));
			//prefix_sum_cdf
			cudaEventElapsedTime(&prefix_elapsed, prefix_start, prefix_stop);
			printf("\033[1;34m");
			printf("prefix_sum_cdf = ");
			printf("\033[1;35m");
			printf("%g ms\n", prefix_elapsed);
			//histogram_equalization_cpu
			printf("\033[1;34m");
			printf("histogram_equalization (cpu) = ");
			printf("\033[1;31m");
			printf("%10g seconds\n", (double)(histogram_equalization_stop.tv_nsec - histogram_equalization_start.tv_nsec) / 1000000000.0 +
				(double)(histogram_equalization_stop.tv_sec - histogram_equalization_start.tv_sec));
			//resultCalc
			cudaEventElapsedTime(&resultCalc_elapsed, resultCalc_start, resultCalc_stop);
			printf("\033[1;34m");
			printf("resultCalc = ");
			printf("\033[0m");
			printf("%g ms\n", resultCalc_elapsed);
			//clipping
			printf("\033[1;34m");
			printf("clipping (cpu) = ");
			printf("\033[1;31m");
			printf("%10g seconds\n", (double)(clipping_stop.tv_nsec - clipping_start.tv_nsec) / 1000000000.0 +
				(double)(clipping_stop.tv_sec - clipping_start.tv_sec));
			//Total
			printf("\033[1;34m");
			printf("Total histogram equalization time = ");
			printf("\033[1;33m");
			printf("%g seconds\n", 
				prefix_elapsed*0.001 + 
				(double)(minFind_t2.tv_nsec - minFind_t1.tv_nsec) / 1000000000.0 + (double)(minFind_t2.tv_sec - minFind_t1.tv_sec) + 
				resultCalc_elapsed*0.001);
			printf("\033[0m");
			//Total
			printf("\033[1;34m");
			printf("Total kernel execution time = ");
			printf("\033[1;33m");
			printf("%g ms\n", histogram_elapsed + prefix_elapsed);
			printf("\033[0m");
		//__|

		//Time for MemCpy HtoD
			printf("MemCpy HtoD Times\n");
			//histogram_gpu
			printf("\033[1;34m");
			printf("histogram_gpu = ");
			printf("\033[1;32m");
			printf("%10g seconds\n", (double)(histogram_time_HtoD2.tv_nsec - histogram_time_HtoD1.tv_nsec) / 1000000000.0 +
				(double)(histogram_time_HtoD2.tv_sec - histogram_time_HtoD1.tv_sec));
			//prefix_sum_cdf
			printf("\033[1;34m");
			printf("prefix_sum_cdf = ");
			printf("\033[1;35m");
			printf("%10g seconds\n", (double)(prefix_time_HtoD2.tv_nsec - prefix_time_HtoD1.tv_nsec) / 1000000000.0 +
				(double)(prefix_time_HtoD2.tv_sec - prefix_time_HtoD1.tv_sec));
			//resultCalc
			printf("\033[1;34m");
			printf("resultCalc = ");
			printf("\033[0m");
			printf("%10g seconds\n", (double)(resultCalc_time_HtoD2.tv_nsec - resultCalc_time_HtoD1.tv_nsec) / 1000000000.0 +
				(double)(resultCalc_time_HtoD2.tv_sec - resultCalc_time_HtoD1.tv_sec));
			//Total
			printf("\033[1;34m");
			printf("Total MemCpy HtoD time = ");
			printf("\033[1;33m");
			printf("%10g seconds\n", 
					(double)(histogram_time_HtoD2.tv_nsec - histogram_time_HtoD1.tv_nsec) / 1000000000.0 +
					(double)(histogram_time_HtoD2.tv_sec - histogram_time_HtoD1.tv_sec) + 
					(double)(prefix_time_HtoD2.tv_nsec - prefix_time_HtoD1.tv_nsec) / 1000000000.0 +
					(double)(prefix_time_HtoD2.tv_sec - prefix_time_HtoD1.tv_sec));
			printf("\033[0m");
		//__|
			
		//Time for MemCpy DtoH
			printf("MemCpy DtoH Times\n");
			//histogram_gpu
			printf("\033[1;34m");
			printf("histogram_gpu = ");
			printf("\033[1;32m");
			printf("%10g seconds\n", (double)(histogram_time_DtoH2.tv_nsec - histogram_time_DtoH1.tv_nsec) / 1000000000.0 +
				(double)(histogram_time_DtoH2.tv_sec - histogram_time_DtoH1.tv_sec));
			//prefix_sum_cdf
			printf("\033[1;34m");
			printf("prefix_sum_cdf = ");
			printf("\033[1;35m");
			printf("%10g seconds\n", (double)(prefix_time_DtoH2.tv_nsec - prefix_time_DtoH1.tv_nsec) / 1000000000.0 +
				(double)(prefix_time_DtoH2.tv_sec - prefix_time_DtoH1.tv_sec));
			//resultCalc
			printf("\033[1;34m");
			printf("resultCalc = ");
			printf("\033[0m");
			printf("%10g seconds\n", (double)(resultCalc_time_DtoH2.tv_nsec - resultCalc_time_DtoH1.tv_nsec) / 1000000000.0 +
				(double)(resultCalc_time_DtoH2.tv_sec - resultCalc_time_DtoH1.tv_sec));
			//Total
			printf("\033[1;34m");
			printf("Total MemCpy DtoH time = ");
			printf("\033[1;33m");
			printf("%10g seconds\n", 
					(double)(histogram_time_DtoH2.tv_nsec - histogram_time_DtoH1.tv_nsec) / 1000000000.0 +
					(double)(histogram_time_DtoH2.tv_sec - histogram_time_DtoH1.tv_sec) + 
					(double)(prefix_time_DtoH2.tv_nsec - prefix_time_DtoH1.tv_nsec) / 1000000000.0 +
					(double)(prefix_time_DtoH2.tv_sec - prefix_time_DtoH1.tv_sec));
			printf("\033[0m");
		//__|

		//Time for MemCpy
			printf("Total MemCpy Times\n");
			//histogram_gpu
			printf("\033[1;34m");
			printf("histogram_gpu = ");
			printf("\033[1;32m");
			printf("%10g seconds\n", 
					(double)(histogram_time_HtoD2.tv_nsec - histogram_time_HtoD1.tv_nsec) / 1000000000.0 +
					(double)(histogram_time_HtoD2.tv_sec - histogram_time_HtoD1.tv_sec) + 
					(double)(histogram_time_DtoH2.tv_nsec - histogram_time_DtoH1.tv_nsec) / 1000000000.0 +
					(double)(histogram_time_DtoH2.tv_sec - histogram_time_DtoH1.tv_sec));
			//prefix_sum_cdf
			printf("\033[1;34m");
			printf("prefix_sum_cdf = ");
			printf("\033[1;35m");
			printf("%10g seconds\n", 
					(double)(prefix_time_HtoD2.tv_nsec - prefix_time_HtoD1.tv_nsec) / 1000000000.0 +
					(double)(prefix_time_HtoD2.tv_sec - prefix_time_HtoD1.tv_sec) + 
					(double)(prefix_time_DtoH2.tv_nsec - prefix_time_DtoH1.tv_nsec) / 1000000000.0 +
					(double)(prefix_time_DtoH2.tv_sec - prefix_time_DtoH1.tv_sec));
			//resultCalc
			printf("\033[1;34m");
			printf("resultCalc = ");
			printf("\033[0m");
			printf("%10g seconds\n", 
					(double)(resultCalc_time_HtoD2.tv_nsec - resultCalc_time_HtoD1.tv_nsec) / 1000000000.0 +
					(double)(resultCalc_time_HtoD2.tv_sec - resultCalc_time_HtoD1.tv_sec) + 
					(double)(resultCalc_time_DtoH2.tv_nsec - resultCalc_time_DtoH1.tv_nsec) / 1000000000.0 +
					(double)(resultCalc_time_DtoH2.tv_sec - resultCalc_time_DtoH1.tv_sec));
			//Total
			printf("\033[1;34m");
			printf("Total MemCpy time = ");
			printf("\033[1;33m");
			printf("%10g seconds\n", 
					(double)(histogram_time_HtoD2.tv_nsec - histogram_time_HtoD1.tv_nsec) / 1000000000.0 +
					(double)(histogram_time_HtoD2.tv_sec - histogram_time_HtoD1.tv_sec) + 
					(double)(histogram_time_DtoH2.tv_nsec - histogram_time_DtoH1.tv_nsec) / 1000000000.0 +
					(double)(histogram_time_DtoH2.tv_sec - histogram_time_DtoH1.tv_sec) +
					(double)(prefix_time_HtoD2.tv_nsec - prefix_time_HtoD1.tv_nsec) / 1000000000.0 +
					(double)(prefix_time_HtoD2.tv_sec - prefix_time_HtoD1.tv_sec) + 
					(double)(prefix_time_DtoH2.tv_nsec - prefix_time_DtoH1.tv_nsec) / 1000000000.0 +
					(double)(prefix_time_DtoH2.tv_sec - prefix_time_DtoH1.tv_sec));
			printf("\033[0m");
		//__|
	//__| 
	printf("\033[0m");
	
	freeMemory(inputImage, g_hist, d_lut, d_cdf, outImage);
	return result;
}
