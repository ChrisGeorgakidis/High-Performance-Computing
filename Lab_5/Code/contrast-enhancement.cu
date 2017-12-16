#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

#define cudaErrorCheck()                                                                      \
{                                                                                             \
    cudaError_t error = cudaGetLastError();                                                   \
    if (error != cudaSuccess)                                                                 \
    {                                                                                         \
        printf("Cuda Error Found %s:%d:  '%s'\n", __FILE__, __LINE__, cudaGetErrorString(error));          \
        freeMemory(d_lut, d_cdf);                                       \
    }                                                                                         \
}

dim3 numBlocks(1, 1);      // μόνο 1 block καθώς χρειαζόμαστε μόνο 256 threads << 1024 threads που έχει ένα block
dim3 threadsPerBlock(256); // γραμμική (1D) γεωμετρία από threads

__constant__ int d_hist[256];

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
}

void freeMemory(int *d_lut, int *d_cdf)
{
    cudaError_t err;
    if (d_lut != NULL)
    {
        err = cudaFree(d_lut);
        if (err != cudaSuccess)
        {
            printf("Error during cudaFree (d_lut):  %s\n", cudaGetErrorString(err));
        }
    }

    if (d_cdf != NULL)
    {
        err = cudaFree(d_cdf);
        if (err != cudaSuccess)
        {
            printf("Error during cudaFree (d_cdf):  %s\n", cudaGetErrorString(err));
        }
    }

    printf("Reset Device\n");
    err = cudaDeviceReset();
    if (err != cudaSuccess){
        printf("Error during cudaDeviceReset:  %s\n", cudaGetErrorString(err));
    }
}

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    //host variables
    PGM_IMG result;
    int h_hist[256];
    int *h_lut = (int *)malloc(256 * sizeof(int));

    //device variables
    cudaError_t error;
    int *d_lut = NULL, *d_cdf = NULL;

    //constant variables
    int imageSize = img_in.w * img_in.h;
    int min;

    int i = 0;
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    error = cudaMalloc((void **)&d_lut, 256 * sizeof(int));
    if (error != cudaSuccess)
    {
        printf("Error allocating memory on host for d_lut:   %s\n", cudaGetErrorString(error));
        freeMemory(d_lut, d_cdf);
        return (result);
    }

    error = cudaMalloc((void **)&d_cdf, 256 * sizeof(int));
    if (error != cudaSuccess)
    {
        printf("Error allocating memory on host for d_cdf:   %s\n", cudaGetErrorString(error));
        freeMemory(d_lut, d_cdf);
        return (result);
    }

    histogram(h_hist, img_in.img, img_in.h * img_in.w, 256);
    //Now h_hist has the correct values, so it can be copied to constant memory

    while(min == 0){
        min = h_hist[i++];
    }

    error = cudaMemcpyToSymbol(d_hist, h_hist, 256 * sizeof(int));
    if (error != cudaSuccess)
    {
      printf("Error during cudaMemcpyToSymbol of h_hist to d_hist:  %s\n", cudaGetErrorString(error));
      freeMemory(d_lut, d_cdf);
      return (result);
    }

    prefix_sum_cdf <<< numBlocks, threadsPerBlock >>> (d_lut, d_cdf, imageSize, min);

    error = cudaMemcpy(h_lut, d_lut, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
      printf("Error during cudaMemcpy of d_lut to h_lut:  %s\n", cudaGetErrorString(error));
      freeMemory(d_lut, d_cdf);
      return (result);
    }

    //Error Checking
    cudaErrorCheck();

    for (i = 0; i < imageSize; i++)
    {
        if (h_lut[img_in.img[i]] < 0)
        {
            result.img[i] = 0;
        }
        else if (h_lut[img_in.img[i]] > 255)
        {
            result.img[i] = 255;
        }
        else
        {
            result.img[i] = (unsigned char)h_lut[img_in.img[i]];
        }
    }
    
    //histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);
    freeMemory(d_lut, d_cdf);
    return result;
}
