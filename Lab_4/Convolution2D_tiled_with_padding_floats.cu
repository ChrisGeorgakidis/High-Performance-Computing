/*
* This sample implements a separable convolution
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

unsigned int filter_radius;

#define FILTER_LENGTH (2 * filter_radius + 1)
#define ABS(val) ((val) < 0.0 ? (-(val)) : (val))
#define accuracy 0.00005
#define ArraySize imageW *imageH
#define ERROR -1
#define FILTER_R_X2 2*filter_radius
#define SH_MEM_SIZE 32
#define NUMBLOCKS 4

typedef float dataType;
__constant__ dataType d_Filter[65536/sizeof(dataType)];

// This checks for cuda errors
#define cudaErrorCheck()                                                                                 \
  {                                                                                                      \
    cudaError_t error = cudaGetLastError();                                                              \
    if (error != cudaSuccess)                                                                            \
    {                                                                                                    \
      printf("Cuda Error Found %s:%d:  '%s'\n", __FILE__, __LINE__, cudaGetErrorString(error));          \
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Input, d_Buffer, d_OutputGPU); \
      return (ERROR);                                                                                    \
    }                                                                                                    \
  }

#define cudaCalloc(pointer, size, sizeOfElement)                                                         \
  {                                                                                                      \
    cudaError_t err = cudaMalloc(pointer, size * sizeOfElement);                                         \
    if (err != cudaSuccess)                                                                              \
    {                                                                                                    \
      printf("Error allocating memory on host:   %s\n", cudaGetErrorString(err));                        \
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Input, d_Buffer, d_OutputGPU); \
      return (ERROR);                                                                                    \
    }                                                                                                    \
    cudaMemset(*pointer, 0.0, size *sizeOfElement);                                                      \
  }

////////////////////////////////////////////////////////////////////////////////
// Kernel Row Convolution Filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRow(dataType *Input, dataType *Output, int filterR, int imageW)
{
  dataType sum = 0;
  int d, k;

  int ix = blockIdx.x * blockDim.x + threadIdx.x + filterR;
  int iy = blockIdx.y * blockDim.y + threadIdx.y + filterR;
  //int dimx = blockDim.x * gridDim.x + 2 * filterR;
  //int idx = iy * dimx + ix;
  int imageWithPaddingW = imageW + 2 * filterR;

  for (k = -filterR; k <= filterR; k++)
  {
    d = ix + k;
    sum += Input[iy * imageWithPaddingW + d] * d_Filter[filterR - k];
  }
  Output[iy * imageWithPaddingW + ix] = sum; //Only 1 time for each thread
}


////////////////////////////////////////////////////////////////////////////////
// Kernel Column Convolution Filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumn(dataType *Input, dataType *Output, int filterR, int imageW, int imageH)
{
  dataType sum = 0;
  int d, k;

  int ix = blockIdx.x * blockDim.x + threadIdx.x + filterR;
  int iy = blockIdx.y * blockDim.y + threadIdx.y + filterR;

  int imageWithPaddingW = imageW + 2 * filterR;

  for (k = -filterR; k <= filterR; k++)
  {
    d = iy + k;

    sum += Input[d * imageWithPaddingW + ix] * d_Filter[filterR - k];

    Output[iy * imageWithPaddingW + ix] = sum;
  }
}


////////////////////////////////////////////////////////////////////////////////
// Kernel Row Convolution Filter using Shared Memory
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowSharedMem(dataType *Input, dataType *Output, int filterR, int imageW, int SH_MEM_SIZE_PAD)
{
  dataType sum = 0;
  int d, k;

  int tx = threadIdx.x + filterR;
  int ix = blockIdx.x * blockDim.x + threadIdx.x + filterR;
  int iy = blockIdx.y * blockDim.y + threadIdx.y + filterR;

  //indexes in arrays including padding
  int ptx = threadIdx.x;
  int pty = threadIdx.y;
  int pix = blockIdx.x * blockDim.x + threadIdx.x;

  int imageWithPaddingW = imageW + 2 * filterR;

  //shared memory for Input
  extern __shared__ dataType s_Input[]; // shared memory with padding

  //collaboratively load tiles into __shared__
  for (int i = 0; i < SH_MEM_SIZE_PAD/32; i++){
    s_Input[pty * SH_MEM_SIZE_PAD + (SH_MEM_SIZE_PAD / 32) * ptx + i] = Input[iy * imageWithPaddingW + (SH_MEM_SIZE_PAD / 32) * pix + i - ((SH_MEM_SIZE_PAD / 32) - 1) * (blockIdx.x * blockDim.x)];
  }

  __syncthreads();

  for (k = -filterR; k <= filterR; k++)
  {
    d = tx + k;
    sum += s_Input[pty * SH_MEM_SIZE_PAD + d] * d_Filter[filterR - k];
  }
  Output[iy * imageWithPaddingW + ix] = sum; //1 time for each thread
}


////////////////////////////////////////////////////////////////////////////////
// Kernel Column Convolution Filter using Shared Memory
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnSharedMem(dataType *Input, dataType *Output, int filterR, int imageW, int imageH, int SH_MEM_SIZE_PAD)
{
  dataType sum = 0;
  int d, k;

  int ty = threadIdx.y + filterR;
  int ix = blockIdx.x * blockDim.x + threadIdx.x + filterR;
  int iy = blockIdx.y * blockDim.y + threadIdx.y + filterR;

  //indexes in arrays including padding
  int ptx = threadIdx.x;
  int pty = threadIdx.y;
  int piy = blockIdx.y * blockDim.y + pty;

  int imageWithPaddingW = imageW + 2 * filterR;

  //shared memory for Input
  extern __shared__ dataType s_Input[];

  //collaboratively load tiles into __shared__
  for (int i = 0; i < SH_MEM_SIZE_PAD/32; i++){
    s_Input[(pty * (SH_MEM_SIZE_PAD / 32) + i) * SH_MEM_SIZE + ptx] = Input[(piy * (SH_MEM_SIZE_PAD / 32) + i - ((SH_MEM_SIZE_PAD / 32) - 1)*(blockIdx.y * blockDim.y)) * imageWithPaddingW + ix];
  }
  __syncthreads();

  for (k = -filterR; k <= filterR; k++)
  {
    d = ty + k;
    sum += s_Input[d * SH_MEM_SIZE + ptx] * d_Filter[filterR - k];
  }
  Output[iy * imageWithPaddingW + ix] = sum; //One time for each thread
}


////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(dataType *h_Dst, dataType *h_Src, dataType *h_Filter,
int imageW, int imageH, int filterR)
{
  int x, y, k;
  int imageWithPaddingW = imageW + 2 * filterR;

  for (y = filterR; y < (imageH + filterR); y++)
  {
    for (x = filterR; x < (imageW + filterR); x++)
    {
      dataType sum = 0;

      for (k = -filterR; k <= filterR; k++)
      {
        int d = x + k;
        sum += h_Src[y * imageWithPaddingW + d] * h_Filter[filterR - k];
      }
      h_Dst[y * imageWithPaddingW + x] = sum; //One time for each x & y
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(dataType *h_Dst, dataType *h_Src, dataType *h_Filter,
int imageW, int imageH, int filterR)
{
  int x, y, k;
  int imageWithPaddingW = imageW + 2 * filterR;

  for (y = filterR; y < (imageH + filterR); y++)
  {
    for (x = filterR; x < (imageW + filterR); x++)
    {
      dataType sum = 0;

      for (k = -filterR; k <= filterR; k++)
      {
        int d = y + k;
        sum += h_Src[d * imageWithPaddingW + x] * h_Filter[filterR - k];
      }
      h_Dst[y * imageWithPaddingW + x] = sum; //One time for each x & y
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Free Alocated Host and Device Memory
////////////////////////////////////////////////////////////////////////////////
int freeMemory(dataType *h_Filter, dataType *h_Input, dataType *h_Buffer, dataType *h_OutputCPU, dataType *h_OutputGPU, dataType *d_Input, dataType *d_Buffer, dataType *d_OutputGPU)
{
  cudaError_t err;

  // free all the allocated memory for the host
  printf("Free host memory...\n");
  if (h_OutputGPU != NULL)
  {
    free(h_OutputGPU);
  }
  if (h_OutputCPU != NULL)
  {
    free(h_OutputCPU);
  }
  if (h_Buffer != NULL)
  {
    free(h_Buffer);
  }
  if (h_Input != NULL)
  {
    free(h_Input);
  }
  if (h_Filter != NULL)
  {
    free(h_Filter);
  }

  //free all the allocated device (GPU) memory
  printf("Free device memory...\n");
  if (d_OutputGPU != NULL)
  {
    err = cudaFree(d_OutputGPU);
    if (err != cudaSuccess)
    {
      printf("Error during cudaFree (d_OutputGPU):  %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
  }
  if (d_Buffer != NULL)
  {
    err = cudaFree(d_Buffer);
    if (err != cudaSuccess)
    {
      printf("Error during cudaFree (d_Buffer):  %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
  }
  if (d_Input != NULL)
  {
    err = cudaFree(d_Input);
    if (err != cudaSuccess)
    {
      printf("Error during cudaFree (d_Input):  %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
  }

  // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
  printf("Reset Device\n");
  err = cudaDeviceReset();
  if (err != cudaSuccess)
  {
    printf("Error during cudaDeviceReset:  %s\n", cudaGetErrorString(err));
    return (ERROR);
  }

  return (0);
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

  //pointers for the host
  dataType
      *h_Filter = NULL,
      *h_Input = NULL,
      *h_Buffer = NULL,
      *h_OutputCPU = NULL,
      *h_OutputGPU = NULL;

  //pointers for the device
  dataType
      *d_Input = NULL,
      *d_Buffer = NULL,
      *d_OutputGPU = NULL;

  int imageW; //image width = N
  int imageH; //image height = N
  unsigned int i, j, block_size, numberOfBlocks;
  dataType diff = 0, max_diff = 0;

  /*-------timing variables-------*/
  struct timespec tv1, tv2;
  cudaError_t err;
  cudaEvent_t start;
  cudaEvent_t stop;
  float elapsed;
  /*------------------------------*/

  /*------padding variables-------*/
  int imageWithPaddingW, newImageSize;
  /*------------------------------*/

  printf("Enter filter radius : ");
  scanf("%d", &filter_radius);

  // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
  // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
  // Gia aplothta thewroume tetragwnikes eikones.

  printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
  scanf("%d", &imageW); //TODO Warning
  imageH = imageW;

  imageWithPaddingW = imageW + 2 * filter_radius;
  newImageSize = imageWithPaddingW * imageWithPaddingW;

  printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
  printf("Image Width x Height = %i x %i\n\n", imageWithPaddingW, imageWithPaddingW);
  printf("Allocating and initializing host arrays...\n");

  //Allocate host (CPU) memory
  {
    h_Filter = (dataType *)malloc(FILTER_LENGTH * sizeof(dataType));
    if (h_Filter == NULL)
    {
      printf("Error allocating memory on host for h_Filter");
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
    h_Input = (dataType *)calloc(newImageSize, sizeof(dataType));
    if (h_Input == NULL)
    {
      printf("Error allocating memory on host for h_Input");
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
    h_Buffer = (dataType *)calloc(newImageSize, sizeof(dataType));
    if (h_Buffer == NULL)
    {
      printf("Error allocating memory on host for h_Buffer");
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
    h_OutputCPU = (dataType *)calloc(newImageSize, sizeof(dataType));
    if (h_OutputCPU == NULL)
    {
      printf("Error allocating memory on host for h_OutputCPU");
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
    h_OutputGPU = (dataType *)calloc(newImageSize, sizeof(dataType));
    if (h_OutputGPU == NULL)
    {
      printf("Error allocating memory on host for h_OutputGPU");
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.
  }

  printf("Allocate device (GPU) memory\n");
  //Allocate device (GPU) memory
  {
    err = cudaMalloc((void **)&d_Input, newImageSize * sizeof(dataType));
    if (err != cudaSuccess)
    {
      printf("Error allocating memory on host for d_Input:   %s\n", cudaGetErrorString(err));
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
    cudaCalloc((void **)&d_Buffer, newImageSize, sizeof(dataType));

    cudaCalloc((void **)&d_OutputGPU, newImageSize, sizeof(dataType));
  }

  if (imageW < 32)
  {
    block_size = imageW;
    numberOfBlocks = 1;
  }
  else
  {
    block_size = 32;
    numberOfBlocks = imageW / block_size;
  }

  dim3 threadsPerBlock(block_size, block_size);   //geometry for block
  dim3 numBlocks(numberOfBlocks, numberOfBlocks); //geometry for grid
  int SH_MEM_SIZE_PAD = 32 + 2 * filter_radius;

  //Initializations
  {
    srand(200);
    // Random initialization of h_Filter
    for (i = 0; i < FILTER_LENGTH; i++)
    {
      h_Filter[i] = (dataType)(rand() % 16);
    }

    // Random initialization of h_Input
    for (i = filter_radius; i < (imageH + filter_radius); i++)
    {
      for (j = filter_radius; j < (imageW + filter_radius); j++)
      {
        h_Input[i * imageWithPaddingW + j] = (dataType)rand() / ((dataType)RAND_MAX / 255) + (dataType)rand() / (dataType)RAND_MAX;
      }
    }
  }

  //CPU Computation
  {
    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation is about to start...\n");
    //Get the starting time
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius);        // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
    //Take the end time
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

    printf("CPU computation finished...\n");
  }

  //Calculate the duration of the CPU computation and report it
  {
    printf("\033[1;33m");
    printf("CPU time = %10g seconds\n",
           (double)(tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
               (double)(tv2.tv_sec - tv1.tv_sec));
  }
  printf("\033[0m");

  //Copy from host to device
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("Copy host memory to device\n");

    cudaEventRecord(start, 0);
    //Copy host memory to device
    err = cudaMemcpyToSymbol(d_Filter, h_Filter, FILTER_LENGTH * sizeof(dataType));
    if (err != cudaSuccess)
    {
      printf("Error during cudaMemcpyToSymbol of h_Filter to d_Filter:  %s\n", cudaGetErrorString(err));
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
    err = cudaMemcpy(d_Input, h_Input, newImageSize * sizeof(dataType), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      printf("Error during cudaMemcpy of h_Input to d_Input:  %s\n", cudaGetErrorString(err));
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
  }

  //GPU Computation
  {
    printf("GPU computation is about to start...\n");

    //kernel for row convolution
    //execute grid of numBlocks blocks of threadsPerBlock threads each
    convolutionRowSharedMem<<<numBlocks, threadsPerBlock, (32 *( 32 + 2 * filter_radius)) * sizeof(dataType)>>>(d_Input, d_Buffer, filter_radius, imageW, SH_MEM_SIZE_PAD);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
      printf("Error during cudaDeviceSynchronize:  %s\n", cudaGetErrorString(err));
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }

    //Error Checking
    cudaErrorCheck();

    //kernel for column convolution
    //execute grid of numBlocks blocks of threadsPerBlock threads each
    convolutionColumnSharedMem<<<numBlocks, threadsPerBlock, (32 * (32 + 2 * filter_radius)) * sizeof(dataType)>>>(d_Buffer, d_OutputGPU, filter_radius, imageW, imageH, SH_MEM_SIZE_PAD);

    err = cudaMemcpy(h_OutputGPU, d_OutputGPU, newImageSize * sizeof(dataType), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
      printf("Error during cudaMemcpy of d_OutputGPU to h_OutputGPU:  %s\n", cudaGetErrorString(err));
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }

    //Error Checking
    cudaErrorCheck();
    cudaEventRecord(stop, 0);
    printf("GPU computation finished...\n");
  }

  //Execution Time of GPU
  {
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("\033[1;35m");
    printf("GPU time = %g ms\n", elapsed);
    printf("\033[0m");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  //Compare the results from CPU and GPU
  {
    for (i = filter_radius; i < imageH + filter_radius; i++)
    {
      for (j = filter_radius; j < imageW + filter_radius; j++)
      {
        diff = ABS(h_OutputCPU[i * imageWithPaddingW + j] - h_OutputGPU[i * imageWithPaddingW + j]);
        //printf("The difference between h_OutputCPU[%d]=%lf and h_OutputGPU[%d]=%lf is diff = %g\n", i * imageWithPaddingW + j, h_OutputCPU[i * imageWithPaddingW + j], i * imageWithPaddingW + j, h_OutputGPU[i * imageWithPaddingW + j], diff);
        if (diff > max_diff)
        {
          max_diff = diff;
        }
        if (diff > accuracy)
        {
          //printf("\t|->ERROR: The difference between the values of h_OutputCPU and h_OutputGPU at index i = %u is bigger than the given accuracy.\n", i);
        }
      }
    }

    if (max_diff > accuracy)
    {
      printf("\033[1;31m");
      printf("ERROR! Max difference between the values of h_OutputCPU and h_OutputGPU is max_diff = %g\n", max_diff);
    }
    else
    {
      printf("\033[1;32m");
      printf("Max difference between the values of h_OutputCPU and h_OutputGPU is max_diff = %g\n", max_diff);
    }
  }

  //Free allocated host and device memory
  printf("\033[0m");
  freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Input, d_Buffer, d_OutputGPU);

  return 0;
}
