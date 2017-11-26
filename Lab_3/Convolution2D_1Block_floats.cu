/*
* This sample implements a separable convolution
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005
#define ArraySize       imageW * imageH
#define ERROR     -1
#define N  imageW
#define NUMBLOCKS 1

// This checks for cuda errors
#define cudaErrorCheck() {\
  cudaError_t error = cudaGetLastError();\
  if (error != cudaSuccess) {\
    printf("Cuda Error Found %s:%d:  '%s'\n", __FILE__, __LINE__, cudaGetErrorString(error));\
    freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Filter, d_Input, d_Buffer, d_OutputGPU);\
    return (ERROR);\
  }\
}\

////////////////////////////////////////////////////////////////////////////////
// Kernel Row Convolution Filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRow(float *Input, float *Filter, float *Output, int filterR, int imageW)
{
  float sum = 0;
  int d, k;
  
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int dimx = blockDim.x * gridDim.x;
  int idx = iy * dimx + ix;

  for (k = -filterR; k <= filterR; k++){
    d = ix + k;

    if (d >= 0 && d < imageW){
      sum += Input[iy * imageW + d] * Filter[filterR - k];
    } 
    
    Output[idx] = sum;
  }

}




////////////////////////////////////////////////////////////////////////////////
// Kernel Column Convolution Filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumn(float *Input, float *Filter, float *Output, int filterR, int imageW, int imageH)
{
  float sum = 0;
  int d, k;

  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int dimx = blockDim.x * gridDim.x;
  int idx = iy * dimx + ix;

  for (k = -filterR; k <= filterR; k++){
    d = iy + k;

    if (d >= 0 && d < imageH){
      sum += Input[d * imageW + ix] * Filter[filterR - k];
    } 
    
    Output[idx] = sum;
  }
}




////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter,
int imageW, int imageH, int filterR) {

  int x, y, k;

  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }

        h_Dst[y * imageW + x] = sum;
      }
    }
  }
}




////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
int imageW, int imageH, int filterR) {

  int x, y, k;

  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }

        h_Dst[y * imageW + x] = sum;
      }
    }
  }
}




////////////////////////////////////////////////////////////////////////////////
// Free Alocated Host and Device Memory
////////////////////////////////////////////////////////////////////////////////
int freeMemory(float * h_Filter, float *h_Input, float *h_Buffer, float *h_OutputCPU, float *h_OutputGPU, float *d_Filter, float *d_Input, float *d_Buffer, float *d_OutputGPU){
  cudaError_t err;

  // free all the allocated memory for the host
  printf("Free host memory...\n");
  if (h_OutputGPU != NULL){
    free(h_OutputGPU);
  }
  if (h_OutputCPU != NULL){
    free(h_OutputCPU);
  }
  if (h_Buffer != NULL){
    free(h_Buffer);
  }
  if (h_Input != NULL){
    free(h_Input);
  }
  if (h_Filter != NULL){
    free(h_Filter);
  }
  
  //free all the allocated device (GPU) memory
  printf("Free device memory...\n");
  if (d_OutputGPU != NULL){
    err = cudaFree(d_OutputGPU);
    if (err != cudaSuccess)
    {
      printf("Error during cudaFree (d_OutputGPU):  %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
  }
  if (d_Buffer != NULL){
    err = cudaFree(d_Buffer);
    if (err != cudaSuccess)
    {
      printf("Error during cudaFree (d_Buffer):  %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
  }
  if (d_Input != NULL){
    err = cudaFree(d_Input);
    if (err != cudaSuccess)
    {
      printf("Error during cudaFree (d_Input):  %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
  }
  if (d_Filter != NULL){
    err = cudaFree(d_Filter);
    if (err != cudaSuccess)
    {
      printf("Error during cudaFree (d_Filter):  %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
  }
  
  // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
  printf("Reset Device\n");
  err = cudaDeviceReset();
  if (err != cudaSuccess){
    printf("Error during cudaDeviceReset:  %s\n", cudaGetErrorString(err));
    return (ERROR);
  }

  return (0);
}




////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

	//pointers for the host
  float
  *h_Filter = NULL,
  *h_Input = NULL,
  *h_Buffer = NULL,
  *h_OutputCPU = NULL,
  *h_OutputGPU = NULL;

  //pointers for the device
  float
  *d_Filter = NULL,
  *d_Input = NULL,
  *d_Buffer = NULL,
  *d_OutputGPU = NULL;


  int imageW;     //image width = N
  int imageH;     //image height = N
  unsigned int i, block_size;
	float diff = 0, max_diff = 0;
  cudaError_t err;
  

  printf("Enter filter radius : ");
	scanf("%d", &filter_radius);					// TODO Warning

  // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
  // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
  // Gia aplothta thewroume tetragwnikes eikones.

  printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
  scanf("%d", &imageW);								//TODO Warning
  imageH = imageW;

  printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
  printf("Allocating and initializing host arrays...\n");

  //Allocate host (CPU) memory
  {
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    if (h_Filter == NULL){
      printf("Error allocating memory on host for h_Filter");
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Filter, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
    h_Input     = (float *)malloc(ArraySize * sizeof(float));
    if (h_Input == NULL){
      printf("Error allocating memory on host for h_Input");
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Filter, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
    h_Buffer    = (float *)malloc(ArraySize * sizeof(float));
    if (h_Buffer == NULL){
      printf("Error allocating memory on host for h_Buffer");
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Filter, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
    h_OutputCPU = (float *)malloc(ArraySize * sizeof(float));
    if (h_OutputCPU == NULL){
      printf("Error allocating memory on host for h_OutputCPU");
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Filter, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
    h_OutputGPU = (float *)malloc(ArraySize * sizeof(float));
    if (h_OutputGPU == NULL){
      printf("Error allocating memory on host for h_OutputGPU");
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Filter, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.
  }

  printf("Allocate device (GPU) memory\n");
  //Allocate device (GPU) memory
  {
    err = cudaMalloc( (void**) &d_Filter, FILTER_LENGTH * sizeof(float) );
    if (err != cudaSuccess){
      printf("Error allocating memory on host for d_Filter:   %s\n", cudaGetErrorString(err));
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Filter, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
    err = cudaMalloc( (void**) &d_Input, ArraySize * sizeof(float) );
    if (err != cudaSuccess){
      printf("Error allocating memory on host for d_Input:   %s\n", cudaGetErrorString(err));
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Filter, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
    err = cudaMalloc( (void**) &d_Buffer, ArraySize * sizeof(float) );
    if (err != cudaSuccess){
      printf("Error allocating memory on host for d_Buffer:   %s\n", cudaGetErrorString(err));
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Filter, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
    err = cudaMalloc( (void**) &d_OutputGPU, ArraySize * sizeof(float) );
    if (err != cudaSuccess){
      printf("Error allocating memory on host for d_OutputGPU:   %s\n", cudaGetErrorString(err));
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Filter, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
  }

  //Geometry for blocks and grid
  
  block_size = N;

  dim3 threadsPerBlock(block_size, block_size); //geometry for block
  dim3 numBlocks(NUMBLOCKS, NUMBLOCKS);         //geometry for grid
  
	
  //Initializations
  {
    srand(200);
    // Random initialization of h_Filter
    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

    // Random initialization of h_Input
    for (i = 0; i < ArraySize; i++) {
        h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
    }
  }

  //CPU Computation
  {
    printf("CPU computation is about to start...\n");
  
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // row convolution
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // column convolution
    
    printf("CPU computation finished...\n");
  }

  //Copy from host to device
  {
    printf("Copy host memory to device\n");
    //Copy host memory to device
    err = cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      printf("Error during cudaMemcpy of h_Filter to d_Filter:  %s\n", cudaGetErrorString(err));
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Filter, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
    err = cudaMemcpy(d_Input, h_Input, ArraySize * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      printf("Error during cudaMemcpy of h_Input to d_Input:  %s\n", cudaGetErrorString(err));
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Filter, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
  }
  //GPU Computation
  {
    printf("GPU computation is about to start...\n");

    //kernel for row convolution
    //execute grid of numBlocks blocks of threadsPerBlock threads each
    convolutionRow <<< numBlocks, threadsPerBlock >>> (d_Input, d_Filter, d_Buffer, filter_radius, imageW);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess){
      printf ("Error during cudaDeviceSynchronize:  %s\n", cudaGetErrorString(err));
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Filter, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }
    
    //Error Checking
    cudaErrorCheck();

    //kernel for column convolution
    //execute grid of numBlocks blocks of threadsPerBlock threads each
    convolutionColumn <<< numBlocks, threadsPerBlock >>> (d_Buffer, d_Filter, d_OutputGPU, filter_radius, imageW, imageH);

    err = cudaMemcpy(h_OutputGPU, d_OutputGPU, ArraySize * sizeof(float), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
      printf("Error during cudaMemcpy of d_OutputGPU to h_OutputGPU:  %s\n", cudaGetErrorString(err));
      freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Filter, d_Input, d_Buffer, d_OutputGPU);
      return (ERROR);
    }

    //Error Checking
    cudaErrorCheck();
  
    printf("GPU computation finished...\n");
  }

  //Compare the results from CPU and GPU
  {
    /*Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas*/
    for (i = 0; i < ArraySize; i++){
			diff = ABS(h_OutputGPU[i] - h_OutputCPU[i]);
			printf("The difference between the values of h_OutputCPU and h_OutputGPU at index i = %u is diff = %g\n", i, diff);
      if (diff > max_diff) {
        max_diff = diff;
      }
      if (diff > accuracy){
        printf("\t|->ERROR: The difference between the values of h_OutputCPU and h_OutputGPU at index i = %u is bigger than the given accuracy.\n", i);
      }
    }

    printf("Max difference between the values of h_OutputCPU and h_OutputGPU is max_diff = %g\n", max_diff);
  }

  //Free allocated host and device memory
  freeMemory(h_Filter, h_Input, h_Buffer, h_OutputCPU, h_OutputGPU, d_Filter, d_Input, d_Buffer, d_OutputGPU);

  return 0;
}
