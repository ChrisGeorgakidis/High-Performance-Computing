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
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

  //pointers for the host
  float
  *h_Filter,
  *h_Input,
  *h_Buffer,
  *h_OutputCPU,
  *h_OutputGPU;

  //pointers for the device
  float
  *d_Filter,
  *d_Input,
  *d_Buffer,
  *d_OutputGPU;


  int imageW;     //image width = N
  int imageH;     //image height = N
  unsigned int i;
  cudaError_t err;

	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

  // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
  // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
  // Gia aplothta thewroume tetragwnikes eikones.

  printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
  scanf("%d", &imageW);
  imageH = imageW;

  printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
  printf("Allocating and initializing host arrays...\n");

  #pragma   //Allocate host (CPU) memory
  {
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    if (h_Filter == NULL){
      printf("Error allocating memory on host for h_Filter");
      return (ERROR);
    }
    h_Input     = (float *)malloc(ArraySize * sizeof(float));
    if (h_Input == NULL){
      printf("Error allocating memory on host for h_Input");
      return (ERROR);
    }
    h_Buffer    = (float *)malloc(ArraySize * sizeof(float));
    if (h_Buffer == NULL){
      printf("Error allocating memory on host for h_Buffer");
      return (ERROR);
    }
    h_OutputCPU = (float *)malloc(ArraySize * sizeof(float));
    if (h_OutputCPU == NULL){
      printf("Error allocating memory on host for h_OutputCPU");
      return (ERROR);
    }
    h_OutputGPU = (float *)malloc(ArraySize * sizeof(float));
    if (h_OutputGPU == NULL){
      printf("Error allocating memory on host for h_OutputGPU");
      return (ERROR);
    }

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.
  }

  printf("Allocate device (GPU) memory\n");
  #pragma   //Allocate device (GPU) memory
  {
    err = cudaMalloc( (void**) &d_Filter, FILTER_LENGTH * sizeof(float) );
    if (err != cudaSuccess){
      printf("Error allocating memory on host for d_Filter:   %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
    err = cudaMalloc( (void**) &d_Input, ArraySize * sizeof(float) );
    if (err != cudaSuccess){
      printf("Error allocating memory on host for d_Input:   %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
    err = cudaMalloc( (void**) &d_Buffer, ArraySize * sizeof(float) );
    if (err != cudaSuccess){
      printf("Error allocating memory on host for d_Buffer:   %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
    err = cudaMalloc( (void**) &d_OutputGPU, ArraySize * sizeof(float) );
    if (err != cudaSuccess){
      printf("Error allocating memory on host for d_OutputGPU:   %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
  }

  dim3 threadsPerBlock(N, N);                //geometry for block
  dim3 numBlocks(NUMBLOCKS, NUMBLOCKS);      //geometry for grid

  #pragma   //Initializations And copy memory from host to device
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

    printf("Copy host memory to device\n");
    //Copy host memory to device
    err = cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
      printf("Error during cudaMemcpy of h_Filter to d_Filter:  %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
    err = cudaMemcpy(d_Input, h_Input, ArraySize * sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
      printf("Error during cudaMemcpy of h_Input to d_Input:  %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
  }

  #pragma   //CPU Computation
  {
    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation is about to start...\n");

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles

    printf("CPU computation finished...\n");
  }

  #pragma   //GPU Computation
  {
    printf("GPU computation is about to start...\n");

    //kernel for row convolution
    //execute grid of numBlocks blocks of threadsPerBlock threads each
    convolutionRow <<< numBlocks, threadsPerBlock >>> (d_Input, d_Filter, d_Buffer, filter_radius, imageW);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess){
      printf ("Error during cudaDeviceSynchronize:  %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
    // err = cudaMemcpy(h_OutputGPU, d_Buffer, ArraySize * sizeof(float), cudaMemcpyDeviceToHost);
    // if(err != cudaSuccess){
    //   printf("Error during cudaMemcpy of d_Buffer to h_OutputGPU:  %s\n", cudaGetErrorString(err));
    //   return (ERROR);
    // }
    //Error Checking
    cudaErrorCheck();

    //kernel for column convolution
    //execute grid of numBlocks blocks of threadsPerBlock threads each
    convolutionColumn <<< numBlocks, threadsPerBlock >>> (d_Buffer, d_Filter, d_OutputGPU, filter_radius, imageW, imageH);

    err = cudaMemcpy(h_OutputGPU, d_OutputGPU, ArraySize * sizeof(float), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
      printf("Error during cudaMemcpy of d_OutputGPU to h_OutputGPU:  %s\n", cudaGetErrorString(err));
      return (ERROR);
    }

    //Error Checking
    cudaErrorCheck();
    printf("GPU computation finished...\n");
  }

  #pragma     //Compare the results from CPU and GPU
  {
    /*Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas*/
    for (i = 0; i < ArraySize; i++){
      if (h_OutputGPU[i] != h_OutputCPU[i]){
        printf("ERROR: Not the same result between h_OutputCPU and h_OutputGPU at index i = %d\n", i);
      }
    }
  }

  #pragma     //Free allocated host and device memory
  {
    // free all the allocated memory for the host
    free(h_OutputGPU);
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    //free all the allocated device (GPU) memory
    err = cudaFree(d_OutputGPU);
    if(err != cudaSuccess){
      printf("Error during cudaFree (d_OutputGPU):  %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
    err = cudaFree(d_Buffer);
    if(err != cudaSuccess){
      printf("Error during cudaFree (d_Buffer):  %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
    err = cudaFree(d_Input);
    if(err != cudaSuccess){
      printf("Error during cudaFree (d_Input):  %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
    err = cudaFree(d_Filter);
    if(err != cudaSuccess){
      printf("Error during cudaFree (d_Filter):  %s\n", cudaGetErrorString(err));
      return (ERROR);
    }
  }

  // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
  err = cudaDeviceReset();
  if (err != cudaSuccess){
    printf ("Error during cudaDeviceReset:  %s\n", cudaGetErrorString(err));
    return (ERROR);
  }
  printf("Bye bye! \n");

  return 0;
}
