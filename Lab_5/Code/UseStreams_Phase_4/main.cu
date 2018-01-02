#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

void run_cpu_gray_test(PGM_IMG img_in, char *out_filename);

int main(int argc, char *argv[]){
    PGM_IMG img_ibuf_g;
    cudaError_t err;

	if (argc != 3) {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}
	
    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm(argv[1]);
    run_cpu_gray_test(img_ibuf_g, argv[2]);
    free_pgm(img_ibuf_g); //free input image

    //Reset Device before exit the program
	printf("Reset Device\n");
	err = cudaDeviceReset();
	if (err != cudaSuccess){
		printf("\033[1;31m");
		printf("Error during cudaDeviceReset:  %s\n", cudaGetErrorString(err));
		printf("\033[0m");
		exit(1);
	}

	return 0;
}



void run_cpu_gray_test(PGM_IMG img_in, char *out_filename)
{
    unsigned int timer = 0;
    PGM_IMG img_obuf;
    
    
    printf("Starting processing...\n");
    img_obuf = contrast_enhancement_g(img_in);
    write_pgm(img_obuf, out_filename);
    free_pgm(img_obuf); //free result image
}


PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    cudaError_t error;
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    
    //Allocate pinned memory for the result image
    error = cudaMallocHost((void **)&result.img, result.w*result.h*sizeof(unsigned char));
    if (error != cudaSuccess)
    {
        printf("\033[1;31m");
        printf("Error allocating pinned memory for image:   %s\n", cudaGetErrorString(error));
        printf("\033[0m");
        exit(1);
    }
    //result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

        
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    //free(img.img);
    //Now the image is pinned memory.
    cudaError_t error;
    error = cudaFreeHost(img.img);
    if (error != cudaSuccess) {
        printf("\033[1;31m");
        printf("Error freeing pinned memory for image:   %s\n", cudaGetErrorString(error));
        printf("\033[0m");
        exit(1);
    }
}

