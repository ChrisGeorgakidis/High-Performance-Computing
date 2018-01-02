#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
	//counting variable
	struct timespec histogram_start, histogram_stop, hist_equal_start, hist_equal_stop;
	double histogram_time, hist_equal_time;

    PGM_IMG result;
    int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
	
	clock_gettime(CLOCK_MONOTONIC_RAW, &histogram_start);
	histogram(hist, img_in.img, img_in.h * img_in.w, 256);
	clock_gettime(CLOCK_MONOTONIC_RAW, &histogram_stop);

	clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_start);
	histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);
	clock_gettime(CLOCK_MONOTONIC_RAW, &hist_equal_stop);

	//Calculate total time
		//histogram
			histogram_time = (double)(histogram_stop.tv_nsec - histogram_start.tv_nsec) / 1000000000.0 +
				(double)(histogram_stop.tv_sec - histogram_start.tv_sec);
			printf("\033[1;34m"); //Bold blue
			printf("histogram = ");
			printf("\033[1;32m"); //Bold green
			printf("%10g seconds\n", histogram_time);
		//histogram_equalization
			hist_equal_time = (double)(hist_equal_stop.tv_nsec - hist_equal_start.tv_nsec) / 1000000000.0 +
				(double)(hist_equal_stop.tv_sec - hist_equal_start.tv_sec);
			printf("\033[1;34m"); //Bold blue
			printf("histogram_equalization = ");
			printf("\033[1;35m"); //Bold magenta
			printf("%10g seconds\n", hist_equal_time);
		//Total Time
			printf("\033[1;34m"); //Bold blue
			printf("Total Time = ");
			printf("\033[1;33m"); //Bold Yellow
			printf("%10g ms\n", histogram_time*1000 + hist_equal_time*1000);
		printf("\033[0m");
	//__|
	printf("\033[0m");

    return result;
}
