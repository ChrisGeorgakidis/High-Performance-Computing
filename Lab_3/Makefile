NVCC = nvcc
OPTFLAGS = -O4
DEBUGFLAGS = -g -G -Xcompiler -rdynamic -lineinfo

Convolution2D_1Block_floats: Convolution2D_1Block_floats.cu
	$(NVCC) $(OPTFLAGS) -o Convolution2D_1Block_floats Convolution2D_1Block_floats.cu

Convolution2D_multipleBlocks_floats: Convolution2D_multipleBlocks_floats.cu
	$(NVCC) $(OPTFLAGS) -o Convolution2D_multipleBlocks_floats Convolution2D_multipleBlocks_floats.cu

Convolution2D_multipleBlocks_doubles: Convolution2D_multipleBlocks_doubles.cu
	$(NVCC) $(OPTFLAGS) -o Convolution2D_multipleBlocks_doubles Convolution2D_multipleBlocks_doubles.cu

Convolution2D_padding: Convolution2D_padding.cu
	$(NVCC) $(OPTFLAGS) -o Convolution2D_padding Convolution2D_padding.cu

clean: rm -rf *.o
