NVCC = nvcc
OPTFLAGS = -O4
DEBUGFLAGS = -g -G -Xcompiler -rdynamic -lineinfo


Convolution2D_tiled: Convolution2D_tiled.cu
	$(NVCC) $(OPTFLAGS) Convolution2D_tiled.cu -o Convolution2D_tiled

Convolution2D_tiled_with_padding_floats: Convolution2D_tiled_with_padding_floats.cu
	$(NVCC) $(OPTFLAGS) Convolution2D_tiled_with_padding_floats.cu  -o Convolution2D_tiled_with_padding_floats

Convolution2D_tiled_with_padding_doubles: Convolution2D_tiled_with_padding_doubles.cu
	$(NVCC) $(OPTFLAGS) Convolution2D_tiled_with_padding_doubles.cu  -o Convolution2D_tiled_with_padding_doubles	

clean: 
	rm -rf Convolution2D_tiled 
	rm -rf Convolution2D_tiled_with_padding_floats
	rm -rf Convolution2D_tiled_with_padding_doubles
