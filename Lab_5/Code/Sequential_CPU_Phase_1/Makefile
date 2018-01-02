Compiler = nvcc
OPTFLAGS = -O4
DEBUGFLAGS = -g -G -Xcompiler -rdynamic

main: main.cu histogram-equalization.cu contrast-enhancement.cu
	$(Compiler) $(OPTFLAGS) main.cu histogram-equalization.cu contrast-enhancement.cu -o histogram-equalization
clean:
	rm -rf histogram-equalization
