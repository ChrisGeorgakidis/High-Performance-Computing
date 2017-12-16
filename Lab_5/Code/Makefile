Compiler = nvcc
OPTFLAGS = -O4 -fmad=false
DEBUGFLAGS = -g -G -Xcompiler -rdynamic

main: main.cu histogram-equalization.cu contrast-enhancement.cu
	$(Compiler) $(DEBUGFLAGS) $(OPTFLAGS) main.cu histogram-equalization.cu contrast-enhancement.cu -o histogram-equalization
clean:
	rm -rf histogram-equalization
