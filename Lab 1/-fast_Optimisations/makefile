# Makefile. If you change it, remember than in makefiles multiple spaces
# ARE NOT EQUIVALENT to tabs. The line after a rule starts with a tab!

#Add any executable you want to be created here.
EXECUTABLES	= sobel_orig opt1_sobel_loop_unrolling opt1.5_not_sobel_loop_interchange opt2_sobel_function_inlining opt3_sobel_loop_invariant_code_motion opt4_sobel_common_subexpression_elimination_strength_reduction opt4.25_not_sobel_compiler_help opt4.5_not_sobel_loop_fusion

#This is the compiler to use
CC = icc

#These are the flags passed to the compiler. Change accordingly
CFLAGS = -Wall -fast

#These are the flags passed to the linker. Nothing in our case
LDFLAGS = -lm


# make all will create all executables
all: $(EXECUTABLES)

# This is the rule to create any executable from the corresponding .c 
# file with the same name.
%: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# make clean will remove all executables, jpg files and the 
# output of previous executions.
clean:
	rm -f $(EXECUTABLES) *.jpg output_sobel.grey

# make image will create the output_sobel.jpg from the output_sobel.grey. 
# Remember to change this rule if you change the name of the output file.
image: output_sobel.grey
	convert -depth 8 -size 4096x4096 GRAY:output_sobel.grey output_sobel.jpg 

