import numpy as np
import sys

# read the clusters from the parallel and the sequential algorithm
parallel_output = np.loadtxt(sys.argv[1], dtype=float, usecols = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20))
sequential_output = np.loadtxt(sys.argv[2], dtype=float, usecols = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20))

# Calculate Residual sum of squares
diff = parallel_output-sequential_output
print("Error: ", np.amax(diff))
