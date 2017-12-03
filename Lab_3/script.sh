#!/bin/bash

make clean
make

for value in {1..12};
do
    ./Convolution2D_multipleBlocks_floats >> 5b.txt
done
