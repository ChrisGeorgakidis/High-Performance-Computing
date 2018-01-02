#!/bin/bash

make clean
rm -rf Measurements
rm -rf Output_Images
make
clear

echo "uth.pgm"
for value in {1..12};
do
	./histogram-equalization ../../Input_Images/uth.pgm uth.pgm >> uth_times.txt
	echo "End of run"
done

echo "x_ray.pgm"
for value in {1..12};
do
	./histogram-equalization ../../Input_Images/x_ray.pgm x_ray.pgm >> x_ray_times.txt
	
	echo "End of run"
done

echo "ship.pgm"
for value in {1..12};
do
	./histogram-equalization ../../Input_Images/ship.pgm ship.pgm >> ship_times.txt
	echo "End of run"
done

echo "planet_surface.pgm"
for value in {1..12};
do
	./histogram-equalization ../../Input_Images/planet_surface.pgm planet_surface.pgm >> planet_surface_times.txt
	echo "End of run"
done

mkdir Output_Images
mv uth.pgm x_ray.pgm ship.pgm planet_surface.pgm Output_Images

mkdir Measurements
mv uth_times.txt x_ray_times.txt ship_times.txt planet_surface_times.txt Measurements