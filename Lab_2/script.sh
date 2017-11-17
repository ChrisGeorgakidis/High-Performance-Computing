#!/bin/bash

export OMP_PROC_BIND=true
export OMP_DYNAMIC=false

#Running 12 times the sequential algorithm
cd sequential-kmeans
rm -rf sequential_metrices.txt
make clean
make
for value in {1..12}
do
  make check
done

#Running 12 times the parallel algorithm for different number of threads
cd ../parallel-kmeans
rm -rf parallel_metrices.txt
make clean
make
export OMP_NUM_THREADS=1
for value in {1..12};
do
    make check
done
python error_calculator.py Image_data/texture17695.bin.cluster_centres ../sequential-kmeans/Image_data/texture17695.bin.cluster_centres >> parallel_metrices.txt

export OMP_NUM_THREADS=4
for value in {1..12};
do
    make check
done
python error_calculator.py Image_data/texture17695.bin.cluster_centres ../sequential-kmeans/Image_data/texture17695.bin.cluster_centres >> parallel_metrices.txt

export OMP_NUM_THREADS=8
for value in {1..12};
do
    make check
done
python error_calculator.py Image_data/texture17695.bin.cluster_centres ../sequential-kmeans/Image_data/texture17695.bin.cluster_centres >> parallel_metrices.txt

export OMP_NUM_THREADS=16
for value in {1..12};
do
    make check
done
python error_calculator.py Image_data/texture17695.bin.cluster_centres ../sequential-kmeans/Image_data/texture17695.bin.cluster_centres >> parallel_metrices.txt

export OMP_NUM_THREADS=28
for value in {1..12};
do
    make check
done
python error_calculator.py Image_data/texture17695.bin.cluster_centres ../sequential-kmeans/Image_data/texture17695.bin.cluster_centres >> parallel_metrices.txt

export OMP_NUM_THREADS=32
for value in {1..12};
do
    make check
done
python error_calculator.py Image_data/texture17695.bin.cluster_centres ../sequential-kmeans/Image_data/texture17695.bin.cluster_centres >> parallel_metrices.txt

export OMP_NUM_THREADS=56
for value in {1..12};
do
    make check
done
python error_calculator.py Image_data/texture17695.bin.cluster_centres ../sequential-kmeans/Image_data/texture17695.bin.cluster_centres >> parallel_metrices.txt
