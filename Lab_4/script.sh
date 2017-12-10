#!/bin/bash

make clean

mkdir measurements
cd measurements
mkdir Tiled
mkdir Tiled_with_padding
cd Tiled_with_padding
mkdir floats
mkdir doubles
cd ../..

#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////

make Convolution2D_tiled

echo "Convolution2D_tiled.cu => floats" 
echo "~~~~~~~~~~~~~~~~~~~~~~"
echo "====> filter radius = 16"
echo "64x64"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f16_64.txt >> tiled_16_64x64.txt
    echo "End of run"
done

echo "128x128"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f16_128.txt >> tiled_16_128x128.txt
    echo "End of run"
done

echo "256x256"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f16_256.txt >> tiled_16_256x256.txt
    echo "End of run"
done

echo "512x512"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f16_512.txt >> tiled_16_512x512.txt
    echo "End of run"
done

echo "1024x1024"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f16_1024.txt >> tiled_16_1024x1024.txt
    echo "End of run"
done

echo "2048x2048"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f16_2048.txt >> tiled_16_2048x2048.txt
    echo "End of run"
done

echo "4096x4096"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f16_4096.txt >> tiled_16_4096x4096.txt
    echo "End of run"
done

echo "8192x8192"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f16_8192.txt >> tiled_16_8192x8192.txt
    echo "End of run"
done

echo "====> filter radius = 32"
echo "128x128"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f32_128.txt >> tiled_32_128x128.txt
    echo "End of run"
done

echo "256x256"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f32_256.txt >> tiled_32_256x256.txt
    echo "End of run"
done

echo "512x512"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f32_512.txt >> tiled_32_512x512.txt
    echo "End of run"
done

echo "1024x1024"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f32_1024.txt >> tiled_32_1024x1024.txt
    echo "End of run"
done

echo "2048x2048"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f32_2048.txt >> tiled_32_2048x2048.txt
    echo "End of run"
done

echo "4096x4096"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f32_4096.txt >> tiled_32_4096x4096.txt
    echo "End of run"
done

echo "8192x8192"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f32_8192.txt >> tiled_32_8192x8192.txt
    echo "End of run"
done

echo "====> filter radius = 64"
echo "256x256"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f64_256.txt >> tiled_64_256x256.txt
    echo "End of run"
done

echo "512x512"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f64_512.txt >> tiled_64_512x512.txt
    echo "End of run"
done

echo "1024x1024"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f64_1024.txt >> tiled_64_1024x1024.txt
    echo "End of run"
done

echo "2048x2048"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f64_2048.txt >> tiled_64_2048x2048.txt
    echo "End of run"
done

echo "4096x4096"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f64_4096.txt >> tiled_64_4096x4096.txt
    echo "End of run"
done

echo "8192x8192"
for value in {1..12};
do
    ./Convolution2D_tiled < Input_files/f64_8192.txt >> tiled_64_8192x8192.txt
    echo "End of run"
done

mv tiled_* measurements/Tiled

#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////


make Convolution2D_tiled_with_padding_floats

echo "Convolution2D_tiled_with_padding_floats.cu => floats" 
echo "~~~~~~~~~~~~~~~~~~~~~~"
echo "====> filter radius = 16"
echo "64x64"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f16_64.txt >> tiled_with_padding_floats_16_64x64.txt
    echo "End of run"
done

echo "128x128"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f16_128.txt >> tiled_with_padding_floats_16_128x128.txt
    echo "End of run"
done

echo "256x256"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f16_256.txt >> tiled_with_padding_floats_16_256x256.txt
    echo "End of run"
done

echo "512x512"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f16_512.txt >> tiled_with_padding_floats_16_512x512.txt
    echo "End of run"
done

echo "1024x1024"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f16_1024.txt >> tiled_with_padding_floats_16_1024x1024.txt
    echo "End of run"
done

echo "2048x2048"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f16_2048.txt >> tiled_with_padding_floats_16_2048x2048.txt
    echo "End of run"
done

echo "4096x4096"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f16_4096.txt >> tiled_with_padding_floats_16_4096x4096.txt
    echo "End of run"
done

echo "8192x8192"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f16_8192.txt >> tiled_with_padding_floats_16_8192x8192.txt
    echo "End of run"
done

echo "====> filter radius = 32"
echo "128x128"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f32_128.txt >> tiled_with_padding_floats_32_128x128.txt
    echo "End of run"
done

echo "256x256"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f32_256.txt >> tiled_with_padding_floats_32_256x256.txt
    echo "End of run"
done

echo "512x512"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f32_512.txt >> tiled_with_padding_floats_32_512x512.txt
    echo "End of run"
done

echo "1024x1024"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f32_1024.txt >> tiled_with_padding_floats_32_1024x1024.txt
    echo "End of run"
done

echo "2048x2048"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f32_2048.txt >> tiled_with_padding_floats_32_2048x2048.txt
    echo "End of run"
done

echo "4096x4096"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f32_4096.txt >> tiled_with_padding_floats_32_4096x4096.txt
    echo "End of run"
done

echo "8192x8192"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f32_8192.txt >> tiled_with_padding_floats_32_8192x8192.txt
    echo "End of run"
done

echo "====> filter radius = 64"
echo "256x256"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f64_256.txt >> tiled_with_padding_floats_64_256x256.txt
    echo "End of run"
done

echo "512x512"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f64_512.txt >> tiled_with_padding_floats_64_512x512.txt
    echo "End of run"
done

echo "1024x1024"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f64_1024.txt >> tiled_with_padding_floats_64_1024x1024.txt
    echo "End of run"
done

echo "2048x2048"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f64_2048.txt >> tiled_with_padding_floats_64_2048x2048.txt
    echo "End of run"
done

echo "4096x4096"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f64_4096.txt >> tiled_with_padding_floats_64_4096x4096.txt
    echo "End of run"
done

echo "8192x8192"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_floats < Input_files/f64_8192.txt >> tiled_with_padding_floats_64_8192x8192.txt
    echo "End of run"
done

mv tiled_with_padding_floats* measurements/Tiled_with_padding/floats

#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////

make Convolution2D_tiled_with_padding_doubles

echo "Convolution2D_tiled_with_padding_doubles.cu => doubles" 
echo "~~~~~~~~~~~~~~~~~~~~~~"
echo "====> filter radius = 16"
echo "64x64"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f16_64.txt >> tiled_with_padding_doubles_16_64x64.txt
    echo "End of run"
done

echo "128x128"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f16_128.txt >> tiled_with_padding_doubles_16_128x128.txt
    echo "End of run"
done

echo "256x256"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f16_256.txt >> tiled_with_padding_doubles_16_256x256.txt
    echo "End of run"
done

echo "512x512"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f16_512.txt >> tiled_with_padding_doubles_16_512x512.txt
    echo "End of run"
done

echo "1024x1024"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f16_1024.txt >> tiled_with_padding_doubles_16_1024x1024.txt
    echo "End of run"
done

echo "2048x2048"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f16_2048.txt >> tiled_with_padding_doubles_16_2048x2048.txt
    echo "End of run"
done

echo "4096x4096"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f16_4096.txt >> tiled_with_padding_doubles_16_4096x4096.txt
    echo "End of run"
done

echo "8192x8192"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f16_8192.txt >> tiled_with_padding_doubles_16_8192x8192.txt
    echo "End of run"
done

echo "====> filter radius = 32"
echo "128x128"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f32_128.txt >> tiled_with_padding_doubles_32_128x128.txt
    echo "End of run"
done

echo "256x256"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f32_256.txt >> tiled_with_padding_doubles_32_256x256.txt
    echo "End of run"
done

echo "512x512"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f32_512.txt >> tiled_with_padding_doubles_32_512x512.txt
    echo "End of run"
done

echo "1024x1024"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f32_1024.txt >> tiled_with_padding_doubles_32_1024x1024.txt
    echo "End of run"
done

echo "2048x2048"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f32_2048.txt >> tiled_with_padding_doubles_32_2048x2048.txt
    echo "End of run"
done

echo "4096x4096"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f32_4096.txt >> tiled_with_padding_doubles_32_4096x4096.txt
    echo "End of run"
done

echo "8192x8192"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f32_8192.txt >> tiled_with_padding_doubles_32_8192x8192.txt
    echo "End of run"
done

echo "====> filter radius = 64"
echo "256x256"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f64_256.txt >> tiled_with_padding_doubles_64_256x256.txt
    echo "End of run"
done

echo "512x512"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f64_512.txt >> tiled_with_padding_doubles_64_512x512.txt
    echo "End of run"
done

echo "1024x1024"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f64_1024.txt >> tiled_with_padding_doubles_64_1024x1024.txt
    echo "End of run"
done

echo "2048x2048"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f64_2048.txt >> tiled_with_padding_doubles_64_2048x2048.txt
    echo "End of run"
done

echo "4096x4096"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f64_4096.txt >> tiled_with_padding_doubles_64_4096x4096.txt
    echo "End of run"
done

echo "8192x8192"
for value in {1..12};
do
    ./Convolution2D_tiled_with_padding_doubles < Input_files/f64_8192.txt >> tiled_with_padding_doubles_64_8192x8192.txt
    echo "End of run"
done

mv tiled_with_padding_doubles* measurements/Tiled_with_padding/doubles