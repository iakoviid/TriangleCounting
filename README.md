# Triangle Counting on the GPU

Tree implementations of triangle counting on the GPU in files triangles-simple.cu triangles-shared.cu and bitmap.cu.

## Getting Started

Compilation Commands:

make simple

make shared

make bitmap

Execution Commands:

simple [graphFile] [numberOfThreads] [numberOfBlocks]

shared [graphFile] [numberOfThreads] [numberOfBlocks] [numberOfColumns]

## Running the tests

./simple ./auto 512 256
./shared ./auto.mtx 128 1024 27 494
!./tr ./co.mtx 64 1024 1 1093



## Authors

Iakovidis Ioannis AM 8952



