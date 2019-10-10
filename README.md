# Triangle Counting on the GPU

Tree implementations of triangle counting on the GPU in files triangles-simple.cu triangles-shared.cu and bitmap.cu.

## Getting Started

Compilation Commands:

make simple

make shared

make bitmap

Execution Commands:

simple [graphFile] [numberOfThreads] [numberOfBlocks]

shared [graphFile] [numberOfThreads] [numberOfBlocks] [numberOfColumns] [MaxNumberOfElementsInacolumn]

## Running the tests

./simple ./auto 512 256

./shared ./auto.mtx 128 1024 27 494

./shared ./co.mtx 64 1024 1 1093


## P.S. (MaxNumberOfElementsInacolumn)
Max Number Of Elements In a column is given and it's time is not counted
because seeing the profiling it would be negligible and the implementation would be 
trivial a check and a max reduction.

Running the c serial test-single file would print you this argument.

## Authors

Iakovidis Ioannis AM 8952



