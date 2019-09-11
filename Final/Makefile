simple: triangles-simple.o mmio.o
	nvcc  triangles-simple.o mmio.o -o simple

triangles-simple.o: triangles-simple.cu
	nvcc -c triangles-simple.cu

shared: triangles-shared.o mmio.o
	nvcc  triangles-shared.o mmio.o -o shared

triangles-shared.o: triangles-shared.cu
	nvcc -c triangles-shared.cu

bitmap: bitmap.o mmio.o
	nvcc  bitmap.o mmio.o -o bitmap

bitmap.o: bitmap.cu
	nvcc -c bitmap.cu

device: 
	nvcc checkDeviceInfor.cu -o device

mmio.o: mmio.cpp mmio.h
	g++ -w -c mmio.cpp

clean:
	rm -f *.o