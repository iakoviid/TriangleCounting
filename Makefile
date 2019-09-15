simple:
	nvcc  triangles-simple.cu mmio.c -w -o simple

shared:
	nvcc triangles-shared.cu mmio.c -w -o shared

bitmap:
	nvcc -w bitmap.cu mmio.c -o bitmap

device: 
	nvcc -w checkDeviceInfor.cu -o device


clean:
	rm -f *.o
