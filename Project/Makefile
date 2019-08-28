CC=nvcc
NVCCLFLAGS= -arch=sm_37 -std=c++11 -cudart=shared -rdc=true
CFLAGS=--device-c -w

all: triangles

triangles: main.o triangles-bitmap.o triangles-simple.o triangles-shared.o mmio.o
	$(CC) --device-link -w $(NVCCLFLAGS) main.o triangles-bitmap.o triangles-simple.o triangles-shared.o mmio.o -o triangles

main.o: main.cu
	$(CC) $(CFLAGS) $(NVCCLFLAGS) main.cu

triangles-bitmap.o: triangles-bitmap.cu
	$(CC) $(CFLAGS) $(NVCCLFLAGS) triangles-bitmap.cu

triangles-simple.o: triangles-simple.cu
	$(CC) $(CFLAGS) $(NVCCLFLAGS) triangles-simple.cu

triangles-shared.o: triangles-shared.cu
	$(CC) $(CFLAGS) $(NVCCLFLAGS) triangles-shared.cu

mmio.o:	mmio.c
	$(CC) $(CFLAGS) $(NVCCLFLAGS) mmio.c

clean:
	rm -rf *o triangles