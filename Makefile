CC=nvcc

CFLAGS=-c -w

all: triangles

triangles: main.o triangles-bitmap.o triangles-simple.o triangles-shared.o mmio.o
	$(CC) main.o triangles-bitmap.o triangles-simple.o triangles-shared.o -o triangles

main.o: main.cu
	$(CC) $(CFLAGS) main.cu

triangles-bitmap.o: triangles-bitmap.cu
	$(CC) $(CFLAGS) triangles-bitmap.cu

triangles-simple.o: triangles-simple.cu
	$(CC) $(CFLAGS) triangles-simple.cu

triangles-shared.o: triangles-shared.cu
	$(CC) $(CFLAGS) triangles-shared.cu

mmio.o:	mmio.c
	$(CC) $(CFLAGS) mmio.c
clean:
	rm -rf *o triangles