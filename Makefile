
GCC = gcc
CFLAGS = -O3 -pthread

CMAIN=qsort

all: qsort-main.o qsort-pthreads.o
	$(GCC) $(CFLAGS) $^ -o $(CMAIN)

%.o: %.c
	$(GCC) -c $(CFLAGS) $^

clean:
	rm -f *.o *~ $(CMAIN)



APPS=hello

all: ${APPS}

%: %.cu
	nvcc -O2 -arch=sm_20 -o $@ $<
clean:
	rm -f ${APPS}

