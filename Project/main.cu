#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "mmio.c"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <time.h>


#define CUDA_CALL(x)                                                           \
  {                                                                            \
    if ((x) != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d\n", __FILE__, __LINE__);                     \
      printf("  %s\n", cudaGetErrorString(cudaGetLastError()));                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

void ReadMatrix(int* M,int* N,int* nz,int* I,int* J,int argc,char** argv);
int Element(int N,int nz,int* dI,int* dJ,int* col_ptr,int Blocks,int threadsPerBlock,int* out);
int RowElement(int N,int nz,int* dI,int* dJ,int* col_ptr,int Blocks,int threadsPerBlock,int* out);



int main(int argc, char *argv[])
{
    /*Read inputs and Matrix Data COO format*/
    int M, N, nz;   
    int *I, *J;

    ReadMatrix(&M,&N,&nz,I,J,argc,argv);
    //mm_write_banner(stdout, matcode);
    printf("nz=%d M=%d N=%d\n",nz,M,N);

    int threadsPerBlock=atoi(argv[2]);
    int Blocks=atoi(argv[3]);    
    
  /*--------Allocate device data and Transfer---------*/
    /*Start time counting with CUDA envents*/
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int* dI;
    int* dJ;
    CUDA_CALL(cudaMalloc(&dI, nz*sizeof(int)));
    CUDA_CALL(cudaMalloc(&dJ, nz*sizeof(int)));
    cudaMemcpy(dI, I, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dJ, J, nz*sizeof(int), cudaMemcpyHostToDevice);
    
    /*Allocate output(nt found by the threds/blocks) and col_ptr (ptrs to columns starts) arrays*/
    int* col_ptr;
    int* out;
    CUDA_CALL(cudaMalloc(&col_ptr, N*sizeof(int)));
    CUDA_CALL(cudaMalloc(&out, nz*sizeof(int)));
    /*Record time*/
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for data Transfers %f  ms \n", time);


 /*------------Simple triangle Counting element-parallelization---------------*/  
    /*Start time counting with CUDA envents*/
    cudaEventRecord(start, 0);
    int tot=Element(N,nz,dI,dJ,col_ptr,Blocks,threadsPerBlock,out); //triangles-simple.cu
    
    /*Record time*/
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Element parallelization Complited %f  ms ", time);
    printf("Number of Trianles = %d\n",tot );

/*----------Shared Memory triangle Counting Column-Element-parallelization-----*/
    /*Start time counting with CUDA envents*/
    cudaEventRecord(start, 0);

    tot=RowElement(N,nz,dI,dJ,col_ptr,Blocks,threadsPerBlock,out);//triangles-shared.cu
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    printf("Column-Element parallelization Complited %f  ms ", time);
    printf("Number of Trianles = %d\n",tot );






    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CALL(cudaFree(out));
    CUDA_CALL(cudaFree(dI));
    CUDA_CALL(cudaFree(dJ));
    CUDA_CALL(cudaFree(col_ptr));


    return 0;
}

    

void ReadMatrix(int* M,int* N,int* nz,int* I,int* J,int argc,char** argv){

    int i;
    int ret_code;
        MM_typecode matcode;
        FILE *f;
        
        if (argc < 4)
        {
            fprintf(stderr, "Usage: %s [martix-market-filename] [threadsPerBlock] [numberOfBlocks]\n", argv[0]);
            exit(1);
        }
        else    
        { 
            if ((f = fopen(argv[1], "r")) == NULL) 
                exit(1);
        }

        if (mm_read_banner(f, &matcode) != 0)
        {
            printf("Could not process Matrix Market banner.\n");
            exit(1);
        }

        if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
                mm_is_sparse(matcode) )
        {
            printf("Sorry, this application does not support ");
            printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
            exit(1);
        }

        /* find out size of sparse matrix .... */

        if ((ret_code = mm_read_mtx_crd_size(f, M, N, nz)) !=0)
            exit(1);

    /* reseve memory for matrices */

    I = (int *) malloc(*nz * sizeof(int));
    J = (int *) malloc(*nz * sizeof(int));
  

    for (i=0; i<*nz; i++)
    {
        fscanf(f, "%d %d\n", &I[i], &J[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f !=stdin) fclose(f);



}
