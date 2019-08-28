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

#define sharedsize 33

#define CUDA_CALL(x)                                                           \
  {                                                                            \
    if ((x) != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d\n", __FILE__, __LINE__);                     \
      printf("  %s\n", cudaGetErrorString(cudaGetLastError()));                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }



__global__ void findCol_ptr(int *dJ, int nz, int *col) {
  
 for(int i = blockIdx.x * blockDim.x + threadIdx.x+1; i<nz;i+=gridDim.x*blockDim.x){
     if(i<nz){
    int x=dJ[i];
    int y=dJ[i-1];
    if (x != y) {
      col[x] = i;
    
    if(y+1!=x){
        col[y + 1] = i;
    }}
    if(i==nz-1){
        col[x+1]=nz;
    }
    if(i==1){
    col[0]=0;

  }
  }
  } 
}

__device__ void warpReduce(volatile int* sdata,int tid ){
    sdata[tid]+=sdata[tid +32];
    sdata[tid]+=sdata[tid +16];
    sdata[tid]+=sdata[tid +8];
    sdata[tid]+=sdata[tid +4];
    sdata[tid]+=sdata[tid +2];
    sdata[tid]+=sdata[tid +1];
}
__device__ void ComputeIntersections(int len,int nz, int tid, int* blockCol,int* col,int* dI,int* s ){

        int k1;
        int k2;
        for(int j=tid;j<len;j+=blockDim.x)
        {   
            k1=j+1;
            int x=blockCol[j];
            k2=col[x];
            int r1;
            int r2;
            if(k2>0){
                int len2=col[x+1];
                r1=blockCol[k1];
                r2=dI[k2];
                while(k1<len && k2<len2 ) {
              
                    if(r1==r2){
                        s++;      
                        k1++;
                        k2++;
                        if(k2==nz || k1==len ){break;}

                        r1=blockCol[k1];
                        r2=dI[k2];

                    }else if(r1>r2){
                        k2++;
                        if(k2==nz){break;}

                        r2=dI[k2];

                    }else{
                    k1++;
                    if(k1==len ){break;}

                    r1=blockCol[k1];

                    }
                    if(k2==nz){break;}
                    
                }}


        }
}
__global__ void computeCol(int* dI,int* dJ,int nz,int* col,int* out, int N) {
    
    int s=0;
    extern __shared__ int nt[];  //results ot the threads
    __shared__ int blockCol[sharedsize]; //array that holds the column
    
    int tid=threadIdx.x;

    for(int i=blockIdx.x;i<N;i+=gridDim.x+3){
        
        int colStart=col[i];
        int len;
        if(i<N-1){
        len= col[i+1]-col[i];}
        else{len= 0;}


        /*If column is int empty*/
        if(colStart>=0 && len!=0){

            /*Fill BlockCol with the column*/
            for(int j=tid;j<len;j+=blockDim.x)
            {   
                    blockCol[j]=dI[j+colStart];
                    
            }
            __syncthreads();
            
            ComputeIntersections(len,nz,tid,blockCol,col,dI,&s);

      }
      
    }

      /*Reduce the results Unrolled the last warp*/
              nt[tid]=s;
        __syncthreads();


        //do reduction in shared mem 
        for( s=blockDim.x/2; s>32;s>>=1){
            if(tid<s){
            nt[tid]+=nt[tid+s];
            }
        __syncthreads();
        }
    if(tid<32){ warpReduce(nt,tid);}

        if(tid<32){
        out[blockIdx.x]+=nt[0];}


}

void ReadMatrix(int* M,int* N,int* nz,int* I,int* J,int argc,char** argv);


int RowElement(int N,int nz,int* dI,int* dJ,int* col_ptr,int Blocks,int threadsPerBlock,int* out){
    /*Initialize out array*/
    CUDA_CALL(cudaMemset(out, 0, Blocks* (sizeof(int))));

     /*Compute col_ptr array done in triangles-simple*/
     //findCol_ptr<<<Blocks,threadsPerBlock>>>(dJ,nz,col_ptr);
   

    /*Compute column Results*/
    computeCol<<<Blocks,threadsPerBlock,threadsPerBlock>>>(dI,dJ,nz,col_ptr,out,N);
      
    /*Sum Reduce*/
    thrust::device_ptr<int> outptr(out);
    int tot = thrust::reduce(outptr, outptr + Blocks); 
    

    return tot; 
    
}



int main(int argc, char *argv[])
{
    /*Read inputs and Matrix Data COO format*/
    int M, N, nz;   
    int *I, *J;

    ReadMatrix(&M,&N,&nz,I,J,argc,argv);
    //mm_write_banner(stdout, matcode);
    //printf("nz=%d M=%d N=%d\n",nz,M,N);

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
            printf("Halla\n");

    /*Allocate output(nt found by the threds/blocks) and col_ptr (ptrs to columns starts) arrays*/
    int* col_ptr;
    int* out;
    CUDA_CALL(cudaMalloc(&col_ptr, N*sizeof(int)));
    CUDA_CALL(cudaMalloc(&out, Blocks*sizeof(int)));
    /*Record time*/
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for data Transfers %f  ms \n", time);


/*----------Shared Memory triangle Counting Column-Element-parallelization-----*/
    /*Start time counting with CUDA envents*/
    cudaEventRecord(start, 0);

    int tot=RowElement(N,nz,dI,dJ,col_ptr,Blocks,threadsPerBlock,out);//triangles-shared.cu
    
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
