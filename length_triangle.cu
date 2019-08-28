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

__global__ void findColStart(int *dJ, int nz, int *col) {
  
 for(int i = blockIdx.x * blockDim.x + threadIdx.x+1; i<nz;i+=gridDim.x*blockDim.x){
     if(i<nz){
    int a=dJ[i];
    int b=dJ[i-1];
    if (a != b) {
      col[a] = i;
    
    if(b+1!=a){
        col[b + 1] = i;
    }}
    if(i==nz-1){
        col[a+1]=nz;
    }
    if(i==1){
    col[0]=0;

  }
  }
  } 
}


__global__ void InitColStart(int len, int *col) {
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i<len;i+=gridDim.x*blockDim.x){
  if(i<len){
        col[i]=-1;

  }}
}

__global__ void compute(int* dI,int* dJ,int nz,int* col,int* out) {
 for(int i = blockIdx.x * blockDim.x + threadIdx.x; i<nz;i+=gridDim.x*blockDim.x){

  if(i<nz){
    int s=0;
    int x=dI[i];
    int y=dJ[i];
    int k1=col[x];
    int k2=col[y];
    int r1;
    int r2;
    if(k1>0){
    int len1=col[x+1];
    int len2=col[y+1];
    while(k1<len1 && k2<len2 ) {
        if(k1>=nz || k2>=nz ){break;}
        r1=dI[k1];
        r2=dI[k2];
        if(r1==r2){
            s++;      
            k1++;
            k2++;
        }else if(r1>r2){
            k2++;

        }else{
            k1++;
        }
        

    }}



    out[i]=s;
  }}
}
int main(int argc, char *argv[])
{
    int ret_code;
        MM_typecode matcode;
        FILE *f;
        int M, N, nz;   
        int i, *I, *J;

        if (argc < 3)
    	{
    		fprintf(stderr, "Usage: %s [martix-market-filename] threadsPerBlock\n", argv[0]);
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

        if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
            exit(1);

    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
  

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d\n", &I[i], &J[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f !=stdin) fclose(f);

    //mm_write_banner(stdout, matcode);
    //printf("nz=%d M=%d N=%d\n",nz,M,N);
    
    int* dI;
    int* dJ;
    int* col;
    int* out;
    CUDA_CALL(cudaMalloc(&dI, nz*sizeof(int)));
    CUDA_CALL(cudaMalloc(&dJ, nz*sizeof(int)));
    CUDA_CALL(cudaMalloc(&col, N*sizeof(int)));
    CUDA_CALL(cudaMalloc(&out, nz*sizeof(int)));



    cudaMemcpy(dI, I, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dJ, J, nz*sizeof(int), cudaMemcpyHostToDevice);


    int threadsPerBlock=atoi(argv[2]);
    int Blocks=atoi(argv[3]);
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    CUDA_CALL(cudaMemset(col, -1, N* (sizeof(int))));

    //InitColStart<<< Blocks,threadsPerBlock>>>(N,col);
    findColStart<<< Blocks,threadsPerBlock>>>(dJ,nz,col);


    compute<<<Blocks,threadsPerBlock>>>(dI,dJ,nz,col,out);
      
    thrust::device_ptr<int> outptr(out);
    int tot = thrust::reduce(outptr, outptr + nz); 
    

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Time for Counting triangles  %f  ms \n", time);





    printf("The sum is  %d\n",tot );





    CUDA_CALL(cudaFree(out));
    CUDA_CALL(cudaFree(dI));
    CUDA_CALL(cudaFree(dJ));
    CUDA_CALL(cudaFree(col));


	return 0;
}

