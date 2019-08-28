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


#define CUDA_CALL(x)    
__global__ void compute(int* dI,int* dJ,int nz,int* col,int* out);
__global__ void findCol_ptr(int *dJ, int nz, int *col) ;

int Element(int N,int nz,int* dI,int* dJ,int* col_ptr,int Blocks,int threadsPerBlock,int* out){

    /*Initializing col_ptr array to detect empty columns*/
    CUDA_CALL(cudaMemset(col_ptr, -1, N* (sizeof(int))));

    /*Compute col_ptr array*/
    findCol_ptr<<<Blocks,threadsPerBlock>>>(dJ,nz,col_ptr);
   
    /*Compute the number of triangles fount by every nz elemnt*/
    compute<<<Blocks,threadsPerBlock>>>(dI,dJ,nz,col_ptr,out);

    /*Sum Reduce*/
    thrust::device_ptr<int> outptr(out);
    int tot = thrust::reduce(outptr, outptr + nz);
    return tot; 
    
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
