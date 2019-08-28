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
#define sharedsize 33
__global__ void computeCol(int* dI,int* dJ,int nz,int* col,int* out, int N);


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

    for(int i=blockIdx.x;i<N;i+=gridDim.x){
        
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



