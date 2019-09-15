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
#define SetBit(A,k)     ( A[(k/32)] |= (1 << (k%32)) )         
#define ClearBit(A,k)   ( A[(k/32)] &= ~(1 << (k%32)) )
#define TestBit(A,k)    ( A[(k/32)] & (1 << (k%32)) )
#define CUDA_CALL(x)                                                           \
  {                                                                            \
    if ((x) != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d\n", __FILE__, __LINE__);                     \
      printf("  %s\n", cudaGetErrorString(cudaGetLastError()));                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }




/*Making the col_ptr array st if i'th column is nonempty then col_ptr[i]=Start of i'th column 
and col_ptr[i+1]-col_ptr[i] = length of i'th column*/
__global__ void findCol_ptr(int *dJ, int nz, int *col_ptr) {
  
 for(int i = blockIdx.x * blockDim.x + threadIdx.x+1; i<nz;i+=gridDim.x*blockDim.x){
     if(i<nz){
    int x=dJ[i];
    int y=dJ[i-1];
    if (x != y) {
      col_ptr[x] = i;
    
    if(y+1!=x){
        col_ptr[y + 1] = i;
    }}
    if(i==nz-1){
        col_ptr[x+1]=nz;
    }
    if(i==1){
    col_ptr[0]=0;

  }
  }
  } 
}

/*Unrolled sum reduction of a Warp*/
__device__ void warpReduce(volatile int* sdata,int tid ){
    sdata[tid]+=sdata[tid +32];
    sdata[tid]+=sdata[tid +16];
    sdata[tid]+=sdata[tid +8];
    sdata[tid]+=sdata[tid +4];
    sdata[tid]+=sdata[tid +2];
    sdata[tid]+=sdata[tid +1];
}

__global__ void computeCol(int* dI,int* dJ,int nz,int* col,int* out, int N,int* bitmap) {
    int s=0;
     extern __shared__ int nt[];  
     
    int tid=threadIdx.x;
    int a;
    int off;
    int pos ;
    unsigned int flag; 
     
    //bitmap=bitmap+blockIdx.x*N;
    for(int i=blockIdx.x;i<N;i+=gridDim.x){

        //if(threadIdx.x==0 && blockIdx.x==0){
        //printf("blockIdx=%d\n",blockIdx.x );}
        int colStart=col[i];
        int len;
        if(i<N-1){
        len= col[i+1]-colStart;}
        else{len= 0;}
        
        if(colStart>=0 && len>0){
            
        for(int j=tid;j<N;j+=blockDim.x)
        {   
                //Write it better no bit bit
                
                 off = (j+blockIdx.x*N)/32;
                 pos = (j+blockIdx.x*N)%32;
                 flag = 1;  // flag = 0000.....00001
                 flag = flag << pos;     // flag = 0000...010...000   (shifted k positions)
                 flag = ~flag;           // flag = 1111...101..111
                bitmap[off] = bitmap[off] & flag;     // RESET the bit at the k-th position in A[i

                
        }
          __syncthreads();


             
        for(int j=tid;j<len;j+=blockDim.x)
        {   
                a=dI[j+colStart];


                 off = (a+blockIdx.x*N)/32;            //array index 
                  pos = (a+blockIdx.x*N)%32;          // pos = bit position in bitmap[off]
                flag = 1;   // flag = 0000.....00001
                 flag = flag << pos;      // flag = 0000...010...000   (shifted k positions)    
                bitmap[off] = bitmap[off] | flag;      // Set the bit at the k-th position in bitmap[i]

                nt[j]=a;

                
        }
          __syncthreads();

        for(a=0;a<len;a++){
            int x=nt[a];
            int k=col[x];
            if(k>0){
                int len2=col[x+1];
                for(int j=tid+k;j<len2;j+=blockDim.x)
                 {   
                      off = (dI[j]+blockIdx.x*N)/32;
                      pos = (dI[j]+blockIdx.x*N)%32;
                      flag = 1;  // flag = 0000.....00001
                      flag = flag << pos;     // flag = 0000...010...000   (shifted k positions)
                if ( bitmap[off] & flag )      // Test the bit at the k-th position in A[i]
                    {
                        s++;}
                 }
            }
        } 
        


      }
      
      }
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


void ReadMatrix(int* M,int* N,int* nz,int** I,int** J,int argc,char** argv){

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

    *I = (int *) malloc(*nz * sizeof(int));
    *J = (int *) malloc(*nz * sizeof(int));
  

    for (i=0; i<*nz; i++)
    {
        fscanf(f, "%d %d\n", &(*I)[i], &(*J)[i]);
        (*I)[i]--;  /* adjust from 1-based to 0-based */
        (*J)[i]--;
    }

    if (f !=stdin) fclose(f);
}



int main(int argc, char *argv[])
{
    
    int M, N, nz;   
    int *I, *J;

    ReadMatrix(&M,&N,&nz,&I,&J,argc,argv);

   //mm_write_banner(stdout, matcode);
    //mm_write_banner(stdout, matcode);
    //printf("nz=%d M=%d N=%d\n",nz,M,N);
    
    int threadsPerBlock=atoi(argv[2]);
    int Blocks=atoi(argv[3]);  
    int length=atoi(argv[4]);
  
    int* dI;
    int* dJ;
    int* col;
    int* out;
    int* bitmap;
    CUDA_CALL(cudaMalloc(&dI, nz*sizeof(int)));
    CUDA_CALL(cudaMalloc(&dJ, nz*sizeof(int)));
    CUDA_CALL(cudaMalloc(&col, N*sizeof(int)));
    CUDA_CALL(cudaMalloc(&out, Blocks*sizeof(int)));
    CUDA_CALL(cudaMalloc(&bitmap, N* Blocks));


    cudaMemcpy(dI, I, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dJ, J, nz*sizeof(int), cudaMemcpyHostToDevice);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    CUDA_CALL(cudaMemset(col, -1, N* (sizeof(int))));
    CUDA_CALL(cudaMemset(out, 0, Blocks* (sizeof(int))));

    findCol_ptr<<< Blocks,threadsPerBlock>>>(dJ,nz,col);

     length=max(threadsPerBlock,length);
    computeCol<<<Blocks,threadsPerBlock,length*sizeof(int)>>>(dI,dJ,nz,col,out,N,bitmap);

      
    
    thrust::device_ptr<int> outptr(out);
    int tot = thrust::reduce(outptr, outptr + Blocks); 
    

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("%f  ms ", time);





    printf(" Trianles =  %d\n",tot );




    CUDA_CALL(cudaFree(bitmap));
    CUDA_CALL(cudaFree(out));
    CUDA_CALL(cudaFree(dI));
    CUDA_CALL(cudaFree(dJ));
    CUDA_CALL(cudaFree(col));


    return 0;
}

