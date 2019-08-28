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
/*Initialize each element of the column pointer to -1 so that we can distinguish the empty columns*/
__global__ void Init(int N, int *col,int* out) {
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i<N;i+=gridDim.x*blockDim.x){
  if(i<N){
        col[i]=-1;
  }
  if(i<gridDim.x){
    out[i]=0;}

}


}

#define sharedsize 65

__global__ void computeRow2(int* dI,int* dJ,int nz,int* col,int* out, int N,int* bitmap) {
    int s=0;
    __shared__ int nt[64];  
    int tid=threadIdx.x;
    __shared__ int blockCol[sharedsize];//len of column
    int a;
     
    bitmap=bitmap+blockIdx.x*N/32;
    for(int i=blockIdx.x;i<N;i+=gridDim.x){

        //if(threadIdx.x==0 && blockIdx.x==0){
        //printf("blockIdx=%d\n",blockIdx.x );}
        int colStart=col[i];
        int len;
        if(i<N-1){
        len= col[i+1]-col[i];}
        else{len= 0;}
        if(colStart<0 || len==0){
          

        }else{

        for(int j=tid;j<N;j+=blockDim.x)
        {   
                
                ClearBit(bitmap,j);

                
        }
          __syncthreads();


             
        for(int j=tid;j<len;j+=blockDim.x)
        {   
                a=dI[j+colStart];
                SetBit(bitmap,a);
                blockCol[j]=a;

                
        }
          __syncthreads();

        for(a=0;a<len;a++){
            int x=blockCol[a];
            int k=col[x];
            if(k>0){
                int len2=col[x+1];
                for(int j=tid+k;j<len2;j+=blockDim.x)
                 {   
                    if(TestBit(bitmap,dI[j])){s++;}
                 }
            }
        } 
        


      }
      
      }
              nt[tid]=s;
        __syncthreads();


        //do reduction in shared mem 
        for( s=blockDim.x/2; s>0;s>>=1){
            if(tid<s){
            nt[tid]+=nt[tid+s];
            }
        __syncthreads();
        }

        if(tid<32){
        out[blockIdx.x]+=nt[0];}


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
    
    int threadsPerBlock=atoi(argv[2]);
    int Blocks=atoi(argv[3]);    
    int* dI;
    int* dJ;
    int* col;
    int* out;
    int* bitmap;
    CUDA_CALL(cudaMalloc(&dI, nz*sizeof(int)));
    CUDA_CALL(cudaMalloc(&dJ, nz*sizeof(int)));
    CUDA_CALL(cudaMalloc(&col, N*sizeof(int)));
    CUDA_CALL(cudaMalloc(&out, nz*sizeof(int)));
    CUDA_CALL(cudaMalloc(&bitmap, N* Blocks));


    cudaMemcpy(dI, I, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dJ, J, nz*sizeof(int), cudaMemcpyHostToDevice);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    //CUDA_CALL(cudaMemset(col, -1, N* (sizeof(int))));

    Init<<</*ceil(N/threadsPerBlock),*/ Blocks,threadsPerBlock>>>(N,col,out);
    findCol_ptr<<</*ceil(nz/threadsPerBlock),*/ Blocks,threadsPerBlock>>>(dJ,nz,col);
    //colLengths<<<ceil(N/threadsPerBlock), threadsPerBlock>>>(N,col);


    computeRow2<<</*ceil(N/Blocks), */Blocks,threadsPerBlock>>>(dI,dJ,nz,col,out,N,bitmap);
      
    
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

