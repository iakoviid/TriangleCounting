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
    if(i==0){
    col[0]=0;

  }
  }
  } 
}
__global__ void InitCol_ptr(int len, int *col) {
  for(int i = blockIdx.x * blockDim.x + threadIdx.x+1; i<len;i+=gridDim.x*blockDim.x){
  if(i<len){
        col[i]=-1;

  }}
}

#define sharedsize 65

__global__ void computeRow2(int* dI,int* dJ,int nz,int* col,int* out, int N) {
    int i=blockIdx.x;
    while(i<N){
        //if(threadIdx.x==0 && blockIdx.x==0){
        //printf("blockIdx=%d\n",blockIdx.x );}
        __shared__ int blockCol[sharedsize];//len of column
        __shared__ int nt[sharedsize];  
        int colStart=col[i];
        int len;
        if(i<N-1){
        len= col[i+1]-col[i];}
        else{len= 0;}
        int tid=threadIdx.x;
        if(colStart<0 || len==0){
            if(tid<32){
            out[i]=0;}
            return;
        }

      
        for(int j=tid;j<len;j+=blockDim.x)
        {   
            if(dJ[j+colStart]==i){
                blockCol[j]=dI[j+colStart];
                }
        }
          __syncthreads();

      
         int k1;
         int k2;
         int s;
        for(int j=tid;j<len;j+=blockDim.x)
        {   s=0;
            k1=0;
            int x=blockCol[j];
            k2=col[x];
            int r1;
            int r2;
            if(k2>0){
                int len2=col[x+1];
                r1=blockCol[k1];
                r2=dI[k2];
                while(k1<len && k2<len2 ) {
                //    if(threadIdx.x==0&& i ==0){
                //       printf("r1=%d,r2=%d\n",r1,r2 );
                //        printf("k1=%d,k2=%d\n",k1,k2 );
                //    }
                    if(r1==r2){
                        s++;      
                        k1++;
                        k2++;

                    }else if(r1>r2){
                        k2++;

                    }else{
                    k1++;
                    }
                    if(k2==nz){break;}
                    r1=blockCol[k1];
                    r2=dI[k2];
                }
            }

            nt[j]=s;
        }
        __syncthreads();
        
        for(int j=tid+blockDim.x;j<len;j+=blockDim.x){
        nt[tid]=nt[tid]+nt[j];
        
        }
        __syncthreads();


        //do reduction in shared mem 
        for( s=blockDim.x/2; s>0;s>>=1){
            if(tid<s){
            nt[tid]+=nt[tid+s];
            }
        __syncthreads();
        }

        if(tid<32){
        out[i]=nt[0];}
      
        i+=gridDim.x;
      }
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
    int Blocks=atoi(argv[3]);    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    //CUDA_CALL(cudaMemset(col, -1, N* (sizeof(int))));

    InitCol_ptr<<<ceil(N/threadsPerBlock), Blocks,threadsPerBlock>>>(N,col);
    findCol_ptr<<<ceil(nz/threadsPerBlock), Blocks,threadsPerBlock>>>(dJ,nz,col);
    //colLengths<<<ceil(N/threadsPerBlock), threadsPerBlock>>>(N,col);


    computeRow2<<<ceil(N/Blocks), Blocks,threadsPerBlock>>>(dI,dJ,nz,col,out,N);
      
    thrust::device_ptr<int> outptr(out);
    int tot = thrust::reduce(outptr, outptr + N); 
    

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("%f  ms ", time);





    printf(" Trianles =  %d\n",tot );





    CUDA_CALL(cudaFree(out));
    CUDA_CALL(cudaFree(dI));
    CUDA_CALL(cudaFree(dJ));
    CUDA_CALL(cudaFree(col));


    return 0;
}

