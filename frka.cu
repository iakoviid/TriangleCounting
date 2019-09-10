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
#define sharedsize 256
__device__ int ComputeIntersection(int* blockCol,int* col,int len,int nz,int* dI,int j){    
        int k1;
        int k2;
        int s=0;
       
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
                    
                }}


        
        return s;
    }

//Make lens nt something
__global__ void computeRow2(int* dI,int* dJ,int nz,int* col,int* out, int N,int k,int* out2) {
    int s=0;
    int j;
    extern __shared__ int nt[];  
    __shared__ int blockCol[sharedsize];

    int tid=threadIdx.x;

    for(int i=blockIdx.x;i<N;i+=gridDim.x){

        //if(k*i>409995){
        //    if(tid==0 && blockIdx.x==0){
        //    printf("kaka\n");
        //}
        //}
        if(k*i<= 391238 && k*i+k-1>= 391238){
            if(tid==0){
            printf("Saka");}
        }
        int a=0;
        int b=-1;
        int len;
        int colStart;
        for(j=0;j<k;j++){

        
        if(k*i+j<N-1){

        colStart=col[k*i+j];
        len= col[k*i+1+j]-colStart;
        }
        else{len= 0;}

        if(colStart>=0 && len>0){
            if(tid==0){
            nt[j]=len;}
            //if(i==gridDim.x+1 && tid==0){printf("len=%d nt[%d]=%d\n",len,j,nt[j] );}
            a=a+len;
            if(b==-1){

                b=colStart;
            }
        }else{ 
            len=0;
             if(tid==0){
            nt[j]=len;}

            //if(i==gridDim.x+1 && tid==0){printf("Empty len=%d nt[%d]=%d\n",len,j,nt[j] );}

        }
        }
        colStart=b;
        for( j=tid;j<a;j+=blockDim.x)
        {       


                blockCol[j]=dI[j+colStart];
                
        }
          __syncthreads();


     for( j=tid;j<a;j+=blockDim.x)
        {   //if(i==gridDim.x+1 &&j==0 ){
            //for(int x=0;x<k;x++){printf("nt[%d]=%d\n",x,nt[x]);}
            //for(int x=0;x<a;x++){printf("dJ[%d+colStart]=%d\n",x,dJ[x+colStart] ); }
            //for(int x=0;x<a;x++){printf("dJ[%d+colStart]%k=%d\n",x,dJ[x+colStart]%k ); }
                
           // }
            //if (i==gridDim.x+1)
            //{
            //    for(int x=0;x<k;x++){printf( "colStart=%d tid=%d nt[%d]=%d\n",colStart,tid,x,nt[x]);}
            //}
        //len=col[dJ[j+colStart]+1]-col[dJ[j+colStart]];
        //if(i==gridDim.x+1 &&j==0 ){
          //  printf("----------------------------------------\n");
            //printf("len =%d \n",len );
                
            //}
            
            b=0;
            for(int x=0;x<k;x++){
                b=b+nt[x];
                if(j<b){break;}
            }
        s=s+ComputeIntersection(blockCol,col,/*nt[dJ[j+colStart]%k]*/b ,nz, dI, j);
            out2[j+colStart]=ComputeIntersection(blockCol,col,/*nt[dJ[j+colStart]%k]*/b ,nz, dI, j);

            if (k*i<= 391238 && k*i+k-1>= 391238){
                if(tid==0){
                printf(" NT =%d %d %d %d %d %d %d %d \n",nt[0],nt[1],nt[2],nt[3],nt[4],nt[5],nt[6],nt[7] );}
            }
            if (k*i<= 391238 && k*i+k-1>= 391238){
                printf("out2[%d]=%d\n",j+colStart,out2[j+colStart] );
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


void findColStart2(int *dJ, int nz, int *col) {
  for(int i=1;i<nz;i++){
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
void compute(int* dI,int* dJ,int nz,int* col,int* out,int N){
  for(int i=0;i<nz;i++){  
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
        

    }
 }


    out[i]=s;
  }
}}

int reduce(int* out,int nz){
 int s=0;
    for(int i =0;i<nz;i++){
        s=s+out[i];
        //if(i<50){
        //printf("out[%d]=%d\n",i,out[i]);}
    }
    return s;
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
    int k=atoi(argv[4]);
    int threadsPerBlock=atoi(argv[2]);
    int Blocks=atoi(argv[3]);
    int* dI;
    int* dJ;
    int* col;
    int* out;
    int* out2;
    CUDA_CALL(cudaMalloc(&dI, nz*sizeof(int)));
    CUDA_CALL(cudaMalloc(&dJ, nz*sizeof(int)));
    CUDA_CALL(cudaMalloc(&col, N*sizeof(int)));
    CUDA_CALL(cudaMalloc(&out, Blocks*sizeof(int)));
    CUDA_CALL(cudaMalloc(&out2, nz*sizeof(int)));


    cudaMemcpy(dI, I, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dJ, J, nz*sizeof(int), cudaMemcpyHostToDevice);
  
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    CUDA_CALL(cudaMemset(col, -1, N* (sizeof(int))));
    CUDA_CALL(cudaMemset(out, 0, Blocks* (sizeof(int))));


    findCol_ptr<<<Blocks,threadsPerBlock>>>(dJ,nz,col);
    

    computeRow2<<<Blocks,threadsPerBlock,threadsPerBlock*sizeof(int)>>>(dI,dJ,nz,col,out,N,k,out2);
      
    
    thrust::device_ptr<int> outptr(out);
    int tot =thrust::reduce(outptr, outptr + Blocks); 
    

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("%f  ms ", time);





    printf(" Trianles =  %d\n",tot );
    int* elem=(int *)malloc(nz*sizeof(int));

    cudaMemcpy(elem, out2,nz*sizeof(int), cudaMemcpyDeviceToHost);
    //for(int i=0;i<50;i++){
    //    printf("out2[%d]=%d\n",i,elem[i] );
    //}

    CUDA_CALL(cudaFree(out2));

    CUDA_CALL(cudaFree(out));
    CUDA_CALL(cudaFree(dI));
    CUDA_CALL(cudaFree(dJ));
    CUDA_CALL(cudaFree(col));


    int* col_ptr=(int *)malloc(N*sizeof(int));
    int* out3=(int *)malloc(nz*sizeof(int));
    findColStart2(J,nz,col_ptr);
    compute(I,J,nz,col_ptr,out3,N);
    printf("%d \n",reduce(elem,nz));
     printf("Errors poy\n");

     for(int i=3114318 ;i<nz;i++){
         //printf("i=%d  out3=%d elem=%d COL[i]=%d\n",i,out3[i],elem[i],J[i] );
        if(out3[i]!=elem[i]){
       // printf("saka\n");

           printf("i=%d  out3=%d elem=%d COL[i]=%d\n",i,out3[i],elem[i],J[i] );
        }
     }










    return 0;
}

