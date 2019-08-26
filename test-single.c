#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"

void findColStart(int *dJ, int nz, int *col) {
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




 void InitColStart(int len, int *col) {

  for(int i=0;i<len;i++){
  
  if(i<len){
        col[i]=-1;

  }
}
}
/*
#define sharedsize 64

 void computeRow(int* dI,int* dJ,int nz,int* col,int* out, int N) {
    int blockCol[sharedsize];//len of column
    int nt[sharedsize];
    for(int i=0;i<N;i++){
        if(i%1000==0){
        printf("%d\n",i );}
        if(i>448600){

        printf("%d\n",i );
        }
        //if(threadIdx.x==0 && blockIdx.x==0){
        //printf("blockIdx=%d\n",blockIdx.x );}
            
        int colStart=col[i];
        int len;
       
       
        if(i<N-1){
        len= col[i+1]-col[i];}
        else{len= 0;}
        if(colStart<0 || len==0){
          
            out[i]=0;

            //return;
        }else{

        for(int tid=0;tid<256;tid++){

        for(int j=tid;j<len;j+=256)
        {   
                blockCol[j]=dI[j+colStart];
                
        }}
         int k1;
         int k2;
         int s;
        for(int tid=0;tid<256;tid++){

        
        for(int j=tid;j<len;j+=256)
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
        }}
        for(int tid=0;tid<256;tid++){

        for(int j=tid+256;j<len;j+=256){
        nt[tid]=nt[tid]+nt[j];
        
        }
        }

        for( s=len/2; s>0;s>>=1){
            for(int tid=0;tid<256;tid++){

            if(tid<s && s<len){
            nt[tid]+=nt[tid+s];
                }
        }  }
            for(int tid=0;tid<256;tid++){

        if(tid<32){
        out[i]=nt[0];}
      }
      }}
   
       
}
*/


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
 if(dJ[i]%7010==7009){
    printf("Ready poy\n");
 }
 if(dJ[i]%7010==0){
    printf("Hala\n");
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

        if (argc < 2)
        {
            fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
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
  
    int* col=(int *)malloc(N* sizeof(int));
    int* out=(int*)malloc(nz*sizeof(int));
    InitColStart(N,col);
    findColStart(J,nz,col);
    //colLengths(N,col);
    compute(I,J,nz,col,out,N);
    int s=reduce(out,nz);
    printf("%d\n",s );

    InitColStart(N,col);
    findColStart(J,nz,col);
    //int* out2=(int*)malloc(N*sizeof(int));
    //computeRow(I,J,nz,col,out2,N);
    //int d=reduce(out2,N);
    //printf("%d\n",d );
    s=0;
    for(int i;i< nz;i++){
        if(i%7010==0){
        s=s+out[i];
        printf("out  %d , s=%d\n",out[i],s );}
    }
    printf("\n");
    printf("\n");

    //for(int i;i< 20;i++){printf("%d-",out2[i] );}



    return 0;
}

