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

int klength(int* col,int N,int k){
    int maxlen=0;
    int len;
    for(int i=0;i<N/k+1;i++){
        int a=0;
        for(int j=0;j<k;j++){
            if(k*i+k<N){
        len=col[k*i+1+j]-col[k*i+j];
    
        if(col[i*k+j]>=0 && len>0){
        a=a+len;
            if(a>maxlen){
                maxlen=a;
            }
        }}
    }
    }
    return maxlen;
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
    int k=atoi(argv[2]);

    InitColStart(N,col);
    findColStart(J,nz,col);
    int maxlen=klength(col,N,k);
    printf("====================================================================\n");
    printf("maxlen=%d\n",maxlen );
    //int* out2=(int*)malloc(N*sizeof(int));
    //computeRow(I,J,nz,col,out2,N);
    //int d=reduce(out2,N);
    //printf("%d\n",d );
    for(int i=0;i<50;i++){
        //printf("out[%d]=%d\n",i,out[i] );
    }
    printf("dJ[%d]=%d\n",137,J[137] );
    //for(int i;i< 20;i++){printf("%d-",out2[i] );}



    return 0;
}

