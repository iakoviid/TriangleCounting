#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"

void findColStart(int *dJ, int len, int *col) {
  for(int i=0;i<len;i++){
  if(i<len){
  if (i > 0 ) {
    if (dJ[i] != dJ[i - 1]) {
      col[dJ[i]] = i;
    }
  } else {
    col[dJ[0]] = 0;
  }}
    }
}

 void colLengths(int len, int *col) {
for(int i=0;i<len ;i++ ){
  if(i<len-1){
        int s=i;
        while(col[s]==-1){
                    if(i==448627){
                        printf("s=%d\n",s );}

            s++;
            if(s==len-1){
                printf("Lexit %d\n",i);
                break;}
        }
        col[i]=col[s];

  }}
}

 void InitColStart(int len, int *col) {
  for(int i=0;i<len;i++){
  if(i<len){
        col[i]=-1;

  }}
}

 void compute(int* dI,int* dJ,int nz,int* col,int* out) {
  for(int i=0;i<nz;i++){
  if(i<nz){
    int s=0;
    int x=dI[i];
    int y=dJ[i];
    int k1=col[x];
    int k2=col[y];
    int r1;
    int r2;
    if(k1>=0){
    while(dJ[k1]==x && dJ[k2]==y ) {
        if(i==2447544){
            printf("k1=%d k2=%d\n",k1,k2 );
                  
        }
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
             if(i==2447544){
            printf("x=%d y=%d  ",x,y );
            printf("r1=%d r2=%d  ",r1,r2 );
            printf("k1=%d k2=%d\n",k1,k2 );
                  
        }
        

    }}



    out[i]=s;
  }}
}
int reduce(int* out,int nz){
 int s=0;
    for(int i =0;i<nz;i++){
        s=s+out[i];
        
    }
    return s;
}

void compute2(int* dI,int* dJ,int nz,int* col,int* out){
  for(int i=0;i<nz;i++){
    if(i==2447544){
            printf("break\n");
        }
    if(i<nz){
    int s=0;
    int x=dI[i];
    int y=dJ[i];
    int k1=col[x];
    int k2=col[y];
    int r1;
    int r2;
    if(k1>=0){
    int len1=col[x+1];
    int len2=col[y+1];

    while(k1<len1 && k2<len2 ) {
        if(i==2447544){
            printf("k1=%d k2=%d\n",k1,k2 );
                  
        }
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
     if(i==2447544){
            printf("x=%d y=%d  ",x,y );
            printf("r1=%d r2=%d  ",r1,r2 );
            printf("k1=%d k2=%d\n",k1,k2 );
                  
        }
    }
    }
    out[i]=s;
  }
}}
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

    

    mm_write_banner(stdout, matcode);


    printf("nz=%d M=%d N=%d\n",nz,M,N);
  
    int* col=(int *)malloc(N* sizeof(int));
    int* out=(int*)malloc(nz*sizeof(int));
    int* out2=(int*)malloc(nz*sizeof(int));

    InitColStart(N,col);
    findColStart(J,nz,col);
    compute(I,J,nz,col,out) ;
    int s=reduce(out,nz);
    printf("%d\n",s );
    colLengths(N,col);
    compute2(I,J,nz,col,out2);
    for(int i=0;i<nz;i++){
        if(out[i]!=out2[i]){
           printf("error i=%d\n",i );
        }
    }
    printf("%d\n",s );
    s=reduce(out2,nz);
    printf("%d\n",s );



	return 0;
}

