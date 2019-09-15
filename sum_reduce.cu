
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#define blockSize 128
__global__ void reduce0(int *g_idata,int *g_odata){
extern __shared__ int sdata[];

//each thread loads one element from global to shared mem
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
sdata[tid]=g_idata[i];
//printf("%d- ",sdata[tid]);
__syncthreads();
//if(threadIdx.x==0){
//printf("\n");}

//do reduction in shared mem 
for(unsigned int s=1 ;s<blockDim.x; s*=2){
	if(tid%(2*s)==0){
		//if(tid==0){
		//printf("Level %d Thread %d Data %d + Data %d \n",s,tid,sdata[tid],sdata[tid+s]);}
		sdata[tid]+=sdata[tid+s];
	}
	__syncthreads();
}
// write the result for this block to global mem
if(tid ==0){ 
	g_odata[blockIdx.x]=sdata[0];
}

}

__global__ void reduce1(int *g_idata,int *g_odata){
extern __shared__ int sdata[];

//each thread loads one element from global to shared mem
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
sdata[tid]=g_idata[i];
//printf("%d- ",sdata[tid]);
__syncthreads();
//if(threadIdx.x==0){
//printf("\n");}

//do reduction in shared mem 
for(unsigned int s=1 ;s<blockDim.x; s*=2){
	int index=2*s*tid;
	if(index<blockDim.x){
		sdata[index]+=sdata[index+s];
	}
	__syncthreads();
}
// write the result for this block to global mem
if(tid ==0){ 
	g_odata[blockIdx.x]=sdata[0];
}

}

__global__ void reduce2(int *g_idata,int *g_odata){
extern __shared__ int sdata[];

//each thread loads one element from global to shared mem
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
sdata[tid]=g_idata[i];
//printf("%d- ",sdata[tid]);
__syncthreads();
//if(threadIdx.x==0){
//printf("\n");}

//do reduction in shared mem 
for(unsigned int s=blockDim.x/2; s>0;s>>=1){
	if(tid<s){
		sdata[tid]+=sdata[tid+s];
	}
	__syncthreads();
}
// write the result for this block to global mem
if(tid ==0){ 
	g_odata[blockIdx.x]=sdata[0];
}

}


__global__ void reduce3(int *g_idata,int *g_odata){
extern __shared__ int sdata[];

//each thread loads one element from global to shared mem
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockDim.x*2)+threadIdx.x;
sdata[tid]=g_idata[i]+g_idata[i+blockDim.x];
//printf("%d- ",sdata[tid]);
__syncthreads();
//if(threadIdx.x==0){
//printf("\n");}

//do reduction in shared mem 
for(unsigned int s=blockDim.x/2; s>0;s>>=1){
	if(tid<s){
		sdata[tid]+=sdata[tid+s];
	}
	__syncthreads();
}
// write the result for this block to global mem
if(tid ==0){ 
	g_odata[blockIdx.x]=sdata[0];
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

__global__ void reduce4(int *g_idata,int *g_odata){
extern __shared__ int sdata[];

//each thread loads one element from global to shared mem
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockDim.x*2)+threadIdx.x;
sdata[tid]=g_idata[i]+g_idata[i+blockDim.x];
//printf("%d- ",sdata[tid]);
__syncthreads();
//if(threadIdx.x==0){
//printf("\n");}

//do reduction in shared mem 
for(unsigned int s=blockDim.x/2; s>32;s>>=1){
	if(tid<s){
		sdata[tid]+=sdata[tid+s];
	}
	__syncthreads();
}
if(tid<32){ warpReduce(sdata,tid);}
// write the result for this block to global mem
if(tid ==0){ 
	g_odata[blockIdx.x]=sdata[0];
}

}
/*
template <unsigned int blockSize> 
__device__ void warpReducenew(volatile int* sdata,int tid ){
	if(blockSize>=64) sdata[tid]+=sdata[tid +32];
	if(blockSize>=32) sdata[tid]+=sdata[tid +16];
	if(blockSize>=16) sdata[tid]+=sdata[tid +8];
	if(blockSize>=8) sdata[tid]+=sdata[tid +4];
	if(blockSize>=4) sdata[tid]+=sdata[tid +2];
	if(blockSize>=2) sdata[tid]+=sdata[tid +1];
}


template <unsigned int blockSize>
__global__ void reduce5(int *g_idata,int * g_odata){

extern __shared__ int sdata[];
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockDim.x*2)+threadIdx.x;
sdata[tid]=g_idata[i]+g_idata[i+blockDim.x];
__syncthreads();

if(blockSize>=512){
	if(tid<256 ){sdata[tid]+=sdata[tid+256];}
	__syncthreads();
}
if(blockSize>=256){
	if(tid<128 ){sdata[tid]+=sdata[tid+128];}
	__syncthreads();
}
if(blockSize>=128){
	if(tid<64 ){sdata[tid]+=sdata[tid+64];}
	__syncthreads();
}

if(tid<32){ warpReducenew<blockSize>(sdata,tid);}
// write the result for this block to global mem
if(tid ==0){ 
	g_odata[blockIdx.x]=sdata[0];
}



}

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) 
{
	if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
	if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
	if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
	if(blockSize >=   8) sdata[tid] += sdata[tid +  4];
	if (blockSize >=   4) sdata[tid] += sdata[tid +  2];
	if (blockSize >=   2) sdata[tid] += sdata[tid +  1];}
template <unsigned int blockSize>
__global__ voidreduce6(int *g_idata, int *g_odata, unsigned int n)
 {extern __shared__ int sdata[];

 unsigned int tid = threadIdx.x;
 unsigned int i = blockIdx.x*(blockSize*2) + tid;
 unsigned int gridSize = blockSize*2*gridDim.x;

 sdata[tid] = 0;

 while (i < n)
 	{
 		sdata[tid] += g_idata[i] + g_idata[i+blockSize];  i += gridSize;
 		  }
 __syncthreads();

 if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
 if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
 if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
 if (tid < 32)warpReduce(sdata, tid);
 if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
*/




int main(int argc, char **argv) {
int N= 1<<atoi(argv[1]);
int Blocks= atoi(argv[2]);
int Threads= atoi(argv[3]);


printf("Doing a parallel reduction for  N= %d NumberOfBlocks=%d Threads=%d \n",N,Blocks,Threads );

// Define input output 
int *h_idata;
int *g_idata,*g_odata;
int d_output;
h_idata=(int *)malloc( sizeof(int)*N);
for(unsigned int i=0 ;i<N;i++){
	h_idata[i]=i%2;
	//printf("-%d",h_idata[i]);
}
//printf("\n");
cudaMalloc(&g_idata, N*sizeof(int));
cudaMalloc(&g_odata, Blocks*sizeof(int));
cudaMemcpy(g_idata, h_idata, N*sizeof(int), cudaMemcpyHostToDevice);

//reduce0<<<ceil(N/(Blocks*Threads)),Threads>>>(g_idata,g_odata);
Blocks=N/Threads;

int Level=Threads;
while(N/Level>0){
	
	//printf(" Level=%d ,N/Level=%d \n",Level,N/Level );
	if(Level==Threads){
	reduce0<<<N/Level,Threads,Threads*sizeof(int)>>>(g_idata,g_odata);
	}else{
	reduce0<<<N/Level,Threads,Threads*sizeof(int)>>>(g_odata,g_odata);
		
	}
	cudaDeviceSynchronize();
	if(N/Level<Threads){
			reduce0<<<1,N/Level,N/Level*sizeof(int)>>>(g_odata,g_odata);

	}
	Level=Level*Threads;
}

	

cudaMemcpy(&d_output, g_odata, sizeof(int), cudaMemcpyDeviceToHost);

int s=0;
for(unsigned int i=0;i<N;i++){
	s=s+h_idata[i];
}
printf("reduce0 Device %d Host %d\n",d_output,s ); 


 Level=Threads;
while(N/Level>0){
	
	if(Level==Threads){
	reduce1<<<N/Level,Threads,Threads*sizeof(int)>>>(g_idata,g_odata);
	}else{
	reduce1<<<N/Level,Threads,Threads*sizeof(int)>>>(g_odata,g_odata);
		
	}
	cudaDeviceSynchronize();
	if(N/Level<Threads){
			reduce1<<<1,N/Level,N/Level*sizeof(int)>>>(g_odata,g_odata);

	}
	Level=Level*Threads;
}

	

cudaMemcpy(&d_output, g_odata, sizeof(int), cudaMemcpyDeviceToHost);
printf("reduce1 Device %d Host %d\n",d_output,s ); 


 Level=Threads;
while(N/Level>0){
	
	if(Level==Threads){
	reduce2<<<N/Level,Threads,Threads*sizeof(int)>>>(g_idata,g_odata);
	}else{
	reduce2<<<N/Level,Threads,Threads*sizeof(int)>>>(g_odata,g_odata);
		
	}
	cudaDeviceSynchronize();
	if(N/Level<Threads){
			reduce2<<<1,N/Level,N/Level*sizeof(int)>>>(g_odata,g_odata);

	}
	Level=Level*Threads;
}

	

cudaMemcpy(&d_output, g_odata, sizeof(int), cudaMemcpyDeviceToHost);
printf("reduce2 Device %d Host %d\n",d_output,s ); 



 Level=Threads;

while(N/Level>0){
	
	if(Level==Threads){
	reduce3<<<N/(Level*2),Threads,Threads*sizeof(int)>>>(g_idata,g_odata);
	}else {
	reduce3<<<N/(Level*2),Threads,Threads*sizeof(int)>>>(g_odata,g_odata);
		
	}
	cudaDeviceSynchronize();
	Level=Level*2;
	if(N/Level<=Threads){
			reduce3<<<1,N/(Level*2),N/(Level*2)*sizeof(int)>>>(g_odata,g_odata);

	}
	Level=Level*Threads;
}

	

cudaMemcpy(&d_output, g_odata, sizeof(int), cudaMemcpyDeviceToHost);
printf("reduce3 Device %d Host %d\n",d_output,s ); 


 Level=Threads;

while(N/Level>0){
	
	if(Level==Threads){
	reduce4<<<N/(Level*2),Threads,Threads*sizeof(int)>>>(g_idata,g_odata);
	}else {
	reduce4<<<N/(Level*2),Threads,Threads*sizeof(int)>>>(g_odata,g_odata);
		
	}
	cudaDeviceSynchronize();
	Level=Level*2;
	if(N/Level<=Threads){
			reduce4<<<1,N/(Level*2),N/(Level*2)*sizeof(int)>>>(g_odata,g_odata);

	}
	Level=Level*Threads;
}

	

cudaMemcpy(&d_output, g_odata, sizeof(int), cudaMemcpyDeviceToHost);
printf("reduce4 Device %d Host %d\n",d_output,s ); 


/*
 Level=Threads;

while(N/Level>0){
	
	if(Level==Threads){
	switch (Threads)
	{
		case 512:
			reduce5<512><<< N/(Level*2), Threads,Threads*sizeof(int) >>>(g_idata, g_odata);
			break;
		case 256:
			reduce5<256><<< N/(Level*2), Threads,Threads*sizeof(int) >>>(g_idata, g_odata); 
			break;
		case 128:
			reduce5<128><<< N/(Level*2), Threads,Threads*sizeof(int) >>>(g_idata, g_odata);
			break;
		case 64:
			reduce5< 64><<<  N/(Level*2), Threads,Threads*sizeof(int) >>>(g_idata, g_odata);
			break;
		case 32:
			reduce5< 32><<<N/(Level*2), Threads,Threads*sizeof(int) >>>(g_idata, g_odata);
			break;
		case 16:
			reduce5< 16><<<N/(Level*2), Threads,Threads*sizeof(int) >>>(g_idata, g_odata);
			break;
		case  8:
			reduce5<  8><<<N/(Level*2), Threads,Threads*sizeof(int) >>>(g_idata, g_odata);
			break;
		case  4:
			reduce5<  4><<<N/(Level*2), Threads,Threads*sizeof(int) >>>(g_idata, g_odata);
			break;
		case  2:
			reduce5< 2><<<N/(Level*2), Threads,Threads*sizeof(int) >>>(g_idata, g_odata);
			break;
		case  1:
			reduce5<  1><<<N/(Level*2), Threads,Threads*sizeof(int) >>>(g_idata, g_odata);
			 break;
			}
	}else {
		switch (Threads)
	{
		case 512:
			reduce5<512><<<N/(Level*2), Threads,Threads*sizeof(int) >>>(g_odata, g_odata);
			break;
		case 256:
			reduce5<256><<<N/(Level*2), Threads,Threads*sizeof(int) >>>(g_odata, g_odata); 
			break;
		case 128:
			reduce5<128><<<N/(Level*2), Threads,Threads*sizeof(int) >>>(g_odata, g_odata);
			break;
		case 64:
			reduce5< 64><<<N/(Level*2), Threads,Threads*sizeof(int) >>>(g_odata, g_odata);
			break;
		case 32:
			reduce5< 32><<<N/(Level*2), Threads,Threads*sizeof(int) >>>(g_odata, g_odata);
			break;
		case 16:
			reduce5< 16><<<N/(Level*2), Threads,Threads*sizeof(int) >>>(g_odata, g_odata);
			break;
		case  8:
			reduce5<  8><<<N/(Level*2), Threads,Threads*sizeof(int) >>>(g_odata, g_odata);
			break;
		case  4:
			reduce5<  4><<<N/(Level*2), Threads,Threads*sizeof(int) >>>(g_odata, g_odata);
			break;
		case  2:
			reduce5< 2><<<N/(Level*2), Threads,Threads*sizeof(int) >>>(g_odata, g_odata);
			break;
		case  1:
			reduce5<  1><<<N/(Level*2), Threads,Threads*sizeof(int) >>>(g_odata, g_odata);
			 break;
			}
		
	}
	cudaDeviceSynchronize();
	Level=Level*2;
	if(N/Level<=Threads){
		switch (N/(Level*2))
	{
		case 512:
			reduce5<512><<<1, N/(Level*2),N/(Level*2)*sizeof(int) >>>(g_odata, g_odata);
			break;
		case 256:
			reduce5<256><<<1, N/(Level*2),N/(Level*2)*sizeof(int) >>>(g_odata, g_odata); 
			break;
		case 128:
			reduce5<128><<<1, N/(Level*2),N/(Level*2)*sizeof(int) >>>(g_odata, g_odata);
			break;
		case 64:
			reduce5< 64><<<1, N/(Level*2),N/(Level*2)*sizeof(int) >>>(g_odata, g_odata);
			break;
		case 32:
			reduce5< 32><<<1, N/(Level*2),N/(Level*2)*sizeof(int) >>>(g_odata, g_odata);
			break;
		case 16:
			reduce5< 16><<<1, N/(Level*2),N/(Level*2)*sizeof(int) >>>(g_odata, g_odata);
			break;
		case  8:
			reduce5<  8><<<1, N/(Level*2),N/(Level*2)*sizeof(int) >>>(g_odata, g_odata);
			break;
		case  4:
			reduce5<  4><<<1, N/(Level*2),N/(Level*2)*sizeof(int) >>>(g_odata, g_odata);
			break;
		case  2:
			reduce5< 2><<<1, N/(Level*2),N/(Level*2)*sizeof(int) >>>(g_odata, g_odata);
			break;
		case  1:
			reduce5<  1><<<1, N/(Level*2),N/(Level*2)*sizeof(int) >>>(g_odata, g_odata);
			 break;
			}
			reduce4<<<1,N/(Level*2),N/(Level*2)*sizeof(int)>>>(g_odata,g_odata);

	}
	Level=Level*Threads;
}

	

cudaMemcpy(&d_output, g_odata, sizeof(int), cudaMemcpyDeviceToHost);
printf("reduce5 Device %d Host %d\n",d_output,s ); 
*/





return 0;
}