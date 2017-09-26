#define N 256
#include<stdio.h>

__global__ void vecAdd(float *a, float *b, float *c){
    c[threadIdx.x] = a[threadIdx.x]+b[threadIdx.x];
}

int main(void){

    // number of bytes to alloc for arrays    
    size_t numBytes = N*sizeof(float);

    // init host and device pointers
    float *ha, *hb, *hc;

    // alloc host memory/arrays (pinned, mapped)
    cudaHostAlloc(&ha,numBytes,cudaHostAllocMapped);
    cudaHostAlloc(&hb,numBytes,cudaHostAllocMapped);
    cudaHostAlloc(&hc,numBytes,cudaHostAllocMapped);

    // init host arrays
    for(int i=0; i<N; i++){ ha[i]=1.0; hb[i]=1.0; }

    // launch configuration
    dim3 gridSz(1,1,1), blockSz(N,1,1);

    // launch CUDA kernel
    vecAdd<<<gridSz,blockSz>>>(ha,hb,hc);

    // wait for kernel to finish
    cudaDeviceSynchronize();

    // kernel result (no memcpy!)
    for (int i=1; i<N; i++){ printf("c[%d]: %f\n",i,hc[i]);}

    // free host memory
    cudaFreeHost(ha);  cudaFreeHost(hb);  cudaFreeHost(hc);
}
