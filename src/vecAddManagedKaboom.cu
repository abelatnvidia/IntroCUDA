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

    // alloc host memory/arrays
    cudaMallocManaged(&ha,numBytes);
    cudaMallocManaged(&hb,numBytes);
    cudaMallocManaged(&hc,numBytes);

    // init host arrays
    for(int i=0; i<N; i++){ ha[i]=(float)i; hb[i]=(float)i; }

    // launch configuration
    dim3 gridSz(1,1,1), blockSz(N,1,1);

    // launch CUDA kernel
    vecAdd<<<gridSz,blockSz>>>(ha,hb,hc);

    printf("invalid managed memory reference: %f\n",ha[0]);

    // wait for kernel to finish
    cudaDeviceSynchronize();

    // kernel result (no memcpy!)
    for (int i=1; i<N; i++){ printf("c[%d]: %f\n",i,hc[i]);}

    // free host memory
    cudaFreeHost(ha);  cudaFreeHost(hb);  cudaFreeHost(hc);
}
