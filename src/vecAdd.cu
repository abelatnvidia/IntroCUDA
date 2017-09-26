#define N 256
#include<stdlib.h>

__global__ void vecAdd(float *a, float *b, float *c){
    c[threadIdx.x] = a[threadIdx.x]+b[threadIdx.x];
}

int main(void){

    // number of bytes to alloc for arrays    
    size_t numBytes = N*sizeof(float);

    // init host and device pointers
    float *ha, *hb, *hc, *da, *db, *dc;

    // alloc host memory/arrays (pagable memory)
    ha = (float*)malloc(numBytes);
    hb = (float*)malloc(numBytes);
    hc = (float*)malloc(numBytes);

    // mem alloc arrays on the GPU device
    cudaMalloc(&da,numBytes);
    cudaMalloc(&db,numBytes);
    cudaMalloc(&dc,numBytes);

    // copy host arrays to device
    cudaMemcpy(da, ha, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, hc, numBytes, cudaMemcpyHostToDevice);

    // launch configuration
    dim3 gridSz (1,1,1), blockSz(N,1,1);

    // launch CUDA kernel
    vecAdd<<<gridSz,blockSz>>>(da,db,dc);

    // wait for kernel to finish
    cudaDeviceSynchronize();

    // free host memory
    free(ha);  free(hb);  free(hc);

    // free device memory
    cudaFree(da);  cudaFree(db);  cudaFree(dc);
}
