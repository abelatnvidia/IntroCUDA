#include<stdio.h>
__global__ void myKernel()
{ 
    __shared__ float sdata[1024*1024*1024]; 
    sdata[blockIdx.x] = blockIdx.x;
}
int main(void){ 
    myKernel<<<100,1>>>(); 
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    printf("status: %s\n",cudaGetErrorString(err));
    return 0; 
}

