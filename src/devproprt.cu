#include<stdio.h>
#include<cuda_runtime.h>

int main(void){    
    int deviceCount;  
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&deviceCount);
    cudaGetDeviceProperties(&deviceProp,0);
    printf("There are %d gpu devices\n",deviceCount);
    printf("Device %s has %f GB of global memory\n",
        deviceProp.name,
        deviceProp.totalGlobalMem/pow(1024.0,3)
    );
}
