#include<stdio.h>

// GPU Kernel definition
__global__ void sayHello(void){
    printf("Hello World from the GPU!\n");
}

int main(void){

    //launch kernel
    sayHello<<<1,5>>>();  
	
    //wait for the kernel to finish
    cudaDeviceSynchronize();  
	
    //that's all
    return 0;
}
