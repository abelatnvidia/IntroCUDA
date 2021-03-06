#include<stdio.h>

int main(void){

  // init pointer to data
  char *data;
  
  // specify 32 GB of memory in bytes
  size_t numBytes = 1024*1024*1024;
  
  // Allocate 32 GB
  cudaError_t err = cudaMallocManaged(&data, numBytes/2);

  // blab about it
  printf("malloc status: %s\n",cudaGetErrorString(err));   

  return 0;
}
