#include "stdio.h"

__global__ void MyKernel(int *array, int arrayCount) 
{ 
  int idx = threadIdx.x + blockIdx.x * blockDim.x; 
  if (idx < arrayCount) 
  { 
    array[idx] *= array[idx]; 
  } 
} 

int main (void)  { 

  int arraySize = 1024*1024;
  int blockSize, minGridSize, gridSize, maxActiveBlocks;

  cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                      MyKernel, 0, 0); 
  // Round up according to array size 
  gridSize = (arraySize + blockSize - 1) / blockSize; 

  //MyKernel<<< gridSize, blockSize >>>(array, arrayCount); 

  // calculate theoretical occupancy
  cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, 
                                                 MyKernel, blockSize, 0);
  // get device properties
  int device;
  cudaDeviceProp props;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);

  // calculate theoretical occupancy
  float occupancy = (maxActiveBlocks * blockSize / props.warpSize) / 
                    (float)(props.maxThreadsPerMultiProcessor / 
                            props.warpSize);

  printf("Thread-blocks of size %d with gridSize %d. Theoretical occupancy: %f\n", 
         blockSize, gridSize,occupancy);
}
