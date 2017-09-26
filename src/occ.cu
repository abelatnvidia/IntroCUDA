int arraySize = 1024*1024;
int blockSize, minGridSize, gridSize, maxActiveBlocks;

cudaOccupancyMaxPotentialBlockSize( 
  &minGridSize, &blockSize, MyKernel, 0, 0);
 
gridSize = (arraySize + blockSize - 1) / blockSize; 

// MyKerel<<<gridSize,blockSize>>>(args);

cudaOccupancyMaxActiveBlocksPerMultiprocessor( 
  &maxActiveBlocks, MyKernel, blockSize, 0);
  
int dev; cudaDeviceProp p;
cudaGetDevice(&dev); 
cudaGetDeviceProperties(&p, dev);

// calculate theoretical occupancy
float occupancy = (maxActiveBlocks * blockSize / props.warpSize) / (float)(props.maxThreadsPerMultiProcessor / props.warpSize);
