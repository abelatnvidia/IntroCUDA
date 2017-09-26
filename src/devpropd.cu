#include<stdio.h>
#include<cuda.h>

int main(void){
    int deviceCount; 
    char deviceName[256];
    CUdevice device;
    size_t szMem; int szProc;
    cuInit(0);
    cuDeviceGetCount(&deviceCount);
    cuDeviceGet(&device,0);
    cuDeviceGetName(deviceName,255,device);
    cuDeviceTotalMem(&szMem,device);
    cuDeviceGetAttribute(&szProc,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,device);
    printf("There are %d devices detected\n",deviceCount);
    printf("Device %s has %f GB of global memory\n",
        deviceName,szMem/pow(1024.0,3));
    printf("Device multiprocessor count: %d\n",szProc);
}
