#include<stdlib.h>
#define N 256

void VecAdd(float* a, float* b, float* c){
    for(int i = 0; i < N; i++){
        c[i] = a[i] + b[i];
    }
}
int main(){
    size_t numBytes = N*sizeof(float);
    float *a = (float*)malloc(numBytes);
    float *b = (float*)malloc(numBytes);
    float *c = (float*)malloc(numBytes);
    VecAdd(a, b, c);
    free(a);  free(b);  free(c);
}
