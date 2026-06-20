#include <hip/hip_runtime.h>

__global__ void gpuHello(){
    printf("Hello World\n");
}

int main() {
    gpuHello<<<1,1>>>();
    hipDeviceSynchronize();
}