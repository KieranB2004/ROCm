#include <hip/hip_runtime.h>

__global__ void vecADD(
    double *a, double *b, double *c,
    int n
) {
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

int main() {
// Declare all CPU arrays here
// Size, in bytes of each vector
size_t bytes = n*sizeof(double);

// Allocate memory for each vector on host
CPUArrayA = (double*)malloc(bytes);
CPUArrayB = (double*)malloc(bytes);
CPUArrayC = (double*)malloc(bytes);
CPUVerifyArrayC = (double*)malloc(bytes);



// Number of threads in each thread block
blockSize = 64;

// Number of thread blocks in grid
gridSize = (int)ceil((float)n/blockSize);

// Execute the kernel
vecADD<<<gridSize,blockSize>>>(GPUArrayA, GPUArrayB, GPUArrayC, n);
}