#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error.h"

#include <stdio.h>

__global__ void printfKernel() {
    printf("blockIdx.x = %i, threadIdx.x = %i, GTID = %i\n", 
            blockIdx.x, threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x); 
}

int main() {

    // Launch a kernel on the GPU with one thread for each element.
    printfKernel<<<5, 2>>>();

    // Check for any errors launching the kernel
    checkCuda(cudaGetLastError());

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    checkCuda(cudaDeviceReset());

    return 0;
}
