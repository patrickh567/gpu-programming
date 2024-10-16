#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error.h"
#include "gpuLib.h"

#include <stdio.h>

__global__ void saxpyKernel(const float a, float* c, const float *y, 
        const float *x, const int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N) {
        c[tid] = a * y[tid] + x[tid]; 
    }
}

int main(int argc, char* argv[]) {

    args_t args;
    parsArgs(&args, argc, argv);
    
    bool verify = args.verify;
    bool print = args.print;
    unsigned int numThreads = args.numThreads_x;
    unsigned int arraySize = args.M;
    const float maxElementSize = 10.0;
    const float a = 10.0;
 
    // Cant assign more threads than there are elements in the arrays
    assert(numThreads <= arraySize);

    float* x = (float*)malloc(arraySize*sizeof(float));
    if(x == NULL) {
        printf("Malloc failed for array x.\n");
        exit(-1);
    }

    float* y = (float*)malloc(arraySize*sizeof(float));
    if(y == NULL) {
        printf("Malloc failed for array y.\n");
        exit(-1);
    }
    
    float* c = (float*)malloc(arraySize*sizeof(float));
    if(c == NULL) {
        printf("Malloc failed for array c.\n");
        exit(-1);
    }

    for(int i = 0; i < arraySize; i++) {
        x[i] = (float)rand()/((float)RAND_MAX/maxElementSize); 
        y[i] = (float)rand()/((float)RAND_MAX/maxElementSize); 
    }

    saxpyWithCuda(c, x, y, a, arraySize, verify, numThreads);
    
    if(print) {
        printResultsVector(c, x, y, a, arraySize);
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    checkCuda(cudaDeviceReset());

    free(x);
    free(y);
    free(c);
    
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void saxpyWithCuda(float* c, const float *x, const float *y, const float a,  
        const unsigned int size, const bool verify, const unsigned int numThreads) {
    float* dev_y = 0;
    float* dev_x = 0;
    float* dev_c = 0;
    //cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCuda(cudaSetDevice(0));

    // Allocate GPU buffers for three vectors (two input, one output)    .
    checkCuda(cudaMalloc((void**)&dev_y, size * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_x, size * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_c, size * sizeof(float)));

    // Copy input vectors from host memory to GPU buffers.
    checkCuda(cudaMemcpy(dev_y, y, size * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_x, x, size * sizeof(float), cudaMemcpyHostToDevice));

    // Launch a kernel on the GPU with one thread for each element.
    saxpyKernel<<<ceil(size/numThreads), numThreads>>>(a, dev_c, dev_y, dev_x, size);

    // Check for any errors launching the kernel
    checkCuda(cudaGetLastError());
    
    // Run CPU Saxpy, overlapping latency with CUDA saxpy computation
    float* cpu_c;
    if(verify) {
        cpu_c = (float*)malloc(size * sizeof(float));
        saxpyWithCpu(cpu_c, x, y, a, size); 
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCuda(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    checkCuda(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify CUDA computation is correct
    if(verify) {
        verifyCudaVector(cpu_c, c, size);
        free(cpu_c);
    }

    checkCuda(cudaFree(dev_y));
    checkCuda(cudaFree(dev_x));
    checkCuda(cudaFree(dev_c));
}
