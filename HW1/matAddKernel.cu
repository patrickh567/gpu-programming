#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error.h"
#include "gpuLib.h"

#include <stdio.h>

__global__ void matAddKernel(float* c, const float* y, const float* x, 
        const int N, const int M) {

    unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if((tid_y < N) && (tid_x < M)) {
        c[tid_x + tid_y * M] = y[tid_x + tid_y * M] + x[tid_x + tid_y * M];
    }

}

int main(int argc, char* argv[]) {

    args_t args;
    parsArgs(&args, argc, argv);
    
    bool verify = args.verify;
    bool print = args.print;
    unsigned int numThreads_x = args.numThreads_x;
    unsigned int numThreads_y = args.numThreads_y;
    unsigned int M = args.M;
    unsigned int N = args.N;

    const unsigned maxElementSize = 10.0;
    
    // Cant assign more threads than there are elements in the matrix
    assert(numThreads_x <= M);
    assert(numThreads_y <= N);

    float* x = allocateMatrix(M, N);
    float* y = allocateMatrix(M, N);
    float* c = allocateMatrix(M, N);

    initializeMatrix(x, N, M, maxElementSize);
    initializeMatrix(y, N, M, maxElementSize);

    matAddWithCuda(c, x, y, N, M, verify, numThreads_x, numThreads_y);
    
    if(print) {
        printResultsMatrix(c, x, y, N, M);
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
void matAddWithCuda(float* c, const float*  x, const float*  y, 
        const unsigned int N, const unsigned int M, const bool verify, 
        const unsigned int numThreads_x, const unsigned int numThreads_y) {

    float* dev_y;
    float* dev_x;
    float* dev_c;
    //cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCuda(cudaSetDevice(0));

    // Allocate GPU buffers for three matrices (two input, one output)    .
    checkCuda(cudaMalloc((void**)&dev_y, M * N * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_x, M * N * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_c, M * N * sizeof(float)));
    //size_t pitch;
    //checkCuda(cudaMallocPitch(&dev_x, &pitch, N, M));
    //printf("%i\n", (int)pitch);
    //checkCuda(cudaMemcpy2D(dev_x, pitch, x, N*sizeof(float), M*sizeof(float), N, cudaMemcpyHostToDevice));
    //exit(0);


    // Copy input vectors from host memory to GPU buffers.
    checkCuda(cudaMemcpy(dev_y, y, M * N * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_x, x, M * N * sizeof(float), cudaMemcpyHostToDevice));


    // Launch a kernel on the GPU with one thread for each element.
    dim3 dimGrid(ceil(((float)M)/((float)numThreads_x)), 
            ceil(((float)N)/((float)numThreads_y)), 1);

    dim3 dimBlock(numThreads_x, numThreads_y, 1);

    matAddKernel<<<dimGrid, dimBlock>>>(dev_c, dev_y, dev_x, N, M);

    // Check for any errors launching the kernel
    checkCuda(cudaGetLastError());
    
    // Run CPU matrix addition, overlapping latency with CUDA matAdd computation
    float* cpu_c;
    if(verify) {
        cpu_c = (float*)malloc(sizeof(float)*M*N); 
        matAddWithCpu(cpu_c, x, y, N, M); 
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCuda(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    float* lin_c = (float*)malloc(M * N * sizeof(float));
    checkCuda(cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify CUDA computation is correct
    if(verify) {
        verifyCudaMatrix(cpu_c, c, M, N);
        free(cpu_c);
    }

    checkCuda(cudaFree(dev_y));
    checkCuda(cudaFree(dev_x));
    checkCuda(cudaFree(dev_c));

}

