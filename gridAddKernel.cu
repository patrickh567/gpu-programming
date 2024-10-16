#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error.h"
#include "gpuLib.h"

#include <stdio.h>

__global__ void gridAddKernel(float* c, const float* y, const float* x, 
        const int N, const int M, const int P) {

    unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int tid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if((tid_y < N) && (tid_x < M) && (tid_z < P)) {
        c[tid_z * M * N + tid_x + tid_y * M] = 
            y[tid_z * M * N + tid_x + tid_y * M] 
            + x[tid_z * M * N + tid_x + tid_y * M];
    }

}

int main(int argc, char* argv[]) {

    args_t args;
    parsArgs(&args, argc, argv);
    
    bool verify = args.verify;
    unsigned int numThreads_x = args.numThreads_x;
    unsigned int numThreads_y = args.numThreads_y;
    unsigned int numThreads_z = args.numThreads_z;
    unsigned int M = args.M;
    unsigned int N = args.N;
    unsigned int P = args.P;
    const unsigned maxElementSize = 10.0;
    
    // Cant assign more threads than there are elements in the matrix
    assert(numThreads_x <= M);
    assert(numThreads_y <= N);
    assert(numThreads_z <= P);

    float* x = allocateGrid(M, N, P);
    float* y = allocateGrid(M, N, P);
    float* c = allocateGrid(M, N, P);

    initializeGrid(x, N, M, P, maxElementSize);
    initializeGrid(y, N, M, P, maxElementSize);

    gridAddWithCuda(c, x, y, N, M, P, verify, numThreads_x, numThreads_y, numThreads_z);
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    checkCuda(cudaDeviceReset());

    free(x);
    free(y);
    free(c);
    
    return 0;

}

// Helper function for using CUDA to add vectors in parallel.
void gridAddWithCuda(float* c, const float*  x, const float*  y, 
        const unsigned int N, const unsigned int M, const unsigned int P,
        const bool verify, const unsigned int numThreads_x, const unsigned int numThreads_y, 
        const unsigned int numThreads_z) {

    float* dev_y;
    float* dev_x;
    float* dev_c;

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCuda(cudaSetDevice(0));

    // Allocate GPU buffers for three matrices (two input, one output)    .
    checkCuda(cudaMalloc((void**)&dev_y, M * N * P * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_x, M * N * P * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_c, M * N * P * sizeof(float)));

    // Copy input vectors from host memory to GPU buffers.
    checkCuda(cudaMemcpy(dev_y, y, M * N * P * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_x, x, M * N * P * sizeof(float), cudaMemcpyHostToDevice));


    // Launch a kernel on the GPU with one thread for each element.
    dim3 dimGrid(ceil(((float)M)/((float)numThreads_x)), 
            ceil(((float)N)/((float)numThreads_y)), 
            ceil(((float)N)/((float)numThreads_z)));

    dim3 dimBlock(numThreads_x, numThreads_y, numThreads_z);

    gridAddKernel<<<dimGrid, dimBlock>>>(dev_c, dev_y, dev_x, N, M, P);

    // Check for any errors launching the kernel
    checkCuda(cudaGetLastError());
    
    // Run CPU matrix addition, overlapping latency with CUDA gridAdd computation
    float* cpu_c;
    if(verify) {
        cpu_c = (float*)malloc(sizeof(float)*M*N*P); 
        gridAddWithCpu(cpu_c, x, y, N, M, P); 
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCuda(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    float* lin_c = (float*)malloc(M * N * P * sizeof(float));
    checkCuda(cudaMemcpy(c, dev_c, M * N * P * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify CUDA computation is correct
    if(verify) {
        verifyCudaGrid(cpu_c, c, M, N, P);
        free(cpu_c);
    }

    checkCuda(cudaFree(dev_y));
    checkCuda(cudaFree(dev_x));
    checkCuda(cudaFree(dev_c));

}

