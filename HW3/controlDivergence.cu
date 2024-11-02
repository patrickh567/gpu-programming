#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <iostream>
#include "cuda_error.h"

void launchKernel(void (*kernel)(const float*, const float*, float*, const unsigned int, const bool), 
        const float* h_inVec0, const float* h_inVec1, float* h_outVec, const unsigned int vecLength, 
        const unsigned int numThreadsPerBlock, const unsigned int numBlocks, const bool diverge);

__global__ void singleBranchDivergenceKernel(const float* d_inVec0, const float* d_inVec1, float* d_outVec, const unsigned int vecLength, const bool diverge) {

    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx_x < vecLength) {
        if(((threadIdx.x % 32) < 16) || !diverge) {
            d_outVec[idx_x] = d_inVec0[idx_x] + d_inVec1[idx_x];
        } else {
            d_outVec[idx_x] = d_inVec0[idx_x] + d_inVec1[idx_x] * 3;
        }
    }

}

__global__ void multiBranchDivergenceKernel(const float* d_inVec0, const float* d_inVec1, float* d_outVec, const unsigned int vecLength, const bool diverge) {

    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx_x < vecLength) {
        int threadIdx32 = threadIdx.x % 32;
        if((threadIdx32 < 8) || !diverge) {
            d_outVec[idx_x] = d_inVec0[idx_x] + d_inVec1[idx_x];
        } else if((threadIdx32 > 7) && (threadIdx32 < 16)) {
            d_outVec[idx_x] = d_inVec0[idx_x] + d_inVec1[idx_x] * 2;
        } else if((threadIdx32 > 15) && (threadIdx32 > 24)){
            d_outVec[idx_x] = d_inVec0[idx_x] + d_inVec1[idx_x] * 3;
        } else {
            d_outVec[idx_x] = d_inVec0[idx_x] + d_inVec1[idx_x] * 4;
        }
    }
}

int main(void) {
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    //const unsigned int maxThreadsPerBlock = dev_prop.maxThreadsPerBlock;
    // Single branch with no warp divergence 
    // Half the vector length is on a warp boundary
    unsigned int numBlocks = 100000;
    unsigned int numThreadsPerBlock = 1024;
    unsigned int vectorLength = numBlocks * numThreadsPerBlock;
    float* h_inVec0 = (float*)malloc(sizeof(float)*vectorLength);
    float* h_inVec1 = (float*)malloc(sizeof(float)*vectorLength);
    float* h_outVec = (float*)malloc(sizeof(float)*vectorLength);
    bool diverge = false;
    launchKernel(&singleBranchDivergenceKernel, h_inVec0, h_inVec1, h_outVec, vectorLength, numThreadsPerBlock, numBlocks, diverge);
    diverge = true;
    launchKernel(&singleBranchDivergenceKernel, h_inVec0, h_inVec1, h_outVec, vectorLength, numThreadsPerBlock, numBlocks, diverge);
    diverge = false;
    launchKernel(&multiBranchDivergenceKernel, h_inVec0, h_inVec1, h_outVec, vectorLength, numThreadsPerBlock, numBlocks, diverge);
    diverge = true;
    launchKernel(&multiBranchDivergenceKernel, h_inVec0, h_inVec1, h_outVec, vectorLength, numThreadsPerBlock, numBlocks, diverge);
    free(h_inVec0);
    free(h_inVec1);
    free(h_outVec);
    return 0;
    //// Single branch with warp divergence
    //// Half the vector length is not the size of a warp
    //numBlocks = 1000000;
    //vecLength = 32 * numBlocks;
    //h_inVec = (float*)malloc(sizeof(float)*vecLength);
    //h_outVec = (float*)malloc(sizeof(float)*vecLength);
    //launchKernel(&singleBranchDivergenceKernel, h_inVec, h_outVec, vecLength, numBlocks, diverge);
    //free(h_inVec);
    //free(h_outVec);
    //return 0;
    //// Multi branch with no warp divergence
    //// 1/4 the length of the vector is the size of a warp
    //vecLength = 128;
    //numBlocks = 1024;
    //h_inVec = (float*)malloc(sizeof(float)*vecLength);
    //h_outVec = (float*)malloc(sizeof(float)*vecLength);
    //launchKernel(&multiBranchDivergenceKernel, h_inVec, h_outVec, vecLength, numBlocks);
    //free(h_inVec);
    //free(h_outVec);
    //// Multi branch with warp divergence
    //// 1/4 the length of the vector is not size of a warp
    //vecLength = 32;
    //numBlocks = 1024;
    //h_inVec = (float*)malloc(sizeof(float)*vecLength);
    //h_outVec = (float*)malloc(sizeof(float)*vecLength);
    //launchKernel(&multiBranchDivergenceKernel, h_inVec, h_outVec, vecLength, numBlocks);
    //free(h_inVec);
    //free(h_outVec);
    //return 0;
}

void launchKernel(void (*kernel)(const float*, const float*, float*, const unsigned int, const bool), 
        const float* h_inVec0, const float* h_inVec1, float* h_outVec, const unsigned int vecLength, 
        const unsigned int numThreadsPerBlock, const unsigned int numBlocks, const bool diverge) {

    float* d_inVec0;
    float* d_inVec1;
    float* d_outVec;

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCuda(cudaSetDevice(0));
    
    // Allocate GPU buffers for three vectors (two input, one output)    .
    checkCuda(cudaMalloc((void**)&d_inVec0, vecLength * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_inVec1, vecLength * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_outVec, vecLength * sizeof(float)));

    // Copy input vectors from host memory to GPU buffers.
    checkCuda(cudaMemcpy(d_inVec0, h_inVec0, vecLength, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_inVec1, h_inVec1, vecLength, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_outVec, h_outVec, vecLength, cudaMemcpyHostToDevice));
     
    // Launch kernel
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(numThreadsPerBlock, 1, 1);

    //printf("%i\n", (int)ceil(((float)numThreads) / maxThreadsPerBlock));
    (*kernel)<<<dimGrid, dimBlock>>>(d_inVec0, d_inVec1, d_outVec, vecLength, diverge);

    // Check for any errors launching the kernel
    checkCuda(cudaGetLastError());
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCuda(cudaDeviceSynchronize());
    
    // Copy output vector from GPU buffer to host memory.
    checkCuda(cudaMemcpy(h_outVec, d_outVec, vecLength * sizeof(float), cudaMemcpyDeviceToHost));
    
    checkCuda(cudaFree(d_inVec0));
    checkCuda(cudaFree(d_inVec1));
    checkCuda(cudaFree(d_outVec));

}
