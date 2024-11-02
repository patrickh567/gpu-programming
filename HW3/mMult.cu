#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <iostream>
#include "cuda_error.h"

#define VERIFY_KERNELS
//#define PROFILE_TILES

#define MAX_TILE_WIDTH 32

void launchMMultKernel(void (*kernel)(const float*, const float*, float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int), 
        const float* h_A, const float* h_B, float* h_C, const unsigned int m, const unsigned int n, const unsigned int p, const unsigned int tileWidth);

float verifyCudaMatrix(const float* matCpu, const float* matCuda, 
        const unsigned int N, const unsigned int M);

void printMatrix(const float* c, const unsigned int N, const unsigned int M);

void printVector(const float* vec, const unsigned int size);

void printResultsMatrix(const float*  c, const float* a, const float* b, 
        const unsigned int M, const unsigned int N, const unsigned int P);

void verifyKernels();

void profileTiles();

void cpuMMult(const float* A, const float* B, float* C, 
        const unsigned int m, const unsigned int n, const unsigned int p) {
    
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            C[(i * n) + j] = 0;
            for(int k = 0; k < p; k++) {
                C[(i * n) + j] += A[(i * p) + k] * B[(k * n) + j];
            }
        }
    }
}

__global__ void cudaGlobalMMult(const float* d_A, const float* d_B, float* d_C, 
        const unsigned int m, const unsigned int n, const unsigned int p, const unsigned int tileWidth) {

    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    float PValue = 0.0;
    if((idx_y < m) && (idx_x < n)) {
        for(int i = 0; i < p; i++) {
            PValue += d_A[idx_y * p + i] * d_B[idx_x + i * n];
        }
        d_C[idx_y * n + idx_x] = PValue;
    }

}


__global__ void cudaSharedTiledMMult(const float* d_A, const float* d_B, float* d_C, 
        const unsigned int m, const unsigned int n, const unsigned int p, const unsigned int tileWidth) {

    __shared__ float s_A[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    __shared__ float s_B[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0;

    for(int phase = 0; phase < (p - 1) / tileWidth + 1; phase++) {
        // Collaboratively load tile into shared memory
        if((phase * tileWidth + threadIdx.x < p) && (idx_y < m)) { 
            s_A[threadIdx.y][threadIdx.x] = d_A[(idx_y * p) + (phase * tileWidth + threadIdx.x)];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0;
        }
        if((phase * tileWidth + threadIdx.y < p) && (idx_x < n)) {
            s_B[threadIdx.y][threadIdx.x] = d_B[(phase * tileWidth + threadIdx.y) * n + idx_x];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0;
        }
        // Wait for threads to complete
        __syncthreads();
        // Perform partial dot-product
        if(idx_y < m && idx_x < n) {
            for(int k = 0; k < tileWidth; k++) {
                Pvalue += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
            }
        }
        // Wait for threads to complete
        __syncthreads();
            
    }
    // Write full dot-product to global memory
    if(idx_y < m && idx_x < n) {
        d_C[idx_y * n + idx_x] = Pvalue;
    }

}


int main(void) {
    
    #ifdef VERIFY_KERNELS
    verifyKernels();
    #endif
    #ifdef PROFILE_TILES
    profileTiles();
    #endif
    return 0;

}

void verifyKernels() {

    float* h_A;
    float* h_B;
    float* h_C;
    float* cpu_C;

    const unsigned maxElementSize = 1.0;

    //launchMMultKernel(&cudaGlobalMMult, h_A, h_B, h_C, m, n, p, 16);
    const unsigned int numTests = 3;
    unsigned int mVals[numTests] = {100, 1000, 2000};
    unsigned int nVals[numTests] = {100, 1000, 3000};
    unsigned int pVals[numTests] = {100, 1000, 2500};

    for(int k = 0; k < numTests; k++) {
        h_A = (float*)malloc(mVals[k] * pVals[k] * sizeof(float));
        if(h_A == NULL) {
            std::cout << "malloc failed." << std::endl;
            exit(-1);
        }

        h_B = (float*)malloc(pVals[k] * nVals[k] * sizeof(float));
        if(h_B == NULL) {
            std::cout << "malloc failed." << std::endl;
            exit(-1);
        }

        h_C = (float*)malloc(mVals[k] * nVals[k] * sizeof(float));
        if(h_C == NULL) {
            std::cout << "malloc failed." << std::endl;
            exit(-1);
        }
 
        cpu_C = (float*)malloc(mVals[k] * nVals[k] * sizeof(float));
        if(cpu_C == NULL) {
            std::cout << "malloc failed." << std::endl;
            exit(-1);
        }
 
        for(int i = 0; i < mVals[k]; i++) {
            for(int j = 0; j < pVals[k]; j++) {
                h_A[i * pVals[k] + j] = (float)rand()/((float)RAND_MAX/maxElementSize);
            }
        }

        for(int i = 0; i < pVals[k]; i++) {
            for(int j = 0; j < nVals[k]; j++) {
                h_B[i * nVals[k] + j] = (float)rand()/((float)RAND_MAX/maxElementSize);
            }
        }
        
        std::cout << std::endl << "Running kernel and cpu verification for matrix M=" << mVals[k] << ", N=" 
            << nVals[k] << ", P=" << pVals[k] << std::endl;
        std::cout << "-------------------------------------------------------------------" << std::endl;
        cpuMMult(h_A, h_B, cpu_C, mVals[k], nVals[k], pVals[k]);
        launchMMultKernel(&cudaSharedTiledMMult, h_A, h_B, h_C, mVals[k], nVals[k], pVals[k], 16);
        float errorFraction = verifyCudaMatrix(cpu_C, h_C, mVals[k], nVals[k]);
        std::cout << "Max Error Fraction: " << errorFraction << std::endl;
        free(h_A);
        free(h_B);
        free(h_C);
        free(cpu_C);
    }

}

void profileTiles() {
    
    const unsigned int m = 2000;
    const unsigned int n = 3000;
    const unsigned int p = 2500;
    
    float* h_A;
    float* h_B;
    float* h_C;

    const unsigned int maxElementSize = 10;
        
    h_A = (float*)malloc(m * p * sizeof(float));
    if(h_A == NULL) {
        std::cout << "malloc failed." << std::endl;
        exit(-1);
    }

    h_B = (float*)malloc(p * n * sizeof(float));
    if(h_B == NULL) {
        std::cout << "malloc failed." << std::endl;
        exit(-1);
    }

    h_C = (float*)malloc(m * n * sizeof(float));
    if(h_C == NULL) {
        std::cout << "malloc failed." << std::endl;
        exit(-1);
    }
    
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            //h_A[i * p + j] = (i * p + j)*alpha;
            h_A[i * p + j] = (float)rand()/((float)RAND_MAX/maxElementSize);
        }
    }
    
    for(int i = 0; i < p; i++) {
        for(int j = 0; j < n; j++) {
            //h_B[i * n + j] = (i * n + j)*alpha;
            h_B[i * n + j] = (float)rand()/((float)RAND_MAX/maxElementSize);
        }
    }
    // global memory
    launchMMultKernel(&cudaGlobalMMult, h_A, h_B, h_C, m, n, p, 16);

    // tile width = 8
    launchMMultKernel(&cudaSharedTiledMMult, h_A, h_B, h_C, m, n, p, 8);
    
    // tile width = 12
    launchMMultKernel(&cudaSharedTiledMMult, h_A, h_B, h_C, m, n, p, 12);
    
    // tile width = 16
    launchMMultKernel(&cudaSharedTiledMMult, h_A, h_B, h_C, m, n, p, 16);
    
    // tile width = 32
    launchMMultKernel(&cudaSharedTiledMMult, h_A, h_B, h_C, m, n, p, 32);

}

void launchMMultKernel(void (*kernel)(const float*, const float*, float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int), 
        const float* h_A, const float* h_B, float* h_C, const unsigned int m, const unsigned int n, const unsigned int p, 
        const unsigned int tileWidth) {

    float* d_A;
    float* d_B;
    float* d_C;

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCuda(cudaSetDevice(0));
    
    // Allocate GPU buffers for three vectors (two input, one output)    .
    checkCuda(cudaMalloc((void**)&d_A, m * p * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_B, p * n * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_C, m * n * sizeof(float)));

    // Copy input vectors from host memory to GPU buffers.
    checkCuda(cudaMemcpy(d_A, h_A, m * p * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, p * n * sizeof(float), cudaMemcpyHostToDevice));
     
    // Launch kernel
    dim3 DimGrid((n-1)/tileWidth+1, (m-1)/tileWidth+1);
    dim3 DimBlock(tileWidth, tileWidth, 1);

    (*kernel)<<<DimGrid, DimBlock>>>(d_A, d_B, d_C, m, n, p, tileWidth);

    // Check for any errors launching the kernel
    checkCuda(cudaGetLastError());
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCuda(cudaDeviceSynchronize());
    
    // Copy output vector from GPU buffer to host memory.
    checkCuda(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));

}

float verifyCudaMatrix(const float*  matCpu, const float* addCuda, 
        const unsigned int N, const unsigned int M) {

    const float tol = 1e-9; 
    bool foundError = false;
    float maxErrorFraction = 0.0;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            float error = abs(matCpu[i * M + j] - addCuda[i * M + j]);
            float errorFraction = error / matCpu[i * M + j];
            if(errorFraction > 0.0) {
                maxErrorFraction = errorFraction;
            }
            if(error > tol) {
                foundError = true;
                printf("CUDA matrix multiplication computation failed at index (%i, %i)\n", i, j);
                printf("CPU: %f, GPU: %f\n", matCpu[i * M + j], addCuda[i * M + j]);
            } 
            //else {
            //    printf("%.3f %.3f\n", matCpu[i * M + j], addCuda[i * M + j]);
            //}
        }
    }
    if(foundError == false) {
        printf("CUDA matrix multiplication verification check passed.\n");
    }
    return maxErrorFraction;
}

void printVector(const float* vec, const unsigned int size) {
    printf("{");
    for(int i = 0; i < size-1; i++) {
        printf("%.2f, ", vec[i]); 
    }
    printf("%.2f}", vec[size-1]);
}

void printMatrix(const float* c, const unsigned int M, const unsigned int N) {

    printf("{"); 
    for(int i = 0; i < M-1; i++) {
        printVector(&c[i*N], N);
        printf(",\n");
    }
    printVector(&c[(M-1)*N], N);
    printf("}\n");

}

void printResultsMatrix(const float*  c, const float* a, const float* b, 
        const unsigned int M, const unsigned int N, const unsigned int P) {
    printMatrix(a, M, P);
    printf("X\n");
    printMatrix(b, P, N);
    printf("=\n");
    printMatrix(c, M, N);

}
