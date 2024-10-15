#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error.h"
#include "matAddKernel.h"

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
        printResults(c, x, y, N, M);
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    checkCuda(cudaDeviceReset());

    free(x);
    free(y);
    
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
        verifyCuda(cpu_c, c, M, N);
        free(cpu_c);
    }

    checkCuda(cudaFree(dev_y));
    checkCuda(cudaFree(dev_x));
    checkCuda(cudaFree(dev_c));

}

void printMatrix(const float* c, const unsigned int N, const unsigned int M) {

    printf("{"); 
    for(int i = 0; i < N-1; i++) {
        printf("{");
        for(int j = 0; j < M-1; j++) {
            printf("%.2f, ", c[i * M + j]);
        }
        printf("%.2f},\n", c[i * M + M-1]);
    }
    printf("{");
    for(int j = 0; j < M-1; j++) {
        printf("%.2f, ", c[N-1 + j]);
    }
    printf("%.2f}}\n", c[N-1 + M-1]);

}

void printResults(const float*  c, const float* x, const float* y, 
        const unsigned int M, const unsigned int N) {
    
    printMatrix(x, N, M);
    printf("+\n");
    printMatrix(y, N, M);
    printf("=\n");
    printMatrix(c, N, M);

}

void matAddWithCpu(float* c, const float* x, const float*  y, 
        const unsigned int N, const unsigned int M) {

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            c[i * M + j] = x[i * M + j] + y[i * M + j];
        }
    }

}

void verifyCuda(const float*  matAddCpu, const float* matAddCuda, 
        const unsigned int N, const unsigned int M) {

    const float tol = 1e-7; 
    bool error = false;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            if(abs(matAddCpu[i * M + j] - matAddCuda[i * M + j]) > tol) {
                error = true;
                printf("CUDA matrix addition computation failed at index (%i, %i)\n", i, j);
                printf("CPU: %f, GPU: %f\n", matAddCpu[i * M + j], matAddCuda[i * M + j]);
                break;
            }
        }
    }
    if(error == false) {
        printf("CUDA matrix addition verification check passed.\n");
    }

}

float* allocateMatrix(const unsigned int M, const unsigned int N) {

    float* x = (float*)malloc(sizeof(float*)*N*M); 
    if(x == NULL) {
        printf("Malloc failed for array x.\n");
        exit(-1);
    }

    //for(int i = 0; i < M; i++) {
    //    x[i] = (float*)malloc(sizeof(float)*M); 
    //    if(x[i] == NULL) {
    //        printf("Malloc failed for array x.\n");
    //        exit(-1);
    //    }
    //}
    return x;

}

void initializeMatrix(float* x, const unsigned int N, const unsigned int M, 
        const unsigned int maxElementSize) {

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            x[i * M + j] = (float)rand()/((float)RAND_MAX/maxElementSize); 
        }
    }

}

void parsArgs(args_t* args, int argc, char* argv[]) {

    if(argc > 7) {
        printf("Wrong number of args.\n");
        exit(1);
    }

    if(argc > 1) {
        for(int i = 1; i < argc; i++) {
            if(strcmp(argv[i], "--verify") == 0) {
                args->verify = true;
            } else if(strcmp(argv[i], "--print_results") == 0) {
                args->print = true;
            } else if(strcmp(argv[i], "--num_threads") == 0) {
                args->numThreads_x = atoi(argv[i+1]);
                args->numThreads_y = atoi(argv[i+1]);
                i++;
            } else if(strcmp(argv[i], "--rows") == 0) {
                args->N = atoi(argv[i+1]);
                i++;
            } else if(strcmp(argv[i], "--cols") == 0) {
                args->M = atoi(argv[i+1]);
                i++;
            } else {
                printf("Unrecognized arg. Aborting\n"); 
                exit(1);
            }
        }
    }

}
