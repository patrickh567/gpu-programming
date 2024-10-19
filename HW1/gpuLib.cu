#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "gpuLib.h"

void verifyCudaVector(const float* saxpyCpu, const float* saxpyCuda, const unsigned int size) {
    const float tol = 1e-4; 
    bool error = false;
    for(int i = 0; i < size; i++) {
        if(abs(saxpyCpu[i] - saxpyCuda[i]) > tol) {
            error = true;
            printf("CUDA saxpy computation failed at index %i\n", i);
            printf("CPU: %f, GPU: %f\n", saxpyCpu[i], saxpyCuda[i]);
            break;
        }
    }
    if(error == false) {
        printf("CUDA saxpy verification check passed.\n");
    }
}

void printVector(const float* vec, const unsigned int size) {
    printf("{");
    for(int i = 0; i < size-1; i++) {
        printf("%.2f, ", vec[i]); 
    }
    printf("%.2f}", vec[size-1]);
}

void printMatrix(const float* c, const unsigned int N, const unsigned int M) {

    printf("{"); 
    for(int i = 0; i < N-1; i++) {
        printVector(&c[i*M], M);
        printf(",\n");
    }
    printVector(&c[N * (M-1)], M);
    printf("}\n");

}

void printResultsVector(const float* c, const float* x, const float* y, const float a, 
        const unsigned int size) {
    
    printVector(y, size);
    printf(" * %.2f + ", a);
    printVector(x, size);
    printf(" = ");
    printVector(c, size);
    printf("\n");

}

void printResultsMatrix(const float*  c, const float* x, const float* y, 
        const unsigned int M, const unsigned int N) {
    
    printMatrix(x, N, M);
    printf("+\n");
    printMatrix(y, N, M);
    printf("=\n");
    printMatrix(c, N, M);

}

void verifyCudaMatrix(const float*  matAddCpu, const float* matAddCuda, 
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

void initializeGrid(float* x, const unsigned int N, const unsigned int M, const unsigned int P, 
        const unsigned int maxElementSize) {

    for(int k = 0; k < P; k++) {
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < M; j++) {
                    x[k * N * M + i * M + j] = (float)rand()/((float)RAND_MAX/maxElementSize); 
            }
        }
    }
}

void verifyCudaGrid(const float* gridAddCpu, const float* gridAddCuda, const unsigned int N, 
        const unsigned int M, const unsigned int P) {

    const float tol = 1e-7; 
    bool error = false;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            for(int k = 0; k < P; k++) {
                if(abs(gridAddCpu[k * M * N + i * M + j] 
                            - gridAddCuda[k * M * N + i * M + j]) > tol) {
                    error = true;
                    printf("CUDA matrix addition computation failed at index (%i, %i)\n", i, j);
                    printf("CPU: %f, GPU: %f\n", gridAddCpu[k * M * N + i * M + j], 
                            gridAddCuda[k * M * N + i * M + j]);
                    break;
                }
            }
        }
    }
    if(error == false) {
        printf("CUDA matrix addition verification check passed.\n");
    }

}

float* allocateGrid(const unsigned int M, const unsigned int N, const unsigned int P) {

    float* x = (float*)malloc(sizeof(float*)*N*M*P); 
    if(x == NULL) {
        printf("Malloc failed for array x.\n");
        exit(-1);
    }

    return x;

}

void gridAddWithCpu(float* c, const float* x, const float*  y, 
        const unsigned int N, const unsigned int M, const unsigned int P) {

    for(int k = 0; k < P; k++) {
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < M; j++) {
                    c[k * M * N + i * M + j] = x[k * M * N + i * M + j] 
                    + y[k * M * N + i * M + j];
            }
        }
    }
}

void saxpyWithCpu(float* c, const float *x, const float *y, const float a, 
        const unsigned int size) {
    for(int i = 0; i < size; i++) {
        c[i] = a * y[i] + x[i]; 
    }
}

void matAddWithCpu(float* c, const float* x, const float*  y, 
        const unsigned int N, const unsigned int M) {

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            c[i * M + j] = x[i * M + j] + y[i * M + j];
        }
    }

}

void parsArgs(args_t* args, int argc, char* argv[]) {

    if(argc > 7) {
        printf("Wrong number of args.\n");
        exit(-1);
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
                exit(-1);
            }
        }
    }

}
