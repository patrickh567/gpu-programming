
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "cuda_error.h"
#include <sstream>
#include "time.h"

#define STB_IMAGE_IMPLEMENTATION // this is needed
#include "../util/stb_image.h"  // download from class website files
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../util/stb_image_write.h"  // download from class website files

// #include your error-check macro header file here

// global gaussian blur filter coefficients array here
#define BLUR_FILTER_WIDTH 9  // 9x9 (square) Gaussian blur filter
const float BLUR_FILT[81] = { 0.1084,0.1762,0.2494,0.3071,0.3292,0.3071,0.2494,0.1762,0.1084,0.1762,0.2865,0.4054,0.4994,0.5353,0.4994,0.4054,0.2865,0.1762,0.2494,0.4054,0.5738,0.7066,0.7575,0.7066,0.5738,0.4054,0.2494,0.3071,0.4994,0.7066,0.8703,0.9329,0.8703,0.7066,0.4994,0.3071,0.3292,0.5353,0.7575,0.9329,1.0000,0.9329,0.7575,0.5353,0.3292,0.3071,0.4994,0.7066,0.8703,0.9329,0.8703,0.7066,0.4994,0.3071,0.2494,0.4054,0.5738,0.7066,0.7575,0.7066,0.5738,0.4054,0.2494,0.1762,0.2865,0.4054,0.4994,0.5353,0.4994,0.4054,0.2865,0.1762,0.1084,0.1762,0.2494,0.3071,0.3292,0.3071,0.2494,0.1762,0.1084};

void blurWithCuda(void (*kernel)(unsigned char*, const unsigned char*, const unsigned char*, const unsigned int, const unsigned int, const unsigned int),
        unsigned char* h_blurred_image, const unsigned char* h_input_image, 
        const unsigned int image_size, const unsigned int x_cols, const unsigned int y_rows);

// DEFINE your CUDA blur kernel function(s) here
// blur kernel #1 - global memory only
__global__ void blurFilterKernel(unsigned char* d_blurred_image, const unsigned char* d_input_image, 
        const unsigned char* d_blur_filter, const unsigned int x_cols, const unsigned int y_rows, 
        const unsigned int filter_size) {
    
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int blur_size = BLUR_FILTER_WIDTH / 2;
    
    if((idx_x < x_cols) && (idx_y < y_rows)) {
        int pixVal = 0;
        float normalizer = 0;

        for(int blurRow = -blur_size; blurRow < blur_size + 1; blurRow++) {
            for(int blurCol = -blur_size; blurCol < blur_size + 1; blurCol++) {
                int curRow = idx_x + blurRow;
                int curCol = idx_y + blurCol;
                unsigned int filter_idx;
                unsigned int img_idx;
                if(curRow > -1 && curRow < y_rows && curCol > -1 && curCol < x_cols) {
                    filter_idx = (blurRow + blur_size) * BLUR_FILTER_WIDTH + (blurCol + blur_size);
                    normalizer += d_blur_filter[filter_idx];
                    img_idx = curRow * x_cols + curCol;
                    pixVal += (int)(d_input_image[img_idx] * d_blur_filter[filter_idx]);
                }
            }
        }
        d_blurred_image[idx_x * x_cols + idx_y] = (unsigned char)(pixVal / normalizer);
    }
}

// blur kernel #2 - device shared memory (static alloc)
__global__ void blurFilterKernelStaticSharedMem(unsigned char* d_blurred_image, const unsigned char* d_input_image, 
        const unsigned char* d_blur_filter, const unsigned int x_cols, const unsigned int y_rows, 
        const unsigned int filter_size) {
    
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int blur_size = BLUR_FILTER_WIDTH / 2;

    __shared__ float ds_blurFilt[BLUR_FILTER_WIDTH][BLUR_FILTER_WIDTH];
    
    // Load filter into shared memory
    for(int ty = threadIdx.y; ty < BLUR_FILTER_WIDTH; ty += blockDim.y) { 
        for(int tx = threadIdx.x; tx < BLUR_FILTER_WIDTH; tx += blockDim.x) { 
            ds_blurFilt[ty][tx] = d_blur_filter[ty * blockDim.x + tx];
        }
    }

    // Wait for kernel load to complete
    __syncthreads();
    
    if((idx_x < x_cols) && (idx_y < y_rows)) {
        int pixVal = 0;
        float normalizer = 0;

        for(int blurRow = -blur_size; blurRow < blur_size + 1; blurRow++) {
            for(int blurCol = -blur_size; blurCol < blur_size + 1; blurCol++) {
                int curRow = idx_x + blurRow;
                int curCol = idx_y + blurCol;
                if(curRow > -1 && curRow < y_rows && curCol > -1 && curCol < x_cols) {
                    float coeff = ds_blurFilt[blurRow + blur_size][blurCol + blur_size];
                    normalizer += coeff;
                    unsigned int img_idx = curRow * x_cols + curCol;
                    pixVal += (int)(d_input_image[img_idx] * coeff);
                }
            }
        }
        d_blurred_image[idx_x * x_cols + idx_y] = (unsigned char)(pixVal / normalizer);
    }
}

// blur kernel #2 - device shared memory (dynamic alloc)
extern __shared__ float s_filter[];

__global__ void blurFilterKernelDynamicSharedMem(unsigned char* d_blurred_image, const unsigned char* d_input_image, 
        const unsigned char* d_blur_filter, const unsigned int x_cols, const unsigned int y_rows, 
        const unsigned int filter_size) {
    
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int blur_size = BLUR_FILTER_WIDTH / 2;

    // Load filter into shared memory
    for(int i = threadIdx.x + blockDim.x * threadIdx.y; i < filter_size; i += blockDim.x + blockDim.y) {
        s_filter[i] = d_blur_filter[i];
    }

    // Wait for kernel load to complete
    __syncthreads();
    
    if((idx_x < x_cols) && (idx_y < y_rows)) {
        int pixVal = 0;
        float normalizer = 0;

        for(int blurRow = -blur_size; blurRow < blur_size + 1; blurRow++) {
            for(int blurCol = -blur_size; blurCol < blur_size + 1; blurCol++) {
                int curRow = idx_x + blurRow;
                int curCol = idx_y + blurCol;
                if(curRow > -1 && curRow < y_rows && curCol > -1 && curCol < x_cols) {
                    unsigned int filter_idx = (blurRow + blur_size) * BLUR_FILTER_WIDTH + (blurCol + blur_size);
                    float coeff = s_filter[filter_idx];
                    normalizer += coeff;
                    unsigned int img_idx = curRow * x_cols + curCol;
                    pixVal += (int)(d_input_image[img_idx] * coeff);
                }
            }
        }
        d_blurred_image[idx_x * x_cols + idx_y] = (unsigned char)(pixVal / normalizer);
    }
}


// EXTRA CREDIT
// define host sequential blur-kernel routine
void sequentialBlurKernel(unsigned char* h_blurred_image, const unsigned char* h_input_image, 
        const unsigned int x_cols, const unsigned int y_rows, const unsigned int filter_size) {

    const int blur_size = BLUR_FILTER_WIDTH / 2;
    
    for(int idx_y = 0; idx_y < y_rows; idx_y++) { 
        for(int idx_x = 0; idx_x < x_cols; idx_x++) { 
            int pixVal = 0;
            float normalizer = 0;
            for(int blurRow = -blur_size; blurRow < blur_size + 1; blurRow++) {
                for(int blurCol = -blur_size; blurCol < blur_size + 1; blurCol++) {
                    int curRow = (idx_y + blurCol) * x_cols;
                    int curCol = idx_x + blurRow;
                    if(curRow > -1 && curRow < y_rows && curCol > -1 && curCol < x_cols) {
                        unsigned int filter_idx = (blurRow + blur_size) * BLUR_FILTER_WIDTH + (blurCol + blur_size);
                        float coeff = BLUR_FILT[filter_idx];
                        normalizer += coeff;
                        unsigned int img_idx = curRow * x_cols + curCol;
                        pixVal += (int)(h_input_image[img_idx] * coeff);
                    }
                }
            }
            unsigned int blurred_image_idx = idx_y * x_cols + idx_x;
            if(blurred_image_idx > x_cols * y_rows) {
                printf("idx_y * x_cols = %i\n", idx_y * x_cols);
                printf("idx_x = %i\n", idx_x);
            }
            h_blurred_image[blurred_image_idx] = (unsigned char)(pixVal / normalizer);
        }
    }
}

int main() {
     // read input image from file - be aware of image pixel bit-depth and resolution (horiz x vertical)
    char filenames[4][30] = {"./hw2_testimage1.png", "./hw2_testimage2.png", "./hw2_testimage3.png", "./hw2_testimage4.png"};
    char outfilenames[4][30] = {"./hw2_outimage1.png", "./hw2_outimage2.png", "./hw2_outimage3.png", "./hw2_outimage4.png"};
    char static_outfilenames[4][30] = {"./hw2_static_outimage1.png", "./hw2_static_outimage2.png", "./hw2_static_outimage3.png", "./hw2_static_outimage4.png"};
    char dynamic_outfilenames[4][30] = {"./hw2_dynamic_outimage1.png", "./hw2_dynamic_outimage2.png", "./hw2_dynamic_outimage3.png", "./hw2_dynamic_outimage4.png"};
    for(int i = 0; i < 4; i++) {
        char* filename = filenames[i];
        printf("Blurring %s\n", filename);
        printf("------------------------------------------ \n\n");
        int x_cols = 0;
        int y_rows = 0;
        int n_pixdepth = 0;

        unsigned char* h_input_image = stbi_load(filename, &x_cols, &y_rows, &n_pixdepth, 1);
        int image_size = x_cols * y_rows * sizeof(unsigned char);
        unsigned char* h_blurred_image = (unsigned char*)malloc(image_size);
        const unsigned int filter_size = sizeof(BLUR_FILT) / sizeof(float);
        
        printf("Running blur kernel using global memory...\n");
        blurWithCuda(&blurFilterKernel, h_blurred_image, h_input_image, image_size, x_cols, y_rows);
        stbi_write_png(outfilenames[i], x_cols, y_rows, 1, h_blurred_image, x_cols * n_pixdepth);
        free(h_blurred_image);
        h_blurred_image = (unsigned char*)malloc(image_size);
        printf("\nRunning blur kernel using static shared memory...\n");
        blurWithCuda(&blurFilterKernelStaticSharedMem, h_blurred_image, h_input_image, image_size, x_cols, y_rows);
        stbi_write_png(static_outfilenames[i], x_cols, y_rows, 1, h_blurred_image, x_cols * n_pixdepth);
        free(h_blurred_image);
        h_blurred_image = (unsigned char*)malloc(image_size);
        printf("\nRunning blur kernel using dynamic shared memory...\n");
        blurWithCuda(&blurFilterKernelDynamicSharedMem, h_blurred_image, h_input_image, image_size, x_cols, y_rows);
        stbi_write_png(dynamic_outfilenames[i], x_cols, y_rows, 1, h_blurred_image, x_cols * n_pixdepth);
        free(h_blurred_image);
        h_blurred_image = (unsigned char*)malloc(image_size);
        printf("\nRunning CPU sequential blur kernel...\n");
        clock_t cpu_timer_begin = clock();
        sequentialBlurKernel(h_blurred_image, h_input_image, x_cols, y_rows, filter_size);
        clock_t cpu_timer_end = clock();
        double time_spent_cpu_timer = (double)((cpu_timer_end - cpu_timer_begin) / ((double)1000));
        printf("CPU Timer: %f ms\n\n", time_spent_cpu_timer);
        free(h_input_image);
        free(h_blurred_image);
    }
    return 0;

}
    
void blurWithCuda(void (*kernel)(unsigned char*, const unsigned char*, const unsigned char*, const unsigned int, const unsigned int, const unsigned int), 
        unsigned char* h_blurred_image, const unsigned char* h_input_image, 
        const unsigned int image_size, const unsigned int x_cols, const unsigned int y_rows) {

    unsigned char* d_blurred_image;
    unsigned char* d_input_image;
    unsigned char* d_blur_filter;
    const unsigned int filter_size = sizeof(BLUR_FILT) / sizeof(float);

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCuda(cudaSetDevice(0));
    
    // START timer #1
    clock_t timer1_begin = clock();

    // Allocate GPU buffers for three vectors (two input, one output)    .
    checkCuda(cudaMalloc((void**)&d_blurred_image, image_size));
    checkCuda(cudaMalloc((void**)&d_input_image, image_size));
    checkCuda(cudaMalloc((void**)&d_blur_filter, sizeof(BLUR_FILT)));

    // Copy input vectors from host memory to GPU buffers.
    checkCuda(cudaMemcpy(d_input_image, h_input_image, image_size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_blur_filter, &BLUR_FILT, BLUR_FILTER_WIDTH * BLUR_FILTER_WIDTH, 
            cudaMemcpyHostToDevice));
    
    dim3 DimGrid((y_rows-1)/16+1, (x_cols-1)/16+1);
    dim3 DimBlock(16, 16, 1);
    int shared_mem_size = filter_size;
    
    // START timer #2
    clock_t timer2_begin = clock();
 
    // Launch kernel
    (*kernel)<<<DimGrid, DimBlock, shared_mem_size, 0>>>(d_blurred_image, d_input_image, d_blur_filter, x_cols, y_rows, filter_size);
    //(*kernel)<<<1, 1>>>(d_blurred_image, d_input_image, d_blur_filter, x_cols, y_rows, filter_size);

    // Check for any errors launching the kernel
    checkCuda(cudaGetLastError());
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCuda(cudaDeviceSynchronize());
    
    // STOP timer #2
    clock_t timer2_end = clock();

    // Copy output vector from GPU buffer to host memory.
    checkCuda(cudaMemcpy(h_blurred_image, d_blurred_image, image_size, cudaMemcpyDeviceToHost));
    
    // STOP timer #1
    clock_t timer1_end = clock();
    
    double time_spent_timer1 = (double)((timer1_end - timer1_begin) / ((double)1000));
    double time_spent_timer2 = (double)((timer2_end - timer2_begin) / ((double)1000));
    printf("Timer 1: %f ms\n", time_spent_timer1);
    printf("Timer 2: %f ms\n", time_spent_timer2);

    checkCuda(cudaFree(d_blurred_image));
    checkCuda(cudaFree(d_input_image));
    checkCuda(cudaFree(d_blur_filter));

}
