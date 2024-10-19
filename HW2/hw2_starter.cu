
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "cuda_error.h"

#define STB_IMAGE_IMPLEMENTATION // this is needed
#include "../util/stb_image.h"  // download from class website files
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../util/stb_image_write.h"  // download from class website files

// #include your error-check macro header file here

// global gaussian blur filter coefficients array here
#define BLUR_FILTER_WIDTH 9  // 9x9 (square) Gaussian blur filter
const float BLUR_FILT[81] = { 0.1084,0.1762,0.2494,0.3071,0.3292,0.3071,0.2494,0.1762,0.1084,0.1762,0.2865,0.4054,0.4994,0.5353,0.4994,0.4054,0.2865,0.1762,0.2494,0.4054,0.5738,0.7066,0.7575,0.7066,0.5738,0.4054,0.2494,0.3071,0.4994,0.7066,0.8703,0.9329,0.8703,0.7066,0.4994,0.3071,0.3292,0.5353,0.7575,0.9329,1.0000,0.9329,0.7575,0.5353,0.3292,0.3071,0.4994,0.7066,0.8703,0.9329,0.8703,0.7066,0.4994,0.3071,0.2494,0.4054,0.5738,0.7066,0.7575,0.7066,0.5738,0.4054,0.2494,0.1762,0.2865,0.4054,0.4994,0.5353,0.4994,0.4054,0.2865,0.1762,0.1084,0.1762,0.2494,0.3071,0.3292,0.3071,0.2494,0.1762,0.1084};

void blurWithCuda(unsigned char* h_blurred_image, const unsigned char* h_input_image, 
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
        int pixels = 0;

        for(int blurRow = -blur_size; blurRow < blur_size + 1; blurRow++) {
            for(int blurCol = -blur_size; blurCol < blur_size + 1; blurCol++) {
                int curRow = idx_x + blurRow;
                int curCol = idx_y + blurCol;
                if(curRow > -1 && curRow < y_rows && curCol > -1 && curCol < x_cols) {
                    pixVal += d_input_image[curRow * x_cols + curCol];
                    pixels++;
                }
            }
        }
        d_blurred_image[idx_x * x_cols + idx_y] = (unsigned char)(pixVal / pixels);
    }
}

// blur kernel #2 - device shared memory (static alloc)
// blur kernel #2 - device shared memory (dynamic alloc)


// EXTRA CREDIT
// define host sequential blur-kernel routine


int main() {
     // read input image from file - be aware of image pixel bit-depth and resolution (horiz x vertical)
    const char filename[] = "./hw2_testimage1.png";
    int x_cols = 0;
    int y_rows = 0;
    int n_pixdepth = 0;
    unsigned char* imgData = stbi_load(filename, &x_cols, &y_rows, &n_pixdepth, 1);
    int imgSize = x_cols * y_rows * sizeof(unsigned char);

    unsigned char* h_blurred_image = (unsigned char*)malloc(imgSize);

    blurWithCuda(h_blurred_image, imgData, imgSize, x_cols, y_rows);

    // setup additional host variables, allocate host memory as needed

    // START timer #1

    // allocate device memory

    // copy host data to device

    // START timer #2
    // launch kernel --- use appropriate heuristics to determine #threads/block and #blocks/grid to ensure coverage of your 2D data range

    // Check for any errors launching the kernel
    //cudaStatus = cudaGetLastError();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    //    goto Error;
    //}

    // call cudaDeviceSynchronize() to wait for the kernel to finish, and return
    // any errors encountered during the launch.
    
    // STOP timer #2
    // 
    // retrieve result data from device back to host

    // STOP timer #1

    // cudaDeviceReset( ) must be called in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

       // save result output image data to file
    const char imgFileOut[] = "./hw2_outimage1.png";
    stbi_write_png(imgFileOut, x_cols, y_rows, 1, h_blurred_image, x_cols * n_pixdepth);


    // EXTRA CREDIT:
    // start timer #3
    // run host sequential blur routine
    // stop timer #3

    // retrieve and save timer results (write to console or file)
 
//Error:  // assumes error macro has a goto Error statement

    // free host and device memory

    return 0;
}

void blurWithCuda(unsigned char* h_blurred_image, const unsigned char* h_input_image, 
        const unsigned int image_size, const unsigned int x_cols, const unsigned int y_rows) {

    unsigned char* d_blurred_image;
    unsigned char* d_input_image;
    unsigned char* d_blur_filter;
    const unsigned int filter_size = sizeof(BLUR_FILT) / sizeof(float);

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCuda(cudaSetDevice(0));

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

    blurFilterKernel<<<DimGrid,DimBlock>>>(d_blurred_image, d_input_image, d_blur_filter, 
            x_cols, y_rows, filter_size);

    // Check for any errors launching the kernel
    checkCuda(cudaGetLastError());
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCuda(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    checkCuda(cudaMemcpy(h_blurred_image, d_blurred_image, image_size, cudaMemcpyDeviceToHost));

    checkCuda(cudaFree(d_blurred_image));
    checkCuda(cudaFree(d_input_image));
    checkCuda(cudaFree(d_blur_filter));

}
