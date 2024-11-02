#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <iostream>

int main(void) {
    int dev_count;
    cudaDeviceProp dev_prop;
    cudaGetDeviceCount(&dev_count);
    std::cout << "Number of devices: " << dev_count << std::endl;
    cudaGetDeviceProperties(&dev_prop, 0);
    std::cout << "Max threads per block: " << dev_prop.maxThreadsPerBlock << std::endl;
    std::cout << "Number of SMs: " << dev_prop.multiProcessorCount << std::endl;
    std::cout << "Clock rate: " << ((float)dev_prop.clockRate) / 1000000 << "GHz" << std::endl;
    std::cout << "Max threads dim x: " << dev_prop.maxThreadsDim[0] << std::endl;
    std::cout << "Max threads dim y: " << dev_prop.maxThreadsDim[1] << std::endl;
    std::cout << "Max threads dim z: " << dev_prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max blocks dim x: " << dev_prop.maxGridSize[0] << std::endl;
    std::cout << "Max blocks dim y: " << dev_prop.maxGridSize[1] << std::endl;
    std::cout << "Max blocks dim z: " << dev_prop.maxGridSize[2] << std::endl;
    std::cout << "Warp Size: " << dev_prop.warpSize << std::endl;
    return 0;
}
