#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "microBench.fatbin.c"
extern void __device_stub__Z21registerBankConflictsPKfS0_Pf(const float *, const float *, float *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z21registerBankConflictsPKfS0_Pf(const float *__par0, const float *__par1, float *__par2){__cudaLaunchPrologue(3);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaLaunch(((char *)((void ( *)(const float *, const float *, float *))registerBankConflicts)));}
# 17 "microBench.cu"
void registerBankConflicts( const float *__cuda_0,const float *__cuda_1,float *__cuda_2)
# 17 "microBench.cu"
{__device_stub__Z21registerBankConflictsPKfS0_Pf( __cuda_0,__cuda_1,__cuda_2);
# 38 "microBench.cu"
}
# 1 "microBench.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T4) {  __nv_dummy_param_ref(__T4); __nv_save_fatbinhandle_for_managed_rt(__T4); __cudaRegisterEntry(__T4, ((void ( *)(const float *, const float *, float *))registerBankConflicts), _Z21registerBankConflictsPKfS0_Pf, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
