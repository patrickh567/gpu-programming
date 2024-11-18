#ifndef __SKIP_INTERNAL_FATBINARY_HEADERS
#include "fatbinary_section.h"
#endif
#define __CUDAFATBINSECTION  ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION  ".nv_fatbin"
asm(
".section .nv_fatbin, \"a\"\n"
".align 8\n"
"fatbinData:\n"
".quad 0x00100001ba55ed50,0x0000000000000940,0x0000004001010002,0x0000000000000820\n"
".quad 0x0000000000000000,0x0000003400010007,0x0000000000000000,0x0000000000000011\n"
".quad 0x0000000000000000,0x0000000000000000,0x33010102464c457f,0x0000000000000007\n"
".quad 0x0000007300be0002,0x0000000000000000,0x0000000000000000,0x00000000000005e0\n"
".quad 0x0000004000340534,0x0001000900400000,0x7472747368732e00,0x747274732e006261\n"
".quad 0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261,0x666e692e766e2e00\n"
".quad 0x2e747865742e006f,0x6967657231325a5f,0x6b6e614272657473,0x7463696c666e6f43\n"
".quad 0x505f3053664b5073,0x6e692e766e2e0066,0x7231325a5f2e6f66,0x4272657473696765\n"
".quad 0x6c666e6f436b6e61,0x53664b5073746369,0x766e2e0066505f30,0x2e6465726168732e\n"
".quad 0x6967657231325a5f,0x6b6e614272657473,0x7463696c666e6f43,0x505f3053664b5073\n"
".quad 0x6f632e766e2e0066,0x2e30746e6174736e,0x6967657231325a5f,0x6b6e614272657473\n"
".quad 0x7463696c666e6f43,0x505f3053664b5073,0x65722e766e2e0066,0x6e6f697463612e6c\n"
".quad 0x72747368732e0000,0x7274732e00626174,0x6d79732e00626174,0x6d79732e00626174\n"
".quad 0x646e68735f626174,0x6e692e766e2e0078,0x7231325a5f006f66,0x4272657473696765\n"
".quad 0x6c666e6f436b6e61,0x53664b5073746369,0x65742e0066505f30,0x7231325a5f2e7478\n"
".quad 0x4272657473696765,0x6c666e6f436b6e61,0x53664b5073746369,0x766e2e0066505f30\n"
".quad 0x5a5f2e6f666e692e,0x7473696765723132,0x6f436b6e61427265,0x50737463696c666e\n"
".quad 0x0066505f3053664b,0x726168732e766e2e,0x7231325a5f2e6465,0x4272657473696765\n"
".quad 0x6c666e6f436b6e61,0x53664b5073746369,0x766e2e0066505f30,0x6e6174736e6f632e\n"
".quad 0x7231325a5f2e3074,0x4272657473696765,0x6c666e6f436b6e61,0x53664b5073746369\n"
".quad 0x61705f0066505f30,0x2e766e2e006d6172,0x697463612e6c6572,0x0000000000006e6f\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0008000300000054\n"
".quad 0x0000000000000000,0x0000000000000000,0x00070003000000d4,0x0000000000000000\n"
".quad 0x0000000000000000,0x000600030000010b,0x0000000000000000,0x0000000000000000\n"
".quad 0x0008101200000032,0x0000000000000000,0x0000000000000040,0x0000000400082f04\n"
".quad 0x0008230400000002,0x0000000000000004,0x0000000400081204,0x0008110400000000\n"
".quad 0x0000000000000004,0x0000007300043704,0x00002a0100003001,0x0000000200080a04\n"
".quad 0x0018190300180140,0x00000000000c1704,0x0021f00000100002,0x00000000000c1704\n"
".quad 0x0021f00000080001,0x00000000000c1704,0x0021f00000000000,0x00041c0400ff1b03\n"
".quad 0x0000000000000030,0x000000000000004b,0x222f0a1008020200,0x0000000008000000\n"
".quad 0x0000000008080000,0x0000000008100000,0x0000000008180000,0x0000000008200000\n"
".quad 0x0000000008280000,0x0000000008300000,0x0000000008380000,0x0000000008000001\n"
".quad 0x0000000008080001,0x0000000008100001,0x0000000008180001,0x0000000008200001\n"
".quad 0x0000000008280001,0x0000000008300001,0x0000000008380001,0x0000000008000002\n"
".quad 0x0000000008080002,0x0000000008100002,0x0000000008180002,0x0000000008200002\n"
".quad 0x0000000008280002,0x0000000008300002,0x0000000008380002,0x0000002c14000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x001fbc00fde007f6,0x4c98078000870001\n"
".quad 0x50b0000000070f00,0x50b0000000070f00,0x001ffc00ffe007ed,0x50b0000000070f00\n"
".quad 0xe30000000007000f,0xe2400fffff87000f,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000300000001,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000040,0x00000000000000f1,0x0000000000000000\n"
".quad 0x0000000000000001,0x0000000000000000,0x000000030000000b,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000131,0x000000000000011a,0x0000000000000000\n"
".quad 0x0000000000000001,0x0000000000000000,0x0000000200000013,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000250,0x0000000000000078,0x0000000400000002\n"
".quad 0x0000000000000008,0x0000000000000018,0x7000000000000029,0x0000000000000000\n"
".quad 0x0000000000000000,0x00000000000002c8,0x0000000000000030,0x0000000000000003\n"
".quad 0x0000000000000004,0x0000000000000000,0x700000000000005a,0x0000000000000000\n"
".quad 0x0000000000000000,0x00000000000002f8,0x000000000000005c,0x0000000800000003\n"
".quad 0x0000000000000004,0x0000000000000000,0x7000000b000000e2,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000358,0x00000000000000d8,0x0000000000000000\n"
".quad 0x0000000000000008,0x0000000000000008,0x00000001000000b2,0x0000000000000002\n"
".quad 0x0000000000000000,0x0000000000000430,0x0000000000000158,0x0000000800000000\n"
".quad 0x0000000000000004,0x0000000000000000,0x0000000100000032,0x0000000000000006\n"
".quad 0x0000000000000000,0x00000000000005a0,0x0000000000000040,0x0200000400000003\n"
".quad 0x0000000000000020,0x0000000000000000,0x0000004801010001,0x0000000000000098\n"
".quad 0x0000004000000097,0x0000003400070005,0x0000000000000000,0x0000000000002011\n"
".quad 0x0000000000000000,0x000000000000011f,0x0000000000000000,0x762e1cf200010a13\n"
".quad 0x37206e6f69737265,0x677261742e0a352e,0x32355f6d73207465,0x7365726464612e0a\n"
".quad 0x3620657a69735f73,0x6973692dff002f34,0x746e652e20656c62,0x7231325a5f207972\n"
".quad 0x4272657473696765,0x6c666e6f436b6e61,0x53664b5073746369,0x702e0a2866505f30\n"
".quad 0x36752e206d617261,0x002d5f110f002f34,0x1f2200372c305f3f,0x290a325023003731\n"
".quad 0x746572a000de7b0a, 0x00000a0a7d0a0a3b\n"
".text\n");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[298];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const __fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) __attribute__ ((section (__CUDAFATBINSECTION)))= 
	{ 0x466243b1, 1, fatbinData, 0 };
#ifdef __cplusplus
}
#endif