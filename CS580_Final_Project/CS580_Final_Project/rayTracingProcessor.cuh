#ifndef RTP_CUH
#define RTP_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

static __device__ float4 *vertexBuffer_CUDA,*normalBuffer_CUDA;
static __device__ uchar4 *colorBuffer_CUDA;

void rayTracingCuda(uchar4 * pixels,int count,float4* vertex,float4* normal,uchar4* color);

#endif