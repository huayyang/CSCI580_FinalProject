#ifndef RTP_CUH
#define RTP_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "defines.h"

static __device__ float3 *vertexBuffer_CUDA,*normalBuffer_CUDA;
static __device__ uchar4 *colorBuffer_CUDA;


void rayTracingCuda(uchar4 * pixels,int count,float3* vertex,float3* normal,uchar4* color);
static __device__ Photon *photonBuffer_CUDA;
void rayTracingCuda2(uchar4 * pixels,int count,float3* vertex,float3* normal,uchar4* color, Photon* photons);


#endif