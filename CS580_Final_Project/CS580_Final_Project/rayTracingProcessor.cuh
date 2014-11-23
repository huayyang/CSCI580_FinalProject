#ifndef RTP_CUH
#define RTP_CUH

#include "KDTree.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "defines.h"

static __device__ float3 *vertexBuffer_CUDA,*normalBuffer_CUDA;
static __device__ uchar4 *colorBuffer_CUDA;
static __device__ Material *materialBuffer_CUDA;
static __device__ uchar1 *materialIndexBuffer_CUDA;

static __device__ Photon *photonBuffer_CUDA;
static __device__ KDTree::KDTriangle *kdTriangles_CUDA;
static __device__ KDTree::KDNode* KDTreeRoot_CUDA;

/* functions */
void rayTracingCuda(uchar4 * pixels,int count,float3* vertex,float3* normal,uchar4* color, Photon* photons, Material* materials, uchar1* materialIndex);
__device__ float dotProduct(float3 a, float3 b);
__device__ float3 crossProduct(float3 a, float3 b);
__device__ bool isInside(float3 point, float3* triangle);
__device__ float3 normalize(float3 vector);
__device__ float hitSurface(float3* vertex, float3 pos, float3 dir, float3* pho);

#endif