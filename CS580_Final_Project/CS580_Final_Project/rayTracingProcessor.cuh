#ifndef RTP_CUH
#define RTP_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "KDTree.h"
#include <stdio.h>
#include <thrust/device_vector.h>
#include "defines.h"


/* KDTree */
/*Line Tracing*/
__device__ bool ClipLine(int d, const BoundingBox& aabbBox, const float3& v0, const float3& v1, float& f_low, float& f_high);
__device__ bool LineAABBIntersection(const BoundingBox& aabbBox, const float3& v0, const float3& dir, float3& vecIntersection, float& flFraction);
__device__ bool KDTreeHit(int cur_node, Object* objects, float3 pos, float3 dir, float3* hitPos, KDTriangle* hitTriangle, float* tmin, KDNode_CUDA * KDTree_GPU, int* TriangleIndexArray_GPU, bool* isFront, int currentIndex);
/* KDTree -- Over */

static  __device__ Object *objects_CUDA;
static  __device__ Material *materialBuffer_CUDA;
static __device__ Camera* mainCamera_CUDA;

static __device__ Photon *photonBuffer_CUDA;

/* functions */
void rayTracingCuda(uchar4 * pixels, int count, Object* objects, Photon* photons, Material* materials, KDNode_CUDA * KDTree_GPU, int* TriangleIndexArray_GPU);


__device__ float dotProduct(float3 a, float3 b);
__device__ float3 crossProduct(float3 a, float3 b);
__device__ bool isInside(float3 point, float3* triangle);
__device__ float3 normalize(float3 vector);
__device__ float hitSurface(float3* vertex, float3 pos, float3 dir, float3* pho,bool* isFront);

#endif