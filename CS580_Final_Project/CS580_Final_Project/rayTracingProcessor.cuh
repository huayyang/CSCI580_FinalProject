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
__device__ bool allHit(int cur_node, Object* objects, float3 pos, float3 dir, float3* hitPos, KDTriangle* hitTriangle, float* tmin, KDNode_CUDA * KDTree_GPU, int* TriangleIndexArray_GPU, bool* isFront, int currentIndex,float3* normal,uchar4* color, uchar1* materialIndex);
__device__ bool ballHit(int currentIndex,float3 pos, float3 dir,float3* hitPos, float3* normal, uchar4* color, uchar1* materialIndex, bool* isFront,float * hitDis);

__device__ bool CheckAlreadAdded(KDNode_Photon_GPU* close_set, int top, KDNode_Photon_GPU* node);
//__device__ float KDTreeKNNSearch(KDNode_Photon_GPU* KDNodePhotonArrayTree_GPU, int cur, float3 target, float3* distance, int rad, float3* near);
__device__ void KDTreeKNNSearch(KDNode_Photon_GPU* pNode, int cur, float3 point, float3* res, float& nMinDis, Photon* distance, int rad, int* searched, int index_i, int index_j, int*pqtop);
__device__ float GetDistanceSquare(float3 pointA, float3 pointB);
static __device__ int* pqtop;

static __device__ KDNode_Photon_GPU* KDNodePhotonArrayTree_GPU;// Array KDTree
/* KDTree -- Over */

static  __device__ Object *objects_CUDA;
static  __device__ Material *materialBuffer_CUDA;
static __device__ Camera* mainCamera_CUDA;

static __device__ Photon *photonBuffer_CUDA;

/* functions */
//void rayTracingCuda(uchar4 * pixels, int count, Object* objects, Photon* photons, Material* materials, KDNode_CUDA * KDTree_GPU, int* TriangleIndexArray_GPU, KDNode_Photon_GPU* KDNodePhotonArrayTree_GPU, Photon* KDPhotonArray_GPU, KDNode_Photon_GPU* KDNodePhotonArrayTree_CPU);
void rayTracingCuda(uchar4 * pixels, int count, Object* objects, Photon* photons, Material* materials, KDNode_CUDA * KDTree_GPU, int* TriangleIndexArray_GPU);

__device__ float dotProduct(float3 a, float3 b);
__device__ float3 crossProduct(float3 a, float3 b);
__device__ bool isInside(float3 point, float3* triangle);
__device__ float3 normalize(float3 vector);
__device__ float hitSurface(float3* vertex, float3 pos, float3 dir, float3* pho,bool* isFront);

#endif