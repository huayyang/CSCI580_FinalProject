#ifndef KDTREE_H
#define KDTREE_H

#include<vector>
#include "defines.h"

using namespace std;

/* KDTree */
class BoundingBox
{
public:
	float3 min, max;
	__device__ BoundingBox(){ min.x = min.y = min.z = INT_MAX; memset(&max, 0, sizeof(float3)); };
	__device__ BoundingBox& operator=(const BoundingBox &rhs)
	{
		this->min = rhs.min;
		this->max = rhs.max;
		return *this;
	}
	int longest_axis();
};


class KDTriangle
{
public:
	void generate_bounding_box();
	int index;
	BoundingBox bbox;
};
float3 get_midpoint(KDTriangle* triangle);


struct KDNode
{
public:
	BoundingBox bbox;
	KDNode* left, *right;
	vector<KDTriangle*> triangles;
	int depth;
	int stIndex, edIndex;
	KDNode* build(vector<KDTriangle*>& tris, int depth, int* TriangleIndexArray_CPU);
};

struct KDNode_CUDA
{
public:
	BoundingBox bbox;
	int stIndex, edIndex;
	int triangle_sz;
	int depth;
	bool isRoot;
};

int setKDNodeIndex(KDNode* root, int cur);
int TreeHeight(KDNode* root);
void copyKDTreeToArray(KDNode_CUDA* root, int cur, KDNode* KDTreeRoot_CPU);

static KDNode* KDTreeRoot_CPU;

static int* TriangleIndexArray_CPU;
static int TI_cur;
static __device__ int* TriangleIndexArray_GPU;
static KDNode_CUDA *KDTree_CPU;
static __device__ KDNode_CUDA *KDTree_GPU;


void expandBoundingBox(KDNode *node, vector<KDTriangle*>& tris);


/* KDTree-finish */

#endif