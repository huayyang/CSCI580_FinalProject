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
	int index;
	BoundingBox bbox;
	int stIndex, edIndex;
	int triangle_sz;
	int depth;
	int left, right;
};

int treeSize(KDNode* node);
int setKDNodeIndex(KDNode* root, int cur);
int TreeHeight(KDNode* root);
//void copyKDTreeToArray(KDNode_CUDA* root, int cur, KDNode* KDTreeRoot_CPU);
int copyKDTreeToArray(KDNode_CUDA* root, KDNode* KDTreeRoot_CPU, int copy_index);

static KDNode* KDTreeRoot_CPU;

static int* TriangleIndexArray_CPU;
static int TI_cur;
static __device__ int* TriangleIndexArray_GPU;
static KDNode_CUDA *KDTree_CPU;
static __device__ KDNode_CUDA *KDTree_GPU;


void expandBoundingBox(KDNode *node, vector<KDTriangle*>& tris);


/* KDTree-finish */

/* KDTree_Photon */
struct KDNode_Photon_CPU
{
	int split;
	Photon photon;
	vector<Photon*>photons;
	KDNode_Photon_CPU* left, *right;
};

struct KDNode_Photon_GPU
{
	int left, right, parent;
	int index;
	int split;
	Photon photon;
};

int Variance(vector<Photon*>photons);
int TreeHeight(KDNode_Photon_CPU* root);
void Photon_KDTree_Init(Photon* photons, KDNode_Photon_GPU* KDNodePhotonArrayTree_GPU);

KDNode_Photon_CPU* Photon_KDTreeBuild(vector<Photon*>photons, int depth);
static KDNode_Photon_CPU* KDNodePhoton_CPU;
static KDNode_Photon_GPU* KDNodePhotonArrayTree_CPU;

int treeSize(KDNode_Photon_CPU* KDTreeRoot_CPU);
int copyKDTreeToArray(KDNode_Photon_GPU* node_GPU, KDNode_Photon_CPU* KDTreeRoot_CPU, int copy_index);
//void copyKDTreeToArray(KDNode_Photon_GPU* root, int cur, KDNode_Photon_CPU* node);
/* KDTree_Photon-Finish */

#endif