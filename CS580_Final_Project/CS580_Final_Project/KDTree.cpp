#include "KDTree.h"
#include "global.h"

using namespace std;

#define eps 10e-6
#define zero(x) (((x)>0?(x):-(x))<eps)
#define offset 0.0001
#define equal(x,v) (((v - eps) < x) && (x <( v + eps)))

void KDTriangle::generate_bounding_box()
{
	float3 bmin, bmax;
	bmin.x = bmin.y = bmin.z = INT_MAX;
	memset(&bmax, 0, sizeof(float3));

	for (int i = 0; i < 3; ++i)
	{
		if (objects[index].vertex[i].x < bmin.x) bmin.x = objects[index].vertex[i].x;
		if (objects[index].vertex[i].x > bmax.x) bmax.x = objects[index].vertex[i].x;

		if (objects[index].vertex[i].y < bmin.y) bmin.y = objects[index].vertex[i].y;
		if (objects[index].vertex[i].y > bmax.y) bmax.y = objects[index].vertex[i].y;

		if (objects[index].vertex[i].z < bmin.z) bmin.z = objects[index].vertex[i].z;
		if (objects[index].vertex[i].z > bmax.z) bmax.z = objects[index].vertex[i].z;
	}
	bbox.max = bmax;
	bbox.min = bmin;
}

float3 get_midpoint(KDTriangle* triangle)
{
	float3 midpoint;
	memset(&midpoint, 0, sizeof(float3));

	midpoint.x = (triangle->bbox.min.x + triangle->bbox.max.x) / 2;
	midpoint.y = (triangle->bbox.min.y + triangle->bbox.max.y) / 2;
	midpoint.z = (triangle->bbox.min.z + triangle->bbox.max.z) / 2;

	return midpoint;
}

void expandBoundingBox(KDNode *node, vector<KDTriangle*>& tris)
{
	float3 bmin, bmax;
	bmin.x = bmin.y = bmin.z = INT_MAX;
	memset(&bmax, 0, sizeof(float3));

	for (int i = 0; i < tris.size(); ++i)
	{
		for (int j = 0; j<3; ++j)
		{
			if (bmin.x > objects[tris[i]->index].vertex[j].x)bmin.x = objects[tris[i]->index].vertex[j].x;
			if (bmin.y > objects[tris[i]->index].vertex[j].y)bmin.y = objects[tris[i]->index].vertex[j].y;
			if (bmin.z > objects[tris[i]->index].vertex[j].z)bmin.z = objects[tris[i]->index].vertex[j].z;

			if (bmax.x < objects[tris[i]->index].vertex[j].x)bmax.x = objects[tris[i]->index].vertex[j].x;
			if (bmax.y < objects[tris[i]->index].vertex[j].y)bmax.y = objects[tris[i]->index].vertex[j].y;
			if (bmax.z < objects[tris[i]->index].vertex[j].z)bmax.z = objects[tris[i]->index].vertex[j].z;
		}

	}

	if (equal(bmin.x, bmax.x))
	{
		bmin.x -= offset;
		bmax.x += offset;
	}
	else if (equal(bmin.y, bmax.y))
	{
		bmin.y -= offset;
		bmax.y += offset;
	}
	else if (equal(bmin.z, bmax.z))
	{
		bmin.z -= offset;
		bmax.z += offset;
	}
	node->bbox.max = bmax;
	node->bbox.min = bmin;
}

int BoundingBox::longest_axis()
{
	int axis = -1;
	int longest = 0;
	if ((max.x - min.x) > longest)
	{
		longest = (max.x - min.x);
		axis = 0;
	}
	if ((max.y - min.y) > longest)
	{
		longest = (max.y - min.y);
		axis = 1;
	}
	if ((max.z - min.z) > longest)
	{
		longest = (max.z - min.z);
		axis = 2;
	}
	return axis;
}

int setKDNodeIndex(KDNode* root, int cur)
{
	if (root == NULL)return 0;
	if (root->left == NULL || root->left->triangles.size() == 0 || root->right== NULL || root->right->triangles.size() == 0)
	{
		root->stIndex = cur;
		root->edIndex = cur + root->triangles.size()-1;
		return root->edIndex;
	}
	else
	{
		setKDNodeIndex(root->left, cur);
		root->stIndex = cur;
		root->edIndex = setKDNodeIndex(root->right, cur + root->left->triangles.size());
		return root->edIndex;
	}
}

int TreeHeight(KDNode* root)
{
	if (root == NULL || root->triangles.size() == 0)return 0;
	int heightLeft = 0, heightRight = 0;
	if (root->left != NULL)heightLeft = TreeHeight(root->left);
	if (root->right != NULL)heightRight = TreeHeight(root->right);
	return heightLeft > heightRight ? heightLeft + 1 : heightRight + 1;
};

int treeSize(KDNode* node)
{
	if (node == NULL)return 0;
	int left = 0, right = 0;
	if (node->left != NULL)left = treeSize(node->left);
	if (node->right != NULL)right = treeSize(node->right);
	return left + right + 1;
}

int copyKDTreeToArray(KDNode_CUDA* root, KDNode* KDTreeRoot_CPU, int copy_index)
{
	if (KDTreeRoot_CPU == NULL)return -1;
	root[copy_index].bbox = KDTreeRoot_CPU->bbox;
	root[copy_index].stIndex = KDTreeRoot_CPU->stIndex;
	root[copy_index].edIndex = KDTreeRoot_CPU->edIndex;
	root[copy_index].triangle_sz = root[copy_index].edIndex - root[copy_index].stIndex + 1;
	root[copy_index].depth = KDTreeRoot_CPU->depth;
	if (KDTreeRoot_CPU->left == NULL && KDTreeRoot_CPU->right == NULL)
	{
		root[copy_index].left = root[copy_index].right = -1;
		root[copy_index].index = copy_index;
		return copy_index;
	}
	if (KDTreeRoot_CPU->left != NULL)
	{
		root[copy_index].left = copyKDTreeToArray(root, KDTreeRoot_CPU->left, copy_index + 1);
	}
	else root[copy_index].left = -1;
	if (KDTreeRoot_CPU->right != NULL)
	{
		int left_size = treeSize(KDTreeRoot_CPU->left);
		root[copy_index].right = copyKDTreeToArray(root, KDTreeRoot_CPU->right, left_size + copy_index + 1);
	}
	else root[copy_index].right = -1;
	root[copy_index].index = copy_index;
	return copy_index;
}

KDNode* KDNode::build(vector<KDTriangle*>& tris, int depth, int* TriangleIndexArray_CPU)
{
	if (tris.size() == 0)
	{
		return NULL;
	}

	KDNode* node = new KDNode();
	node->triangles = tris;
	node->left = NULL;
	node->right = NULL;
	node->bbox = BoundingBox();
	node->depth = depth;

	KDTriangle* ptr = (tris[0]);
	if (tris.size() == 1)
	{
		for (int i = 0; i < tris.size(); ++i)
			TriangleIndexArray_CPU[TI_cur++] = tris[i]->index;

		node->bbox = (*tris[0]).bbox;
		node->left = NULL;
		node->right = NULL;
		return node;
	}

	node->bbox = (*tris[0]).bbox;

	expandBoundingBox(node, tris);

	float3 midpt;
	memset(&midpt, 0, sizeof(float3));

	for (int i = 0; i<tris.size(); ++i)
	{
		float3 midPoint = get_midpoint(tris[i]);
		midpt.x += midPoint.x * (1.0 / tris.size());
		midpt.y += midPoint.y * (1.0 / tris.size());
		midpt.z += midPoint.z * (1.0 / tris.size());
	}

	vector<KDTriangle*> left_tris;
	vector<KDTriangle*> right_tris;
	int axis = node->bbox.longest_axis();
	for (int i = 0; i < tris.size(); ++i)
	{
		switch (axis)
		{
		case 0:
			midpt.x <= get_midpoint(tris[i]).x ? right_tris.push_back(tris[i]) : left_tris.push_back(tris[i]);
			break;
		case 1:
			midpt.y <= get_midpoint(tris[i]).y ? right_tris.push_back(tris[i]) : left_tris.push_back(tris[i]);
			break;
		case 2:
			midpt.z <= get_midpoint(tris[i]).z ? right_tris.push_back(tris[i]) : left_tris.push_back(tris[i]);
			break;
		}
	}

	if (left_tris.size() != 0 && right_tris.size() != 0 && tris.size() > 16)
	{
		node->left = build(left_tris, depth + 1, TriangleIndexArray_CPU);
		node->right = build(right_tris, depth + 1, TriangleIndexArray_CPU);
	}
	else
	{
		for (int i = 0; i < tris.size(); ++i)
			TriangleIndexArray_CPU[TI_cur++] = tris[i]->index;

		node->left = NULL;
		node->right = NULL;
	}

	return node;
}

/* KDTree For Photon */
int Variance(vector<Photon*>photons)
{
	float variance[3], avg[3];
	memset(variance, 0, sizeof(float) * 3);
	memset(avg, 0, sizeof(float) * 3);

	/* calculate avg */
	for (int i = 0; i < photons.size(); ++i)
	{
		avg[0] += photons[i]->pos.x;
		avg[1] += photons[i]->pos.y;
		avg[2] += photons[i]->pos.z;
	}
	avg[0] /= photons.size();
	avg[1] /= photons.size();
	avg[2] /= photons.size();

	for (int i = 0; i < photons.size(); ++i)
	{
		variance[0] += (photons[i]->pos.x - avg[0]) * (photons[i]->pos.x - avg[0]);
		variance[1] += (photons[i]->pos.y - avg[1]) * (photons[i]->pos.y - avg[1]);
		variance[2] += (photons[i]->pos.z - avg[2]) * (photons[i]->pos.z - avg[2]);
	}

	variance[0] /= photons.size();
	variance[1] /= photons.size();
	variance[2] /= photons.size();

	float max = 0;
	int maxIndex = 0;
	for (int i = 0; i < 3;++i)
		if (max < variance[i])
		{
			max = variance[i];
			maxIndex = i;
		}
		return maxIndex;
}

bool cmp_x(const Photon* a, const Photon* b)
{
	return a->pos.x < b->pos.x;
}

bool cmp_y(const Photon* a, const Photon* b)
{
	return a->pos.y < b->pos.y;
}

bool cmp_z(const Photon* a, const Photon* b)
{
	return a->pos.z < b->pos.z;
}

//void Photon_KDTree_Init(Photon* photons, KDNode_Photon_GPU* KDNodePhotonArrayTree_GPU, Photon* KDPhotonArray_GPU)
//{
//	/* KDTree For Photon */
//	photonArray = (Photon*)malloc(PHOTON_NUM * sizeof(Photon));
//	memset(photonArray, 0, PHOTON_NUM * sizeof(Photon));
//
//	cudaMalloc(&KDPhotonArray_GPU, PHOTON_NUM * sizeof(Photon));
//
//	//KDPhotonArray_CPU: Photon Array, each's node's stIndex, edIndex refer to this Array
//	KDPhotonArray_CPU = (Photon*)malloc(PHOTON_NUM * sizeof(Photon));
//	memset(KDPhotonArray_CPU, 0, PHOTON_NUM * sizeof(Photon));
//
//	cudaMemcpy(photonArray, photons, PHOTON_NUM * sizeof(Photon), cudaMemcpyDeviceToHost);
//
//	//cudaMalloc(&KDNodePhoton_GPU, PHOTON_NUM * sizeof(Photon));
//
//
//	vector<Photon*> phos;
//	for (int i = 0; i < PHOTON_NUM; ++i)
//		phos.push_back(&photonArray[i]);
//
//	KDNodePhoton_CPU = Photon_KDTreeBuild(phos, 0, KDPhotonArray_CPU);
//
//	int ArraySize = pow(2, TreeHeight(KDNodePhoton_CPU)) * sizeof(Photon);
//	KDNodePhotonArrayTree_CPU = (KDNode_Photon_GPU*)malloc(ArraySize);
//
//	setKDNodeIndex(KDNodePhoton_CPU, 0);
//
//	copyKDTreeToArray(KDNodePhotonArrayTree_CPU, 0, KDNodePhoton_CPU);
//
//	cudaMalloc(&KDNodePhotonArrayTree_GPU, ArraySize);
//	cudaMemcpy(KDNodePhotonArrayTree_GPU, KDNodePhotonArrayTree_CPU, ArraySize, cudaMemcpyHostToDevice);
//
//	cudaMemcpy(KDPhotonArray_GPU, KDPhotonArray_CPU, PHOTON_NUM * sizeof(Photon), cudaMemcpyHostToDevice);
//
//	free(photonArray);
//	free(KDPhotonArray_CPU);
//	free(KDNodePhotonArrayTree_CPU);
//}

int TreeHeight(KDNode_Photon_CPU* root)
{
	if (root == NULL)return 0;
	int heightLeft = 0, heightRight = 0;
	if (root->left != NULL)heightLeft = TreeHeight(root->left);
	if (root->right != NULL)heightRight = TreeHeight(root->right);
	return heightLeft > heightRight ? heightLeft + 1 : heightRight + 1;
}

//int setKDNodeIndex(KDNode_Photon_CPU* root, int cur)
//{
//	if (root == NULL)return 0;
//	if (root->left == NULL || root->left->photons.size() == 0 || root->right == NULL || root->right->photons.size() == 0)
//	{
//		root->stIndex = cur;
//		root->edIndex = cur + root->photons.size() - 1;
//		return root->edIndex;
//	}
//	else
//	{
//		setKDNodeIndex(root->left, cur);
//		root->stIndex = cur;
//		root->edIndex = setKDNodeIndex(root->right, cur + root->left->photons.size());
//		return root->edIndex;
//	}
//}

//void copyKDTreeToArray(KDNode_CUDA* root, int cur, KDNode* KDTreeRoot_CPU)
//void copyKDTreeToArray(KDNode_Photon_GPU* node_GPU, int cur, KDNode_Photon_CPU* KDTreeRoot_CPU)
//{
//	if (KDTreeRoot_CPU == NULL)return;
//	node_GPU[cur].isValidate = 1;
//	//node_GPU[cur].stIndex = KDTreeRoot_CPU->stIndex;
//	//node_GPU[cur].edIndex = KDTreeRoot_CPU->edIndex;
//	//node_GPU[cur].photon_sz = node_GPU[cur].edIndex - node_GPU[cur].stIndex + 1;
//	node_GPU[cur].split = KDTreeRoot_CPU->split;
//	node_GPU[cur].node_pos = KDTreeRoot_CPU->node_pos;
//	node_GPU[cur].Index = cur;
//
//	node_GPU[cur].photon.phi = KDTreeRoot_CPU->photon.phi;
//	node_GPU[cur].photon.pos = KDTreeRoot_CPU->photon.pos;
//	node_GPU[cur].photon.power = KDTreeRoot_CPU->photon.power;
//	node_GPU[cur].photon.theta = KDTreeRoot_CPU->photon.theta;
//
//
//	if (KDTreeRoot_CPU->left == NULL && KDTreeRoot_CPU->right == NULL)
//	{
//		node_GPU[cur].isRoot = true;
//	}
//		
//	else
//	{
//		node_GPU[cur].isRoot = false;
//		copyKDTreeToArray(node_GPU, cur * 2 + 1, KDTreeRoot_CPU->left);
//		copyKDTreeToArray(node_GPU, cur * 2 + 2, KDTreeRoot_CPU->right);
//	}
//}

int treeSize(KDNode_Photon_CPU* KDTreeRoot_CPU)
{
	if (KDTreeRoot_CPU == NULL)return 0;
	int left = 0, right = 0;
	if (KDTreeRoot_CPU->left != NULL)left = treeSize(KDTreeRoot_CPU->left);
	if (KDTreeRoot_CPU->right != NULL)right = treeSize(KDTreeRoot_CPU->right);
	return left + right + 1;
}

int copyKDTreeToArray(KDNode_Photon_GPU* node_GPU, KDNode_Photon_CPU* KDTreeRoot_CPU, int copy_index)
{
	if (KDTreeRoot_CPU == NULL)return -1;
	node_GPU[copy_index].photon = KDTreeRoot_CPU->photon;
	node_GPU[copy_index].split = KDTreeRoot_CPU->split;
	if (KDTreeRoot_CPU->left == NULL && KDTreeRoot_CPU->right == NULL)
	{
		node_GPU[copy_index].left = node_GPU[copy_index].right = -1;
		node_GPU[copy_index].index = copy_index;
		return copy_index;
	}
	if (KDTreeRoot_CPU->left != NULL)
	{
		node_GPU[copy_index].left = copyKDTreeToArray(node_GPU, KDTreeRoot_CPU->left, copy_index+1);
		node_GPU[node_GPU[copy_index].left].parent = copy_index;
	}
	else node_GPU[copy_index].left = -1;
	if (KDTreeRoot_CPU->right != NULL)
	{
		int left_size = treeSize(KDTreeRoot_CPU->left);
		node_GPU[copy_index].right = copyKDTreeToArray(node_GPU, KDTreeRoot_CPU->right, left_size + copy_index + 1);
		node_GPU[node_GPU[copy_index].right].parent = copy_index;
	}
	else node_GPU[copy_index].right = -1;
	node_GPU[copy_index].index = copy_index;
	return copy_index;
}


KDNode_Photon_CPU* Photon_KDTreeBuild(vector<Photon*>photons, int depth)
{
	if (photons.size() == 0)return NULL;
	if (photons.size() == 1)
	{
		KDNode_Photon_CPU* node = new KDNode_Photon_CPU();
		node->photon.phi = photons[0]->phi;
		node->photon.pos = photons[0]->pos;
		node->photon.power = photons[0]->power;
		node->photon.theta = photons[0]->theta;
		node->split = -1;
		node->left = NULL;
		node->right = NULL;
		return node;
	}

	KDNode_Photon_CPU* node = new KDNode_Photon_CPU();
	int axis = Variance(photons);

	switch (axis)
	{
	case 0:
		sort(photons.begin(), photons.end(), cmp_x);
		break;
	case 1:
		sort(photons.begin(), photons.end(), cmp_y);
		break;
	case 2:
		sort(photons.begin(), photons.end(), cmp_z);
		break;
	}

	int mid = photons.size() / 2;
	float3 midPos = photons[mid]->pos;
	node->photon.phi = photons[mid]->phi;
	node->photon.pos = photons[mid]->pos;
	node->photon.power = photons[mid]->power;
	node->photon.theta = photons[mid]->theta;
	photons.erase(photons.begin() + int(photons.size() / 2));
	vector<Photon*>left_photons;
	vector<Photon*>right_photons;
	for (int i = 0; i < photons.size(); ++i)
	{
		switch (axis)
		{
		case 0:
			midPos.x <= photons[i]->pos.x ? right_photons.push_back(photons[i]) : left_photons.push_back(photons[i]);
			break;
		case 1:
			midPos.y <= photons[i]->pos.y ? right_photons.push_back(photons[i]) : left_photons.push_back(photons[i]);
			break;
		case 2:
			midPos.z <= photons[i]->pos.z ? right_photons.push_back(photons[i]) : left_photons.push_back(photons[i]);
			break;
		}
	}

	node->photon.pos = midPos;
	node->split = axis;
	node->left = Photon_KDTreeBuild(left_photons, depth + 1);
	node->right = Photon_KDTreeBuild(right_photons, depth + 1);


	return node;
}

//KDNode_Photon_CPU* Photon_KDTreeBuild(vector<Photon*>photons, int depth, Photon* KDPhotonArray_CPU)
//{
//	if (photons.size() == 0)return NULL;
//	if (photons.size() == 1)
//	{
//		KDNode_Photon_CPU* node = new KDNode_Photon_CPU();
//		node->node_pos = photons[0]->pos;
//		node->photons = photons;
//		node->split = -1;
//		node->left = NULL;
//		node->right = NULL;
//		KDPhotonArray_CPU[PI_cur].phi = photons[0]->phi;
//		KDPhotonArray_CPU[PI_cur].pos = photons[0]->pos;
//		KDPhotonArray_CPU[PI_cur].power = photons[0]->power;
//		KDPhotonArray_CPU[PI_cur].theta = photons[0]->theta;
//		PI_cur++;
//		return node;
//	}
//
//	//KDNode_Photon_CPU* node = (KDNode_Photon_CPU*)malloc(sizeof(KDNode_Photon_CPU));
//	KDNode_Photon_CPU* node = new KDNode_Photon_CPU();
//	node->photons = photons;
//	int axis = Variance(photons);
//	
//	switch (axis)
//	{
//	case 0:
//		sort(photons.begin(), photons.end(), cmp_x);
//		break;
//	case 1:
//		sort(photons.begin(), photons.end(), cmp_y);
//		break;
//	case 2:
//		sort(photons.begin(), photons.end(), cmp_z);
//		break;
//	}
//
//	float3 midPos = photons[int(photons.size() / 2)]->pos;
//	photons.erase(photons.begin() + int(photons.size() / 2));
//	vector<Photon*>left_photons;
//	vector<Photon*>right_photons;
//	for (int i = 0; i < photons.size(); ++i)
//	{
//		switch (axis)
//		{
//		case 0:
//			midPos.x <= photons[i]->pos.x ? right_photons.push_back(photons[i]) : left_photons.push_back(photons[i]);
//			break;
//		case 1:
//			midPos.y <= photons[i]->pos.y ? right_photons.push_back(photons[i]) : left_photons.push_back(photons[i]);
//			break;
//		case 2:
//			midPos.z <= photons[i]->pos.z ? right_photons.push_back(photons[i]) : left_photons.push_back(photons[i]);
//			break;
//		}
//	}
//	
//	//if ((left_photons.size() != 0 && right_photons.size() != 0) || (left_photons.size() + right_photons.size()) > 16)
//	//if ((left_photons.size() != 0 && right_photons.size() != 0))
//	//{
//	//	node->node_pos = midPos;
//	//	node->split = axis;
//	//	node->left = Photon_KDTreeBuild(left_photons, depth + 1, KDPhotonArray_CPU);
//	//	node->right = Photon_KDTreeBuild(right_photons, depth + 1, KDPhotonArray_CPU);
//	//}
//	//else
//	//{
//	//	for (int i = 0; i < photons.size(); ++i)
//	//	{
//	//		KDPhotonArray_CPU[PI_cur].phi = photons[i]->phi;
//	//		KDPhotonArray_CPU[PI_cur].pos = photons[i]->pos;
//	//		KDPhotonArray_CPU[PI_cur].power = photons[i]->power;
//	//		KDPhotonArray_CPU[PI_cur].theta = photons[i]->theta;
//	//		PI_cur++;
//	//	}
//	//	node->node_pos = midPos;
//	//	node->split = -1;
//	//	node->left = NULL;
//	//	node->right = NULL;
//	//}
//	node->node_pos = midPos;
//	node->split = axis;
//	node->left = Photon_KDTreeBuild(left_photons, depth + 1, KDPhotonArray_CPU);
//	node->right = Photon_KDTreeBuild(right_photons, depth + 1, KDPhotonArray_CPU);
//	return node;
//}