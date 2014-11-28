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

void copyKDTreeToArray(KDNode_CUDA* root, int cur, KDNode* KDTreeRoot_CPU)
{
	if (KDTreeRoot_CPU == NULL || KDTreeRoot_CPU->triangles.size() == 0)return;
	root[cur].bbox = KDTreeRoot_CPU->bbox;
	root[cur].stIndex = KDTreeRoot_CPU->stIndex;
	root[cur].edIndex = KDTreeRoot_CPU->edIndex;
	root[cur].triangle_sz = root[cur].edIndex - root[cur].stIndex + 1;
	root[cur].depth = KDTreeRoot_CPU->depth;
	if (KDTreeRoot_CPU->left == NULL && KDTreeRoot_CPU->right == NULL)
		root[cur].isRoot = true;
	else root[cur].isRoot = false;
	copyKDTreeToArray(root, cur*2 + 1, KDTreeRoot_CPU->left);
	copyKDTreeToArray(root, cur*2 + 2, KDTreeRoot_CPU->right);
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

	if (left_tris.size() != 0 && right_tris.size() != 0)
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

