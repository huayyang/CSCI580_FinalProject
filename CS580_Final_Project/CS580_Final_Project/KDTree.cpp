#include "KDTree.h"
#include "global.h"

using namespace std;
using namespace KDTree;


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

float3 KDTriangle::get_midpoint()
{
	float3 midpoint;
	memset(&midpoint, 0, sizeof(float3));

	midpoint.x = (bbox.min.x + bbox.max.x) / 2;
	midpoint.y = (bbox.min.y + bbox.max.y) / 2;
	midpoint.z = (bbox.min.z + bbox.max.z) / 2;

	return midpoint;
}

void KDTriangle::generate_bounding_box(int index)
{
	float3 bmin, bmax;
	bmin.x = bmin.y = bmin.z = INT_MAX;
	memset(&bmax, 0, sizeof(float3));

	for (int i = 0; i < 3; ++i)
	{

	}
}

void KDNode::expand(vector<KDTriangle*>& tris)
{
	float3 bmin, bmax;
	bmin.x = INT_MAX;
	bmin.y = INT_MAX;
	bmin.z = INT_MAX;
	memset(&bmax, 0, sizeof(float3));
	for (vector<KDTriangle*>::iterator it = tris.begin(); it != tris.end(); it++)
	{
		if (bmin.x > (*it)->bbox.min.x)bmin.x = (*it)->bbox.min.x;
		if (bmin.y > (*it)->bbox.min.y)bmin.y = (*it)->bbox.min.y;
		if (bmin.z > (*it)->bbox.min.z)bmin.z = (*it)->bbox.min.z;

		if (bmax.x < (*it)->bbox.max.x)bmax.x = (*it)->bbox.max.x;
		if (bmax.y < (*it)->bbox.max.y)bmax.y = (*it)->bbox.max.y;
		if (bmax.z < (*it)->bbox.max.z)bmax.z = (*it)->bbox.max.z;
	}
};

KDNode* KDNode::build(vector<KDTriangle*>& tris, int depth) const
{
	KDNode* node = new KDNode();
	node->triangles = tris;
	node->left = NULL;
	node->right = NULL;
	node->bbox = BoundingBox();

	if (tris.size() == 0)
		return node;
	 
	if (tris.size() == 1)
	{
		node->bbox = tris[0]->bbox;
		node->left = new KDNode();
		node->right = new KDNode();
		node->left->triangles = vector<KDTriangle*>();
		node->right->triangles = vector<KDTriangle*>();
		return node;
	}

	node->bbox = tris[0]->bbox;

	node->expand(tris);

	float3 midpt;
	memset(&midpt, 0, sizeof(float3));

	for (vector<KDTriangle*>::iterator it = tris.begin(); it != tris.end(); it++)
	{
		midpt.x += ((*it)->get_midpoint().x * (1.0 / tris.size()));
		midpt.y += ((*it)->get_midpoint().y * (1.0 / tris.size()));
		midpt.z += ((*it)->get_midpoint().z * (1.0 / tris.size()));
	}

	vector<KDTriangle*> left_tris;
	vector<KDTriangle*> right_tris;
	int axis = node->bbox.longest_axis();
	for (int i = 0; i < tris.size(); ++i)
	{
		switch (axis)
		{
		case 0:
			midpt.x >= tris[i]->get_midpoint().x ? right_tris.push_back(tris[i]) : left_tris.push_back(tris[i]);
			break;
		case 1:
			midpt.y >= tris[i]->get_midpoint().y ? right_tris.push_back(tris[i]) : left_tris.push_back(tris[i]);
			break;
		case 2:
			midpt.z >= tris[i]->get_midpoint().z ? right_tris.push_back(tris[i]) : left_tris.push_back(tris[i]);
			break;
		}
	}

	if (left_tris.size() == 0 && right_tris.size() > 0)left_tris = right_tris;
	if (right_tris.size() == 0 && left_tris.size() > 0)right_tris = left_tris;

	int matches = 0;
	for (int i = 0; i < left_tris.size(); ++i)
	{
		for (int j = 0; j < right_tris.size(); ++j)
		{
			if (left_tris[i] == right_tris[j])++matches;
		}
	}

	if ((float)matches / left_tris.size() < 0.5 && (float)matches / right_tris.size())
	{
		node->left = build(left_tris, depth + 1);
		node->right = build(right_tris, depth + 1);
	}
	else
	{
		node->left = new KDNode();
		node->right = new KDNode();
		node->left->triangles = vector<KDTriangle*>();
		node->right->triangles = vector<KDTriangle*>();
	}
	return node;
}