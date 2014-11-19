#ifndef KDTREE_H
#define KDTREE_H

#include<vector>
#include "defines.h"

using namespace std;

namespace KDTree
{
	class BoundingBox
	{
	public:
		float3 min, max;
		BoundingBox& operator=(const BoundingBox &rhs)
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
		void generate_bounding_box(int index);
		int index;
		BoundingBox bbox;
		float3 get_midpoint();
	};

	class KDNode
	{
	public:
		BoundingBox bbox;
		KDNode* left, *right;
		vector<KDTriangle*> triangles;
		KDNode(){};
		KDNode* build(vector<KDTriangle*>& tris, int depth) const;
		void expand(vector<KDTriangle*>& tris);
	};
};
#endif