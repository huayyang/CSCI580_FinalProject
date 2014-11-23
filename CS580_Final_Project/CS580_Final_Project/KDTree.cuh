#ifndef KDTREE_H
#define KDTREE_H

#include <vector>
#include "defines.h"

using namespace std;

namespace KDTree
{
	class BoundingBox
	{
	public:
		float3 min, max;
		BoundingBox(){ min.x = min.y = min.z = INT_MAX; memset(&max, 0, sizeof(float3)); };
		BoundingBox& operator=(const BoundingBox &rhs)
		{
			this->min = rhs.min;
			this->max = rhs.max;
			return *this;
		}
		int longest_axis();
	};

	/*Line Tracing*/
	__device__ float3 pevc(float3 s1, float3 s2, float3 s3);
	__device__ int dots_onplane(float3 a, float3 b, float3 c, float3 d);
	__device__ int dots_inline(float3 p1, float3 p2, float3 p3);
	__device__ float length(float3 a);
	__device__ bool Check_PointInRec(float3 t_points[], float3 p);
	__device__ int dot_online_in(float3 p, float3 l1, float3 l2);
	__device__ int same_side(float3 p1, float3 p2, float3 l1, float3 l2);
	__device__ bool intersect_in(float3 u1, float3 u2, float3 v1, float3 v2);
	__device__ int LineRectangleIntersect(float3 l1, float3 l2, float3 t_points[]);
	__device__ float3 ProjectionPoint(float3 p, float3 o, float3 n);
	__device__ bool isValidatePlane(float3 plane[3]);
	__device__ bool BoundingBoxHit(BoundingBox bbox, const float3& pos, const float3& dir);
	

	class KDTriangle
	{
	public:
		void generate_bounding_box();
		int index[3];
		BoundingBox bbox;
	};

	__device__ float3 get_midpoint(KDTriangle triangle);
	__device__ float TriangleHit(KDTriangle triangle, float3* vertex, const float3& pos, const float3& dir, float3* hitPos);
	float3 host_get_midpoint(KDTriangle triangle);

	class KDNode
	{
	public:
		BoundingBox bbox;
		KDNode* left, *right;
		int size;
		KDTriangle * d_tris;
		vector<KDTriangle*> triangles;
		KDNode(){};
		KDNode* build(vector<KDTriangle*>& tris, int depth);
		void setSize();
	};
	
	__device__ bool KDTreeHit(KDNode* node, float3* vertex, const float3 &pos, const float3 &dir, float3* hitPos, KDTriangle* hitTriangle);
};
#endif