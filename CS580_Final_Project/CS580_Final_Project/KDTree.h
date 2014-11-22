#ifndef KDTREE_H
#define KDTREE_H

#include<vector>
#include "defines.h"

using namespace std;

namespace KDTree
{
	float dotProduct(float3 a, float3 b);
	float3 crossProduct(float3 a, float3 b);
	bool isInside(float3 point, float3* triangle);
	float3 normalize(float3 vector);
	float hitSurface(float3* vertex, float3 pos, float3 dir, float3* pho);

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

		/*Line Tracing*/
		//test operator+(const int & a)

		float3 pevc(float3 s1, float3 s2, float3 s3);
		int dots_onplane(float3 a, float3 b, float3 c, float3 d);
		int dots_inline(float3 p1, float3 p2, float3 p3);
		float length(float3 a){ return sqrt(a.x*a.x + a.y*a.y + a.z*a.z); };
		bool Check_PointInRec(float3 t_points[], float3 p);
		int dot_online_in(float3 p, float3 l1, float3 l2);
		int same_side(float3 p1, float3 p2, float3 l1, float3 l2);
		bool intersect_in(float3 u1, float3 u2, float3 v1, float3 v2);
		int LineRectangleIntersect(float3 l1, float3 l2, float3 t_points[]);
		bool isValidatePlane(float3 plane[3]);
		bool hit(const float3& pos, const float3& dir);
		float3 normalize(float3 vector);
		float3 ProjectionPoint(float3 p, float3 o, float3 n);
	};
	
	class KDTriangle
	{
	public:
		void generate_bounding_box();
		int index[3];
		BoundingBox bbox;
		float3 get_midpoint();
		float hit(const float3& pos, const float3& dir, float3* hitPos);
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
		bool hit(KDNode* node, const float3 &pos, const float3 &dir, float3* hitPos, KDTriangle* hitTriangle);
	};
};
#endif