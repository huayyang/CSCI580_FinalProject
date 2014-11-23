#include "KDTree.cuh"
#include "global.h"

using namespace std;
using namespace KDTree;

#define eps 10e-6
#define zero(x) (((x)>0?(x):-(x))<eps)
#define RAY_LENGTH 2000

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

/* operators */
__device__ float3 operator+(const float3 &a, const float3 &b) {

	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}

__device__ float3 operator-(const float3 &a, const float3 &b) {

	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);

}

__device__ float3 operator*(const float3 &a, const float3 &b) {

	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);

}

__device__ float3 operator*(const float3 &a, const float &b) {

	return make_float3(a.x * b, a.y * b, a.z * b);

}

__device__ bool operator==(const float3 &a, const float3 &b) {

	return ((a.x == b.x) && (a.y == b.y) && (a.z == b.z));

}

__device__ float3 operator/(const float3 &a, const float3 &b) {

	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);

}

/* functions */
__device__ float3 CrossProduct(float3 a, float3 b)
{
	float3 result;
	result.x = a.y * b.z - a.z * b.y;
	result.y = a.z * b.x - a.x * b.z;
	result.z = a.x * b.y - a.y * b.x;

	return result;
}

__device__ float DotProduct(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 Normalize(float3 vector)
{
	float3 result;
	float value = (vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
	value = sqrtf(value);
	if (value < 0.001 && value > -0.001)
	{
		result.x = 0;
		result.y = 0;
		result.z = 0;
	}
	else
	{
		result.x = vector.x / value;
		result.y = vector.y / value;
		result.z = vector.z / value;
	}

	return result;
}

__device__ bool IsInside(float3 point, float3* triangle)
{
	float3 pointTo0, pointTo1, pointTo2;
	float3 edge0to1, edge1to2, edge2to0;
	float3 cross0, cross1, cross2;
	float value0, value1, value2;

	pointTo0.x = triangle[0].x - point.x;
	pointTo0.y = triangle[0].y - point.y;
	pointTo0.z = triangle[0].z - point.z;

	pointTo1.x = triangle[1].x - point.x;
	pointTo1.y = triangle[1].y - point.y;
	pointTo1.z = triangle[1].z - point.z;

	pointTo2.x = triangle[2].x - point.x;
	pointTo2.y = triangle[2].y - point.y;
	pointTo2.z = triangle[2].z - point.z;

	edge0to1.x = triangle[1].x - triangle[0].x;
	edge0to1.y = triangle[1].y - triangle[0].y;
	edge0to1.z = triangle[1].z - triangle[0].z;

	edge1to2.x = triangle[2].x - triangle[1].x;
	edge1to2.y = triangle[2].y - triangle[1].y;
	edge1to2.z = triangle[2].z - triangle[1].z;

	edge2to0.x = triangle[0].x - triangle[2].x;
	edge2to0.y = triangle[0].y - triangle[2].y;
	edge2to0.z = triangle[0].z - triangle[2].z;

	cross0 = Normalize(CrossProduct(pointTo0, edge0to1));
	cross1 = Normalize(CrossProduct(pointTo1, edge1to2));
	cross2 = Normalize(CrossProduct(pointTo2, edge2to0));
	value0 = DotProduct(cross0, cross1);
	value1 = DotProduct(cross1, cross2);
	value2 = DotProduct(cross2, cross0);

	if ((value0 >= -eps && value1 >= -eps && value2 >= -eps))
		return true;
	else
		return false;
}

// 
__device__ float HitSurface(float3* vertex, float3 pos, float3 dir, float3* pho)
{
	//step1 calculate normal
	float3 edge1, edge2, normal;
	edge1.x = vertex[1].x - vertex[0].x;
	edge1.y = vertex[1].y - vertex[0].y;
	edge1.z = vertex[1].z - vertex[0].z;

	edge2.x = vertex[2].x - vertex[1].x;
	edge2.y = vertex[2].y - vertex[1].y;
	edge2.z = vertex[2].z - vertex[1].z;

	normal = Normalize(CrossProduct(edge1, edge2));

	//step2 calculate the projected vector
	float3 linkEdge, projectedVector;
	linkEdge.x = vertex[0].x - pos.x;
	linkEdge.y = vertex[0].y - pos.y;
	linkEdge.z = vertex[0].z - pos.z;

	float projectedValue = -DotProduct(linkEdge, normal);
	projectedVector.x = -projectedValue * normal.x;
	projectedVector.y = -projectedValue * normal.y;
	projectedVector.z = -projectedValue * normal.z;

	//step3 calculate the intersected point
	float3 intersected;
	float projectedValueOntoLine = DotProduct(projectedVector, dir);
	if (projectedValueOntoLine <= 0)
		return MAX_DIS;

	float distance = projectedValue * projectedValue / projectedValueOntoLine;
	intersected.x = pos.x + distance * dir.x;
	intersected.y = pos.y + distance * dir.y;
	intersected.z = pos.z + distance * dir.z;

	//step4 check if intersected
	if (IsInside(intersected, vertex))
	{
		pho->x = intersected.x;
		pho->y = intersected.y;
		pho->z = intersected.z;
		return distance;
	}
	else
		return MAX_DIS;
}

/* Line Tracing */
__device__ float3 KDTree::pevc(float3 s1, float3 s2, float3 s3){ return CrossProduct((s1 - s2), (s2 - s3)); }

__device__ int KDTree::dots_onplane(float3 a, float3 b, float3 c, float3 d)
{
	return zero(DotProduct(pevc(a, b, c), (d - a)));
}

__device__ int KDTree::dots_inline(float3 p1, float3 p2, float3 p3)
{
	return length(CrossProduct((p1 - p2), (p2 - p3)));
}

__device__ float KDTree::length(float3 a)
{ 
	return sqrt(a.x*a.x + a.y*a.y + a.z*a.z); 
}

__device__ bool KDTree::Check_PointInRec(float3 t_points[], float3 p)
{
	bool sign1 = false, sign2 = false, sign3 = false;
	//sign1 = (t_points[0] - p).CrossProduct(t_points[1] - p).DotProduct((t_points[1] - p).CrossProduct(t_points[2] - p)) >= eps;
	//sign2 = (t_points[1] - p).CrossProduct(t_points[2] - p).DotProduct((t_points[2] - p).CrossProduct(t_points[3] - p)) >= eps;
	//sign3 = (t_points[2] - p).CrossProduct(t_points[3] - p).DotProduct((t_points[3] - p).CrossProduct(t_points[0] - p)) >= eps;
	sign1 = DotProduct(CrossProduct((t_points[0] - p), (t_points[1] - p)), CrossProduct((t_points[1] - p), (t_points[2] - p))) >= eps;
	sign2 = DotProduct(CrossProduct((t_points[1] - p), (t_points[2] - p)), CrossProduct((t_points[2] - p), (t_points[3] - p))) >= eps;
	sign3 = DotProduct(CrossProduct((t_points[2] - p), (t_points[3] - p)), CrossProduct((t_points[3] - p), (t_points[0] - p))) >= eps;
	if ((sign1 == sign2) && (sign2 == sign3))
	{
		return true;
	}
	return false;
}

__device__ int KDTree::same_side(float3 p1, float3 p2, float3 l1, float3 l2)
{
	return DotProduct(CrossProduct((l1 - l2), (p1 - l2)), CrossProduct((l1 - l2), (p2 - l2))) > eps;
}

__device__ int KDTree::dot_online_in(float3 p, float3 l1, float3 l2)
{
	return zero(length(CrossProduct((p - l1), (p - l2)))) && (l1.x - p.x) * (l2.x - p.x)<eps && (l1.y - p.y) * (l2.y - p.y)<eps && (l1.z - p.z)*(l2.z - p.z)<eps;
}

__device__ bool KDTree::intersect_in(float3 u1, float3 u2, float3 v1, float3 v2)
{
	if (!dots_onplane(u1, u2, v1, v2)) return false;
	if (!dots_inline(u1, u2, v1) || !dots_inline(u1, u2, v2)) return !same_side(u1, u2, v1, v2) && !same_side(v1, v2, u1, u2);
	return dot_online_in(u1, v1, v2) || dot_online_in(u2, v1, v2) || dot_online_in(v1, u1, u2) || dot_online_in(v2, u1, u2);
}

__device__ int KDTree::LineRectangleIntersect(float3 l1, float3 l2, float3 t_points[])
{
	if (Check_PointInRec(t_points, l1) || Check_PointInRec(t_points, l2))return 1;
	bool res1 = intersect_in(l1, l2, t_points[0], t_points[1]);
	bool res2 = intersect_in(l1, l2, t_points[1], t_points[2]);
	bool res3 = intersect_in(l1, l2, t_points[2], t_points[3]);
	bool res4 = intersect_in(l1, l2, t_points[3], t_points[0]);
	if (intersect_in(l1, l2, t_points[0], t_points[1]) || intersect_in(l1, l2, t_points[1], t_points[2]) || intersect_in(l1, l2, t_points[2], t_points[3]) || intersect_in(l1, l2, t_points[3], t_points[0]))return true;
	return 0;
}

__device__ float3 KDTree::ProjectionPoint(float3 p, float3 o, float3 n)
{
	float3 v = p - o, pp, poj;
	n = Normalize(n);
	float k = (v.x * n.x + v.y * n.y + v.z * n.z);
	pp = o + n * k;
	poj = o + p - pp;
	return poj;
}

__device__ bool KDTree::isValidatePlane(float3 plane[3])
{
	if (plane[0] == plane[1] || plane[0] == plane[2] || plane[1] == plane[2])return false;
	return true;
}

__device__ bool KDTree::BoundingBoxHit(BoundingBox bbox, const float3& pos, const float3& dir)
{
	float3 plane1[3], plane2[3], plane3[3];
	float3 t_points[3][4];
	float3 projectPoint[3][2];
	float3 st = pos;
	float3 ed = pos + dir * RAY_LENGTH;
	//zoy left
	plane1[0] = bbox.min;
	plane1[1] = bbox.min;
	plane1[1].z = bbox.max.z;
	plane1[2] = plane1[1];
	plane1[2].y = bbox.max.y;

	t_points[0][0] = plane1[0];
	t_points[0][1] = plane1[1];
	t_points[0][2] = plane1[2];
	t_points[0][3] = plane1[2];
	t_points[0][3].z = bbox.min.z;

	//xoy front
	plane2[0] = bbox.min;
	plane2[1] = bbox.min;
	plane2[1].x = bbox.max.x;
	plane2[2] = bbox.min;
	plane2[2].y = bbox.max.y;

	t_points[1][0] = plane2[0];
	t_points[1][1] = plane2[1];
	t_points[1][2] = plane2[1];
	t_points[1][3] = plane2[2];
	t_points[1][2].y = bbox.max.y;

	//xoz bottom
	plane3[0] = bbox.min;
	plane3[1] = plane3[0];
	plane3[1].z = bbox.max.z;
	plane3[2] = plane3[0];
	plane3[2].x = bbox.max.x;

	t_points[2][0] = plane3[0];
	t_points[2][1] = plane3[1];
	t_points[2][2] = plane3[1];
	t_points[2][2].z = bbox.max.z;
	t_points[2][3] = plane3[2];

	projectPoint[0][0] = ProjectionPoint(st, plane1[0], pevc(plane1[0], plane1[1], plane1[2]));
	projectPoint[0][1] = ProjectionPoint(ed, plane1[0], pevc(plane1[0], plane1[1], plane1[2]));

	projectPoint[1][0] = ProjectionPoint(st, plane2[0], pevc(plane2[0], plane2[1], plane2[2]));
	projectPoint[1][1] = ProjectionPoint(ed, plane2[0], pevc(plane2[0], plane2[1], plane2[2]));

	//xoz top
	projectPoint[2][0] = ProjectionPoint(st, plane3[0], pevc(plane3[0], plane3[1], plane3[2]));
	projectPoint[2][1] = ProjectionPoint(ed, plane3[0], pevc(plane3[0], plane3[1], plane3[2]));

	bool res = true;

	if (isValidatePlane(plane1))
	{
		res = res & LineRectangleIntersect(projectPoint[0][0], projectPoint[0][1], t_points[0]);
	}


	if (isValidatePlane(plane2))
	{
		res = res & LineRectangleIntersect(projectPoint[1][0], projectPoint[1][1], t_points[1]);
	}

	if (isValidatePlane(plane3))
	{
		res = res & LineRectangleIntersect(projectPoint[2][0], projectPoint[2][1], t_points[2]);
	}


	return res;
}

__device__ float3 KDTree::get_midpoint(KDTriangle triangle)
{
	float3 midpoint;
	memset(&midpoint, 0, sizeof(float3));

	midpoint.x = (triangle.bbox.min.x + triangle.bbox.max.x) / 2;
	midpoint.y = (triangle.bbox.min.y + triangle.bbox.max.y) / 2;
	midpoint.z = (triangle.bbox.min.z + triangle.bbox.max.z) / 2;

	return midpoint;
}

float3 KDTree::host_get_midpoint(KDTriangle triangle)
{
	float3 midpoint;
	memset(&midpoint, 0, sizeof(float3));

	midpoint.x = (triangle.bbox.min.x + triangle.bbox.max.x) / 2;
	midpoint.y = (triangle.bbox.min.y + triangle.bbox.max.y) / 2;
	midpoint.z = (triangle.bbox.min.z + triangle.bbox.max.z) / 2;

	return midpoint;
}

void KDTriangle::generate_bounding_box()
{
	float3 bmin, bmax;
	bmin.x = bmin.y = bmin.z = INT_MAX;
	memset(&bmax, 0, sizeof(float3));

	for (int i = 0; i < 3; ++i)
	{
		if (vertexBuffer[index[i]].x < bmin.x) bmin.x = vertexBuffer[index[i]].x;
		if (vertexBuffer[index[i]].x > bmax.x) bmax.x = vertexBuffer[index[i]].x;

		if (vertexBuffer[index[i]].y < bmin.y) bmin.y = vertexBuffer[index[i]].y;
		if (vertexBuffer[index[i]].y > bmax.y) bmax.y = vertexBuffer[index[i]].y;

		if (vertexBuffer[index[i]].z < bmin.z) bmin.z = vertexBuffer[index[i]].z;
		if (vertexBuffer[index[i]].z > bmax.z) bmax.z = vertexBuffer[index[i]].z;
	}

	bbox.max = bmax;
	bbox.min = bmin;
}

__device__ float KDTree::TriangleHit(KDTriangle triangle, float3* vertex, const float3& pos, const float3& dir, float3* hitPos)
{
	//hitSurface(float3* vertex, float3 pos, float3 dir, float3* pho);
	float3 points[3];
	float distance;
	for (int i = 0; i < 3; ++i)
		points[i] = vertex[triangle.index[i]];

	return HitSurface(points, pos, dir, hitPos);
}

KDNode* KDNode::build(vector<KDTriangle*>& tris, int depth)
{
	KDNode* node = new KDNode();
	node->triangles = tris;
	node->left = NULL;
	node->right = NULL;
	node->bbox = BoundingBox();

	if (tris.size() == 0)
	{
		return node;
	}
		
	KDTriangle* ptr = (tris[0]);
	if (tris.size() == 1)
	{
		node->bbox = (*tris[0]).bbox;
		node->left = new KDNode();
		node->right = new KDNode();
		node->left->triangles = vector<KDTriangle*>();
		node->right->triangles = vector<KDTriangle*>();
		return node;
	}

	node->bbox = (*tris[0]).bbox;

	float3 bmin, bmax;
	bmin.x = INT_MAX;
	bmin.y = INT_MAX;
	bmin.z = INT_MAX;
	memset(&bmax, 0, sizeof(float3));
	for (int i = 0; i<tris.size(); ++i)
	{
		if (bmin.x >(*(tris[i])).bbox.min.x)bmin.x = (*(tris[i])).bbox.min.x;
		if (bmin.y > (*(tris[i])).bbox.min.y)bmin.y = (*(tris[i])).bbox.min.y;
		if (bmin.z > (*(tris[i])).bbox.min.z)bmin.z = (*(tris[i])).bbox.min.z;

		if (bmax.x < (*(tris[i])).bbox.max.x)bmax.x = (*(tris[i])).bbox.max.x;
		if (bmax.y < (*(tris[i])).bbox.max.y)bmax.y = (*(tris[i])).bbox.max.y;
		if (bmax.z < (*(tris[i])).bbox.max.z)bmax.z = (*(tris[i])).bbox.max.z;
	}

	float3 midpt;
	memset(&midpt, 0, sizeof(float3));

	for (int i = 0; i<tris.size(); ++i)
	{
		float3 midPoint = host_get_midpoint((*(tris[i])));
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
			midpt.x <= host_get_midpoint((*(tris[i]))).x ? right_tris.push_back(tris[i]) : left_tris.push_back(tris[i]);
			break;
		case 1:
			midpt.y <= host_get_midpoint((*(tris[i]))).y ? right_tris.push_back(tris[i]) : left_tris.push_back(tris[i]);
			break;
		case 2:
			midpt.z <= host_get_midpoint((*(tris[i]))).z ? right_tris.push_back(tris[i]) : left_tris.push_back(tris[i]);
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

	if (matches == 0 || ((float)matches / left_tris.size() < 0.5 && (float)matches / right_tris.size() < 0.5))
	{
		node->left = build(left_tris, depth + 1);
		node->left->setSize();
		node->right = build(right_tris, depth + 1);
		node->right->setSize();
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

void KDNode::setSize()
{
	size = triangles.size();
	if (size > 0)
	{
		d_tris = *(triangles.data());
	}
	else
	{
		d_tris = NULL;
	}
}

__device__ bool KDTree::KDTreeHit(KDNode* node, float3* vertex, const float3 &pos, const float3 &dir, float3* hitPos, KDTriangle* hitTriangle)
{
	if (BoundingBoxHit(node->bbox, pos, dir))
	{
		bool hit_tri = false;

		if (node->left->size > 0 || node->right->size > 0)
		{
			bool hitleft = KDTreeHit(node->left, vertex, pos, dir, hitPos, hitTriangle);
			bool hitright = KDTreeHit(node->right, vertex, pos, dir, hitPos, hitTriangle);
			return hitleft || hitright;
		}
		else
		{
			float t = INT_MAX, tmin = INT_MAX;
			for (int i = 0; i < node->size; ++i)
			{
				t = TriangleHit((node->d_tris[i]), vertex, pos, dir, hitPos);
				if (t < MAX_DIS)
				{
					if (t < tmin)
					{
						tmin = t;
						for (int j = 0; j < 3;++j)
						(*hitTriangle).index[j] = (node->d_tris[i]).index[j];
						hit_tri = true;
					}
				}
			}
			if (hit_tri)
			{
				return true;
			}
		}
	}
	return false;
}