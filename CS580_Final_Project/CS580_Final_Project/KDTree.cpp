#include "KDTree.h"
#include "global.h"

using namespace std;
using namespace KDTree;

#define eps 10e-6
#define zero(x) (((x)>0?(x):-(x))<eps)
#define RAY_LENGTH 2000

/* Operators */
float3 operator+(const float3 &a, const float3 &b) {

	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}

float3 operator-(const float3 &a, const float3 &b) {

	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);

}

float3 operator*(const float3 &a, const float3 &b) {

	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);

}

float3 operator*(const float3 &a, const float &b) {

	return make_float3(a.x * b, a.y * b, a.z * b);

}

bool operator==(const float3 &a, const float3 &b) {

	return ((a.x == b.x) && (a.y == b.y) && (a.z == b.z));

}

float3 operator/(const float3 &a, const float3 &b) {

	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);

}

float3 KDTree::crossProduct(float3 a, float3 b)
{
	float3 result;
	result.x = a.y * b.z - a.z * b.y;
	result.y = a.z * b.x - a.x * b.z;
	result.z = a.x * b.y - a.y * b.x;

	return result;
}

float KDTree::dotProduct(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 KDTree::normalize(float3 vector)
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

bool KDTree::isInside(float3 point, float3* triangle)
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

	cross0 = normalize(crossProduct(pointTo0, edge0to1));
	cross1 = normalize(crossProduct(pointTo1, edge1to2));
	cross2 = normalize(crossProduct(pointTo2, edge2to0));
	value0 = dotProduct(cross0, cross1);
	value1 = dotProduct(cross1, cross2);
	value2 = dotProduct(cross2, cross0);

	if ((value0 >= -eps && value1 >= -eps && value2 >= -eps))
		return true;
	else
		return false;
}

float KDTree::hitSurface(float3* vertex, float3 pos, float3 dir, float3* pho)
{
	//step1 calculate normal
	float3 edge1, edge2, normal;
	edge1.x = vertex[1].x - vertex[0].x;
	edge1.y = vertex[1].y - vertex[0].y;
	edge1.z = vertex[1].z - vertex[0].z;

	edge2.x = vertex[2].x - vertex[1].x;
	edge2.y = vertex[2].y - vertex[1].y;
	edge2.z = vertex[2].z - vertex[1].z;

	normal = normalize(crossProduct(edge1, edge2));

	//step2 calculate the projected vector
	float3 linkEdge, projectedVector;
	linkEdge.x = vertex[0].x - pos.x;
	linkEdge.y = vertex[0].y - pos.y;
	linkEdge.z = vertex[0].z - pos.z;

	float projectedValue = -dotProduct(linkEdge, normal);
	projectedVector.x = -projectedValue * normal.x;
	projectedVector.y = -projectedValue * normal.y;
	projectedVector.z = -projectedValue * normal.z;

	//step3 calculate the intersected point
	float3 intersected;
	float projectedValueOntoLine = dotProduct(projectedVector, dir);
	if (projectedValueOntoLine <= 0)
		return MAX_DIS;

	float distance = projectedValue * projectedValue / projectedValueOntoLine;
	intersected.x = pos.x + distance * dir.x;
	intersected.y = pos.y + distance * dir.y;
	intersected.z = pos.z + distance * dir.z;

	//step4 check if intersected
	if (isInside(intersected, vertex))
	{
		pho->x = intersected.x;
		pho->y = intersected.y;
		pho->z = intersected.z;
		return distance;
	}
	else
		return MAX_DIS;
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

/* Line Tracing */

float3 BoundingBox::pevc(float3 s1, float3 s2, float3 s3){ return crossProduct((s1 - s2), (s2 - s3)); }

int BoundingBox::dots_onplane(float3 a, float3 b, float3 c, float3 d)
{
	return zero(dotProduct(pevc(a, b, c), (d - a)));
}

int BoundingBox::dots_inline(float3 p1, float3 p2, float3 p3)
{
	return length(crossProduct((p1 - p2), (p2 - p3)));
}

float3 BoundingBox::normalize(float3 vector)
{
	float3 result;
	float value = (vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
	value = sqrt(value);
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

bool BoundingBox::Check_PointInRec(float3 t_points[], float3 p)
{
	bool sign1 = false, sign2 = false, sign3 = false;
	//sign1 = (t_points[0] - p).crossProduct(t_points[1] - p).dotProduct((t_points[1] - p).crossProduct(t_points[2] - p)) >= eps;
	//sign2 = (t_points[1] - p).crossProduct(t_points[2] - p).dotProduct((t_points[2] - p).crossProduct(t_points[3] - p)) >= eps;
	//sign3 = (t_points[2] - p).crossProduct(t_points[3] - p).dotProduct((t_points[3] - p).crossProduct(t_points[0] - p)) >= eps;
	sign1 = dotProduct(crossProduct((t_points[0] - p), (t_points[1] - p)), crossProduct((t_points[1] - p), (t_points[2] - p))) >= eps;
	sign2 = dotProduct(crossProduct((t_points[1] - p), (t_points[2] - p)), crossProduct((t_points[2] - p), (t_points[3] - p))) >= eps;
	sign3 = dotProduct(crossProduct((t_points[2] - p), (t_points[3] - p)), crossProduct((t_points[3] - p), (t_points[0] - p))) >= eps;
	if ((sign1 == sign2) && (sign2 == sign3))
	{
		return true;
	}
	return false;
}

int BoundingBox::same_side(float3 p1, float3 p2, float3 l1, float3 l2)
{ 
	return dotProduct(crossProduct((l1 - l2), (p1 - l2)), crossProduct((l1 - l2), (p2 - l2))) > eps;
}

int BoundingBox::dot_online_in(float3 p, float3 l1, float3 l2)
{ 
	return zero(length(crossProduct((p - l1), (p - l2)))) && (l1.x - p.x) * (l2.x - p.x)<eps && (l1.y - p.y) * (l2.y - p.y)<eps && (l1.z - p.z)*(l2.z - p.z)<eps;
}

bool BoundingBox::intersect_in(float3 u1, float3 u2, float3 v1, float3 v2)
{
	if (!dots_onplane(u1, u2, v1, v2)) return false;
	if (!dots_inline(u1, u2, v1) || !dots_inline(u1, u2, v2)) return !same_side(u1, u2, v1, v2) && !same_side(v1, v2, u1, u2);
	return dot_online_in(u1, v1, v2) || dot_online_in(u2, v1, v2) || dot_online_in(v1, u1, u2) || dot_online_in(v2, u1, u2);
}

int BoundingBox::LineRectangleIntersect(float3 l1, float3 l2, float3 t_points[])
{
	if (Check_PointInRec(t_points, l1) || Check_PointInRec(t_points, l2))return 1;
	bool res1 = intersect_in(l1, l2, t_points[0], t_points[1]);
	bool res2 = intersect_in(l1, l2, t_points[1], t_points[2]);
	bool res3 = intersect_in(l1, l2, t_points[2], t_points[3]);
	bool res4 = intersect_in(l1, l2, t_points[3], t_points[0]);
	if (intersect_in(l1, l2, t_points[0], t_points[1]) || intersect_in(l1, l2, t_points[1], t_points[2]) || intersect_in(l1, l2, t_points[2], t_points[3]) || intersect_in(l1, l2, t_points[3], t_points[0]))return true;
	return 0;
}

float3 BoundingBox::ProjectionPoint(float3 p, float3 o, float3 n)
{
	float3 v = p - o, pp, poj;
	n = normalize(n);
	float k = (v.x * n.x + v.y * n.y + v.z * n.z);
	pp = o + n * k;
	poj = o + p - pp;
	return poj;
}

bool BoundingBox::isValidatePlane(float3 plane[3])
{
	if (plane[0] == plane[1] || plane[0] == plane[2] || plane[1] == plane[2])return false;
	return true;
}

bool BoundingBox::hit(const float3& pos, const float3& dir)
{
	float3 plane1[3], plane2[3], plane3[3];
	float3 t_points[3][4];
	float3 projectPoint[3][2];
	float3 st = pos;
	float3 ed = pos + dir * RAY_LENGTH;
	//zoy left
	plane1[0] = min;
	plane1[1] = min;
	plane1[1].z = max.z;
	plane1[2] = plane1[1];
	plane1[2].y = max.y;

	t_points[0][0] = plane1[0];
	t_points[0][1] = plane1[1];
	t_points[0][2] = plane1[2];
	t_points[0][3] = plane1[2];
	t_points[0][3].z = min.z;

	//xoy front
	plane2[0] = min;
	plane2[1] = min;
	plane2[1].x = max.x;
	plane2[2] = min;
	plane2[2].y = max.y;

	t_points[1][0] = plane2[0];
	t_points[1][1] = plane2[1];
	t_points[1][2] = plane2[1];
	t_points[1][3] = plane2[2];
	t_points[1][2].y = max.y;

	//xoz bottom
	plane3[0] = min;
	plane3[1] = plane3[0];
	plane3[1].z = max.z;
	plane3[2] = plane3[0];
	plane3[2].x = max.x;

	t_points[2][0] = plane3[0];
	t_points[2][1] = plane3[1];
	t_points[2][2] = plane3[1];
	t_points[2][2].z = max.z;
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

float3 KDTriangle::get_midpoint()
{
	float3 midpoint;
	memset(&midpoint, 0, sizeof(float3));

	midpoint.x = (bbox.min.x + bbox.max.x) / 2;
	midpoint.y = (bbox.min.y + bbox.max.y) / 2;
	midpoint.z = (bbox.min.z + bbox.max.z) / 2;

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

float KDTriangle::hit(const float3& pos, const float3& dir, float3* hitPos)
{
	//hitSurface(float3* vertex, float3 pos, float3 dir, float3* pho);
	float3 vertex[3];
	float distance;
	for (int i = 0; i < 3; ++i)
		vertex[i] = vertexBuffer[index[i]];

	return hitSurface(vertex, pos, dir, hitPos);
}

void KDNode::expand(device_vector<KDTriangle*>& tris)
{
	float3 bmin, bmax;
	bmin.x = INT_MAX;
	bmin.y = INT_MAX;
	bmin.z = INT_MAX;
	memset(&bmax, 0, sizeof(float3));
	for (int i = 0; i<tris.size(); ++i)
	{
		if (bmin.x > (*(tris[i])).bbox.min.x)bmin.x = (*(tris[i])).bbox.min.x;
		if (bmin.y > (*(tris[i])).bbox.min.y)bmin.y = (*(tris[i])).bbox.min.y;
		if (bmin.z > (*(tris[i])).bbox.min.z)bmin.z = (*(tris[i])).bbox.min.z;

		if (bmax.x < (*(tris[i])).bbox.max.x)bmax.x = (*(tris[i])).bbox.max.x;
		if (bmax.y < (*(tris[i])).bbox.max.y)bmax.y = (*(tris[i])).bbox.max.y;
		if (bmax.z < (*(tris[i])).bbox.max.z)bmax.z = (*(tris[i])).bbox.max.z;
	}
	//for (device_vector<KDTriangle*>::iterator it = tris.begin(); it != tris.end(); it++)
	//{
	//	if (bmin.x > (*it)->bbox.min.x)bmin.x = (*it)->bbox.min.x;
	//	if (bmin.y > (*it)->bbox.min.y)bmin.y = (*it)->bbox.min.y;
	//	if (bmin.z > (*it)->bbox.min.z)bmin.z = (*it)->bbox.min.z;

	//	if (bmax.x < (*it)->bbox.max.x)bmax.x = (*it)->bbox.max.x;
	//	if (bmax.y < (*it)->bbox.max.y)bmax.y = (*it)->bbox.max.y;
	//	if (bmax.z < (*it)->bbox.max.z)bmax.z = (*it)->bbox.max.z;
	//}
};

KDNode* KDNode::build(device_vector<KDTriangle*>& tris, int depth) const
{
	KDNode* node = new KDNode();
	node->triangles = tris;
	node->left = NULL;
	node->right = NULL;
	node->bbox = BoundingBox();

	if (tris.size() == 0)
		return node;
	KDTriangle* ptr = (tris[0]);
	if (tris.size() == 1)
	{
		node->bbox = (*tris[0]).bbox;
		node->left = new KDNode();
		node->right = new KDNode();
		node->left->triangles = device_vector<KDTriangle*>();
		node->right->triangles = device_vector<KDTriangle*>();
		return node;
	}

	node->bbox = (*tris[0]).bbox;

	node->expand(tris);

	float3 midpt;
	memset(&midpt, 0, sizeof(float3));

	for (int i = 0; i<tris.size(); ++i)
	{
		midpt.x += (*(tris[i])).get_midpoint().x * (1.0 / tris.size());
		midpt.y += (*(tris[i])).get_midpoint().y * (1.0 / tris.size());
		midpt.z += (*(tris[i])).get_midpoint().z * (1.0 / tris.size());
	}

	device_vector<KDTriangle*> left_tris;
	device_vector<KDTriangle*> right_tris;
	int axis = node->bbox.longest_axis();
	for (int i = 0; i < tris.size(); ++i)
	{
		switch (axis)
		{
		case 0:
			midpt.x <= (*(tris[i])).get_midpoint().x ? right_tris.push_back(tris[i]) : left_tris.push_back(tris[i]);
			break;
		case 1:
			midpt.y <= (*(tris[i])).get_midpoint().y ? right_tris.push_back(tris[i]) : left_tris.push_back(tris[i]);
			break;
		case 2:
			midpt.z <= (*(tris[i])).get_midpoint().z ? right_tris.push_back(tris[i]) : left_tris.push_back(tris[i]);
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
		node->right = build(right_tris, depth + 1);
	}
	else
	{
		node->left = new KDNode();
		node->right = new KDNode();
		node->left->triangles = device_vector<KDTriangle*>();
		node->right->triangles = device_vector<KDTriangle*>();
	}
	return node;
}

bool KDNode::hit(KDNode* node, const float3 &pos, const float3 &dir, float3* hitPos, KDTriangle* hitTriangle)
{
	if (node->bbox.hit(pos, dir))
	{
		float3 normal;
		bool hit_tri = false;
		float3 hit_pt, local_hiy_pt;

		if (node->left->triangles.size() > 0 || node->right->triangles.size() > 0)
		{
			bool hitleft = hit(node->left, pos, dir, hitPos, hitTriangle);
			bool hitright = hit(node->right, pos, dir, hitPos, hitTriangle);
			return hitleft || hitright;
		}
		else
		{
			float t = INT_MAX, tmin = INT_MAX;
			for (int i = 0; i < node->triangles.size(); ++i)
			{
				t = (*node->triangles[i]).hit(pos, dir, hitPos);
				if (t < MAX_DIS)
				{
					if (t < tmin)
					{
						tmin = t;
						(*hitTriangle) = (*node->triangles[i]);
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