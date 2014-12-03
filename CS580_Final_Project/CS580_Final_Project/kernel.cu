#include "rayTracingProcessor.cuh"
#include "KDTree.h"
#include "defines.h"
#include "math_functions.h"
#include "global.h"
#include <curand.h>
#include <iostream>

#define eps 10e-8

__device__ unsigned int x = 123456789,
y = 362436000,
z = 521288629,
c = 7654321; /* Seed variables */

/* operators */
__device__ float3 operator+(const float3 &a, const float3 &b) {

	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}

__device__ float3 operator-(const float3 &a, const float3 &b) {
	float tmpx = a.x - b.x;
	float tmpy = a.y - b.y;
	float tmpz = a.z - b.z;
	if (abs(tmpx) < eps)tmpx = 0;
	if (abs(tmpy) < eps)tmpy = 0;
	if (abs(tmpz) < eps)tmpz = 0;
	return make_float3(tmpx, tmpy, tmpz);

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

__device__ unsigned int KISS()
{
	unsigned long long t, A = 698769069ULL;

	x = 69069 * x + 12345;

	y ^= (y << 13); y ^= (y >> 17); y ^= (y << 5);

	t = (A*z + c);
	c = (t >> 32);
	z = t;
	//return cuRAND();
	return x + y + z;
}

__device__ double IntegerNoise (int n)

{

 n = (n >> 13) ^ n;

 int nn = (n * (n * n * 60493 + 19990303) + 1376312589) & 0x7fffffff;

 return 1 - ((double)nn / 1073741824.0);

}


__device__ float3 crossProduct(float3 a, float3 b)
{
	float3 result;
	result.x = a.y * b.z - a.z * b.y;
	result.y = a.z * b.x - a.x * b.z;
	result.z = a.x * b.y - a.y * b.x;

	return result;
}

__device__ float dotProduct(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 normalize(float3 vector)
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

__device__ bool isInside(float3 point, float3* triangle)
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

	if ((value0 >= -0.001 && value1 >= -0.001 && value2 >= -0.001))
		return true;
	else
		return false;
}

__device__ float checkDis(float3* vertex, float3 pos, float3 dir)
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
		return distance;
	else
		return MAX_DIS;
}

// 
__device__ float hitSurface(float3* vertex, float3 pos, float3 dir, float3* pho, bool* isFront)
{
	*isFront = true;
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

	float projectedValue = dotProduct(linkEdge, normal);
	if (projectedValue > 0.001)
		*isFront = false;
	projectedVector.x = projectedValue * normal.x;
	projectedVector.y = projectedValue * normal.y;
	projectedVector.z = projectedValue * normal.z;

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

__device__ bool getR(float3* outdir, float ni, float3 I, float3 N)
{
	if(dotProduct(I,N)> -0.001)
	{
		outdir->x = I.x;
		outdir->y = I.y;
		outdir->z = I.z;

		return true;
	}

	float3 planeNormal = crossProduct(N,I);
	float3 tempN = N;
	float xxx = dotProduct(I, N);
	tempN.x = I.x - xxx * N.x;
	tempN.y = I.y - xxx * N.y;
	tempN.z = I.z - xxx * N.z;
	float3 biNormal = normalize(tempN);
	float sinI = dotProduct(biNormal,normalize(I));
	float sinR = sinI / ni;
	if (sinR >= 0.9999)
		return false;
	float cosR = sqrtf(1 - sinR *sinR);

	outdir->x = biNormal.x * sinR - N.x * cosR;
	outdir->y = biNormal.y * sinR - N.y * cosR;
	outdir->z = biNormal.z * sinR - N.z * cosR;

	//float tmp = dotProduct(N,crossProduct(I,*outdir));
	//if (tmp < -0.001 || tmp > 0.001)
	//	printf("%f ..%f... %f,%f,%f......%f,%f,%f......%f,%f,%f\n",ni,tmp,N.x,N.y,N.z,I.x,I.y,I.z,outdir->x,outdir->y,outdir->z);
	return true;
}

__device__ float3 getI(float ni,float3 R,float3 N)
{
	float3 planeNormal = crossProduct(R,N);
	float3 biNormal = crossProduct(planeNormal,N);
	float sinR = -dotProduct(biNormal,R);
	float sinI = sinR * ni;
	float cosI = sqrtf(1 - sinI *sinI);

	float3 I;
	I.x = -biNormal.x * sinI + N.x * cosI;
	I.y = -biNormal.y * sinI + N.y * cosI;
	I.z = -biNormal.z * sinI + N.z * cosI;
}

__device__ void swapValue(float &a, float &b){
	float temp = a;
	a = b;
	b = temp;
}

__device__ void splitSort(float *A, int n, int low, int high)
{
	if (low >= high)
		return;
	int left = low;
	int right = high;
	bool moveRight = true;
	while (left != right){
		if (moveRight){
			if (A[left] > A[right])
			{
				swapValue(A[left], A[right]);
				moveRight = false;
			}
			else
			{
				right--;
			}
		}
		else{
			if (A[left] > A[right])
			{
				swapValue(A[left], A[right]);
				moveRight = true;
			}
			else
			{
				left++;
			}
		}
	}
	splitSort(A, n, low, left - 1);
	splitSort(A, n, left + 1, high);
}

//yating add lerp
__device__ float getIntrValue(float3 *curVertex, float3 hitpoint)
{
	float res;
	float3 edge01,edge02;
	edge01.x = curVertex[1].x-curVertex[0].x;
	edge01.y = curVertex[1].y-curVertex[0].y;  
	edge01.z = curVertex[1].z-curVertex[0].z; 

	edge02.x = curVertex[2].x-curVertex[0].x;
	edge02.y = curVertex[2].y-curVertex[0].y;  
	edge02.z = curVertex[2].z-curVertex[0].z; 

	float3 cross = crossProduct(edge02,edge01);

	//fumula  ax + by + cz = d
	float d = (cross.x*curVertex[1].x  + cross.y*curVertex[1].y + cross.z*curVertex[1].z );

	//calculate z value  z =( d-ax -by )/c
	if(cross.z >10e-6 || cross.z <-10e-6){
		res = ( d - (cross.x* hitpoint.x) - (cross.y*hitpoint.y)) / cross.z; 
		 
	}
	else return NULL;
	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

//
//	printf("%d %f %f %f \n",faceIndex,curVertex[2].x,curVertex[2].y,curVertex[2].z);
	return res;
}
__device__ float3  lerp(int faceIndex, float3 *curVertex,float3 *curFnormal,float3 hitpoint)
{
		//IsInside(hitpoint, curVertex);
 //printf("%d %f %f %f \n",faceIndex,curFnormal[2].x,curFnormal[2].y,curFnormal[2].z);
		float3 res;

		float3 test[3];

		test[0].x = curVertex[0].x;		test[0].y = curVertex[0].y; 		test[0].z = curFnormal[0].x;
		test[1].x = curVertex[1].x;		test[1].y = curVertex[1].y; 		test[1].z = curFnormal[1].x;
		test[2].x = curVertex[2].x;		test[2].y = curVertex[2].y; 		test[2].z = curFnormal[2].x;
		res.x =getIntrValue(test,hitpoint);
		if(res.x ==NULL)
			res.x = (curFnormal[0].x+ curFnormal[1].x+ curFnormal[2].x)/3;

		test[0].x = curVertex[0].x;		test[0].y = curVertex[0].y; 		test[0].z = curFnormal[0].y;
		test[1].x = curVertex[1].x;		test[1].y = curVertex[1].y; 		test[1].z = curFnormal[1].y;
		test[2].x = curVertex[2].x;		test[2].y = curVertex[2].y; 		test[2].z = curFnormal[2].y;
		res.y =getIntrValue(test,hitpoint);
				if(res.y ==NULL)
			res.y = (curFnormal[0].y +curFnormal[1].y+curFnormal[2].y)/3;
		test[0].x = curVertex[0].x;		test[0].y = curVertex[0].y; 		test[0].z = curFnormal[0].z;
		test[1].x = curVertex[1].x;		test[1].y = curVertex[1].y; 		test[1].z = curFnormal[1].z;
		test[2].x = curVertex[2].x;		test[2].y = curVertex[2].y; 		test[2].z = curFnormal[2].z;
		res.z =getIntrValue(test,hitpoint);
		if(res.z ==NULL)
			res.z = (curFnormal[0].z+curFnormal[1].z+curFnormal[2].z)/3;
		normalize(res);
		/*
		for(int i=0; i<3;i++)
			curVertex[i].z = curFnormal[i].x;
		res.x =getIntrValue(curVertex,hitpoint);
		for(int i=0; i<3;i++)
			curVertex[i].z = curFnormal[i].y;
		res.y =getIntrValue(curVertex,hitpoint);
		for(int i=0; i<3;i++)
			curVertex[i].z = curFnormal[i].z;
		res.z =getIntrValue(curVertex,hitpoint);*/
			return res;
}

__device__ uchar4 getColor(int depth, int currentIndex, uchar4 * pixels, int count, Object* objects, Material* materials, float3 pos, float3 dir, Photon* photons, KDNode_CUDA * KDTree_GPU, int* TriangleIndexArray_GPU, KDNode_Photon_GPU* KDNodePhotonArrayTree_GPU, int* pqtop)
{
	uchar4 resultColor;

	resultColor.x = 0;
	resultColor.y = 0;
	resultColor.z = 0;

	if (depth > 10)
	{
		//printf("aaa!\n");
		return make_uchar4(0,0,0,255);
	}

	float minDis = MAX_DIS;
	int index = -1;
	float3 hitpoint;
	bool isFront = true;
	KDTriangle hitTriangle;
	float dis = MAX_DIS;
	bool kdIsFront = true;

	int KDIndex = -1;
	float3 kdHit;

	hitTriangle.index = -1;

	KDTreeHit(0, objects, pos, dir, &kdHit, &hitTriangle, &dis, KDTree_GPU, TriangleIndexArray_GPU, &isFront, currentIndex);
	//KDTreeHit(0, objects, pos, dir, &hitpoint, &hitTriangle, &minDis, KDTree_GPU, TriangleIndexArray_GPU, &isFront, currentIndex);
	//printf("Hit Pos: (%f,%f,%f) Hit Dis: (%f,%f,%f)\n", pos.x, pos.y, pos.z, dir.x, dir.y, dir.z);
	KDIndex = hitTriangle.index;
	if(KDIndex == 8 || KDIndex == 9)
	{
		float3 tempDir = kdHit - make_float3(50,30,100);
		float tempDistance = dotProduct(tempDir, tempDir);
		resultColor.x += 400 - tempDistance > 255 ? 255 : 400 -tempDistance;
		resultColor.y += 400 - tempDistance > 255 ? 255 : 400 -tempDistance;
		resultColor.z += 400 - tempDistance > 255 ? 255 : 400 -tempDistance;
	}
	index = KDIndex;
	hitpoint = kdHit;
	minDis = dis;
	//isFront = kdIsFront;
	isFront = true;

	if (index != -1)
	{
		Material hitMat = materials[(int)(objects[index].materialIndex.x)];
		float Kd = hitMat.Kd;
		float Ks = hitMat.Ks;
		float Kni = hitMat.Kni;
		if (Kd > eps && isFront)
		{
			//int radius = PHOTON_RADIUS;
			////float kdDistance[PHOTON_RADIUS] = { 0 };
			////float distances[PHOTON_NUM] = { 0 };
			//int searched[PHOTON_NUM] = { -1 };
			//Photon pDistance[PHOTON_NUM] = { 0 };
			//float3 near;
			//float nearestDis = -1;
			//
			//int index_i = blockIdx.x;
			//int index_j = threadIdx.x;
			//pqtop[index_i*UNIT_Y + index_j] = 0;
			//bool found = false;
			//KDTreeKNNSearch(KDNodePhotonArrayTree_GPU, 0, hitpoint, &near, nearestDis, pDistance, radius, searched, index_i, index_j, pqtop);
			////printf("Hit Point: (%f,%f,%f)\n", hitpoint.x, hitpoint.y, hitpoint.z);
			////nearestDis = KDTreeKNNSearch(KDNodePhotonArrayTree_GPU, 0, hitpoint, distance, radius, &near);
			////nearestDis = KDTreeKNNSearch(KDNodePhotonArrayTree_GPU, 0, KDPhotonArray_GPU, hitpoint, distance, 0, radius, &near);
			////distance[radius-1] = near;
			//float fardis = 0;
			//float fardis2 = 0;
			//for (int k = 0; k < radius; k++)
			//{
			//	float tempdis = (GetDistanceSquare(pDistance[k].pos, hitpoint));
			//	if (fardis < tempdis)
			//	{
			//		fardis = tempdis;
			//	}
			//	fardis2 = tempdis + fardis2*k;
			//	fardis2 /= (k+1);

			//}
			//bool notEq = false;
			//float bf_min1 = INT_MAX;
			//float bf_min2 = INT_MAX;
			//float bf_min3 = INT_MAX;
			//for (int k = 0; k < PHOTON_NUM; k++)
			//{
			//	float3 temp;
			//	temp.x = hitpoint.x - photons[k].pos.x;
			//	temp.y = hitpoint.y - photons[k].pos.y;
			//	temp.z = hitpoint.z - photons[k].pos.z;
			//	float dis = dotProduct(temp, temp);
			//	if (dis < 0.1)
			//	{
			//		found = true;
			//		break;
			//	}
			//}
			//	/*if (abs(fardis - dis) < eps)
			//	{
			//		notEq = true;
			//	}
			//	if (dis < bf_min1)
			//	{
			//		bf_min3 = bf_min2;
			//		bf_min2 = bf_min1;
			//		bf_min1 = dis;
			//	}
			//	else if (dis < bf_min2 && abs(bf_min1 - bf_min2) > eps)
			//	{
			//		bf_min3 = bf_min2;
			//		bf_min2 = dis;
			//	}
			//	else if (dis < bf_min3 && abs(bf_min1 - bf_min3) > eps && abs(bf_min2 - bf_min3) > eps)
			//		bf_min3 = dis;*/
			//}
			//if (!notEq)printf("Not In!\n");
			//
			//if (abs(bf_min3 - fardis) > eps)
			//{
			//	printf("Not Equal! %.2f, %.2f\n", bf_min3, fardis2);

			//	printf("Pos (%f,%f,%f)\n", hitpoint.x, hitpoint.y, hitpoint.z);
			//	printf("\n");
			//	//printf("KD Result: (%f)\n", minDis);
			//	printf("\n");
			//	for (int i = 0; i < radius; ++i)
			//	{
			//		printf("KD Result: (%f)\n", kdDistance[i]);
			//	}
			//	printf("\n");
			//	for (int i = 0; i < PHOTON_NUM; ++i)
			//	{
			//		printf("BF Result: (%f)\n", distances[i]);
			//	}
			//}
			//

			float nearestDis = INT_MAX;
			float distances[PHOTON_RADIUS] = { 0 };
			for (int k = 0; k < PHOTON_RADIUS; ++k)
			{
				float3 temp;
				temp.x = hitpoint.x - photons[k].pos.x;
				temp.y = hitpoint.y - photons[k].pos.y;
				temp.z = hitpoint.z - photons[k].pos.z;
				distances[k] = dotProduct(temp, temp);
			}
				
			for (int k = PHOTON_RADIUS; k<PHOTON_NUM; k++)
			{
				float3 temp;
				temp.x = hitpoint.x - photons[k].pos.x;
				temp.y = hitpoint.y - photons[k].pos.y;
				temp.z = hitpoint.z - photons[k].pos.z;
				float dis = dotProduct(temp, temp);
				float maxInArray = 0;
				int maxIndex = -1;
				nearestDis = nearestDis < dis ? nearestDis : dis;
				for (int i = 0; i < PHOTON_RADIUS; ++i)
				{
					if (maxInArray < distances[i])
					{
						maxInArray = distances[i];
						maxIndex = i;
					}
				}
				if (maxInArray > dis)
				{
					distances[maxIndex] = dis;
				}
			}
			float avg = 0;
			for (int k = 0; k < PHOTON_RADIUS; ++k)
			{
				avg += distances[k];
			}
			avg /= PHOTON_RADIUS;


			if (!PHOTON_SHOW || nearestDis > 0.2)
			{
				int3 colorInt;
				colorInt.x = resultColor.x + Kd * objects[index].color[0].x / avg * PHOTON_FORCE ;
				colorInt.y = resultColor.y + Kd * objects[index].color[1].y / avg * PHOTON_FORCE ;
				colorInt.z = resultColor.z + Kd * objects[index].color[2].z / avg * PHOTON_FORCE ;
				resultColor.x = colorInt.x > 255 ? 255 : colorInt.x; 
				resultColor.y = colorInt.y > 255 ? 255 : colorInt.y;
				resultColor.z = colorInt.z > 255 ? 255 : colorInt.z;
			}
			else
			{
				resultColor.x = 244;
				resultColor.y = 0;
				resultColor.z = 0;
			}

		}

		float tF = 0;

		if (Kni > 0.001)
		{
			float3 edge1, edge2, realN;
			edge1.x = objects[index].vertex[1].x - objects[index].vertex[0].x;
			edge1.y = objects[index].vertex[1].y - objects[index].vertex[0].y;
			edge1.z = objects[index].vertex[1].z - objects[index].vertex[0].z;

			edge2.x = objects[index].vertex[2].x - objects[index].vertex[0].x;
			edge2.y = objects[index].vertex[2].y - objects[index].vertex[0].y;
			edge2.z = objects[index].vertex[2].z - objects[index].vertex[0].z;

			realN = normalize(crossProduct(edge1, edge2));


			float Ni = hitMat.Ni;

			float3 curVex[3] = { objects[index].vertex[0], objects[index].vertex[1], objects[index].vertex[2] };
			float3 curFNormal[3] = { objects[index].normal[0], objects[index].normal[1], objects[index].normal[2] };
			float3 lerpNormal = lerp(index,curVex,curFNormal, hitpoint);
			lerpNormal = normalize(lerpNormal);

			float3 outDir;
			float3 n;
			if (isFront)
			{
				n.x = lerpNormal.x;
				n.y = lerpNormal.y;
				n.z = lerpNormal.z;
			}
			else
			{
				n.x = -lerpNormal.x;
				n.y = -lerpNormal.y;
				n.z = -lerpNormal.z;
				Ni = 1/Ni;
			}

			if (!getR(&outDir,Ni,dir,n))
			{
				tF = Kni;
			}
			else
			{

				uchar4 refractColor = getColor(depth + 1, index, pixels, count, objects, materials, hitpoint, outDir, photons, KDTree_GPU, TriangleIndexArray_GPU, KDNodePhotonArrayTree_GPU, pqtop);
			
				int3 colorInt;
				colorInt.x = resultColor.x + Kni * refractColor.x;
				colorInt.y = resultColor.y + Kni * refractColor.y;
				colorInt.z = resultColor.z + Kni * refractColor.z;
				resultColor.x = colorInt.x > 255 ? 255 : colorInt.x;
				resultColor.y = colorInt.y > 255 ? 255 : colorInt.y;
				resultColor.z = colorInt.z > 255 ? 255 : colorInt.z;
				//{
				//	float maxcolor = colorInt.x;
				//	maxcolor = colorInt.y > maxcolor ? colorInt.y : maxcolor;
				//	maxcolor = colorInt.z > maxcolor ? colorInt.z : maxcolor;
				//	if(maxcolor > 255)
				//	{
				//		float3 resultcolor = normalize(make_float3(colorInt.x, colorInt.y, colorInt.z));
				//		resultColor.x = 255*resultcolor.x;
				//		resultColor.y = 255*resultcolor.y;
				//		resultColor.z = 255*resultcolor.z;
				//	}
				//}
			}

		}

		if ( (Ks + tF) > 0.001)
		{
			//yating create lerp
			float3 curVex[3] = { objects[index].vertex[0], objects[index].vertex[1], objects[index].vertex[2] };
			float3 curFNormal[3] = { objects[index].normal[0], objects[index].normal[1], objects[index].normal[2] };
			float3 lerpNormal = lerp(index,curVex,curFNormal, hitpoint);
			lerpNormal = normalize(lerpNormal);

			float NdotDir = -dotProduct(lerpNormal, dir);


			if(NdotDir < 0 && isFront)
			{
				resultColor = getColor(depth+1, index, pixels, count, objects, materials, hitpoint, dir, photons,
					KDTree_GPU, TriangleIndexArray_GPU, KDNodePhotonArrayTree_GPU, pqtop);
				//resultColor.x = 0;
				//resultColor.y = 0;
				//resultColor.z = 255;

			}
			else
			{
				if(NdotDir < 0)
				{
					NdotDir = -NdotDir;
					lerpNormal.x  = -lerpNormal.x ;
					lerpNormal.y  = -lerpNormal.y ;
					lerpNormal.z  = -lerpNormal.z ;
				}
				float3 reflectDir;
				reflectDir.x =lerpNormal.x * 2 * NdotDir + dir.x;
				reflectDir.y =lerpNormal.y * 2 * NdotDir + dir.y;
				reflectDir.z =lerpNormal.z * 2 * NdotDir + dir.z;

				uchar4 speculateColor = getColor(depth + 1, index, pixels, count, objects, materials, hitpoint, reflectDir, photons, KDTree_GPU, TriangleIndexArray_GPU, KDNodePhotonArrayTree_GPU, pqtop);
			
				int3 colorInt;
				colorInt.x = resultColor.x + (Ks + tF) * speculateColor.x;
				colorInt.y = resultColor.y + (Ks + tF) * speculateColor.y;
				colorInt.z = resultColor.z + (Ks + tF) * speculateColor.z;
				resultColor.x = colorInt.x > 255 ? 255 : colorInt.x;
				resultColor.y = colorInt.y > 255 ? 255 : colorInt.y;
				resultColor.z = colorInt.z > 255 ? 255 : colorInt.z;
			}

		}

	}

	return resultColor;
}

__global__ void kernel(int indexX, int indexY, int unitX, int unitY, uchar4 * pixels, int count, Object* objects, Material* materials, unsigned int width, unsigned int height, Camera* cam, Photon* photons, KDNode_CUDA * KDTree_GPU, int* TriangleIndexArray_GPU, KDNode_Photon_GPU* KDNodePhotonArrayTree_GPU, int *pqtop)
{
	int i = blockIdx.x + indexX * unitX;
	int j = threadIdx.x + indexY * unitY;
	int offsetX = i - width / 2;
	int offsetY = height / 2 - j;
	float3 dir;
	dir.x = cam->lookat.x + (cam->tan_fov_2 * 2 * offsetY / height) * cam->up.x + (cam->tan_fov_2 * 2 * offsetX / height) * cam->right.x;
	dir.y = cam->lookat.y + (cam->tan_fov_2 * 2 * offsetY / height) * cam->up.y + (cam->tan_fov_2 * 2 * offsetX / height) * cam->right.y;
	dir.z = cam->lookat.z + (cam->tan_fov_2 * 2 * offsetY / height) * cam->up.z + (cam->tan_fov_2 * 2 * offsetX / height) * cam->right.z;

	dir = normalize(dir);

	int id = i + j * width;

	pixels[id] = getColor(0, -1, pixels, count, objects, materials, cam->pos, dir, photons, KDTree_GPU, TriangleIndexArray_GPU, KDNodePhotonArrayTree_GPU, pqtop);
	pixels[id].x = pixels[id].x > 225 ? 255 : pixels[id].x+30;
	pixels[id].y = pixels[id].y > 225 ? 255 : pixels[id].y+30;
	pixels[id].z = pixels[id].z > 225 ? 255 : pixels[id].z+30;
	pixels[id].w = pixels[id].w > 225 ? 255 : pixels[id].w+30;

}

__global__ void CastPhoton(uchar4 * pixels, int count, Object* objects, Photon* photons, float3 lightPos, KDNode_CUDA * KDTree_GPU, int* TriangleIndexArray_GPU, Material* materials)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	float3 dir;

	if (i >= PHOTON_SQR || j >= PHOTON_SQR)
		return;

	dir.x = photons[i * PHOTON_SQR + j].pos.x;
	dir.y = photons[i * PHOTON_SQR + j].pos.y;
	dir.z = photons[i * PHOTON_SQR + j].pos.z;
	//dir.z = -1 * PHOTON_ANGLE;
	photons[i * PHOTON_SQR + j].pos.x = photons[i * PHOTON_SQR + j].pos.y = photons[i * PHOTON_SQR + j].pos.z = -100;
	dir = normalize(dir);

	float minDis = MAX_DIS;
	int index = -1;
	int currentIndex = -1;
	bool isFront = true;
	float3 hitPos;
	float3 hitStart = lightPos;
	KDTriangle hitTriangle; 
	int loopcount = 0;

PHOTON_CAST:

	hitTriangle.index = -1;
	loopcount++;
	if(loopcount > 30)
	{
		printf("too many!\n");
		photons[i*PHOTON_SQR + j].pos = hitStart;
		photons[i * PHOTON_SQR + j].power.x = 0;
		photons[i * PHOTON_SQR + j].power.y = 0;
		photons[i * PHOTON_SQR + j].power.z = 0;
		goto END_CAST;
	}

	KDTreeHit(0, objects, hitStart, dir, &hitPos, &hitTriangle, &minDis, KDTree_GPU, TriangleIndexArray_GPU, &isFront, currentIndex);

	index = hitTriangle.index;

	if (index != -1)
	{

		Material hitMat = materials[(int)(objects[index].materialIndex.x)];
		float Kd = hitMat.Kd;
		float Ks = hitMat.Ks;
		float Kni = hitMat.Kni;
		if( Kd > 0.5)
		{
			if( IntegerNoise(i*j + dir.x* 50 + dir.y * 30 + dir.z * 10) > PHOTON_DIFFUSE_RATE)
			{
				float3 newDir = make_float3(IntegerNoise(i*j + dir.x* 10 + dir.y * 30 + dir.z * 50), IntegerNoise(i*j + dir.x* 50 + dir.y * 30 + dir.z * 10), IntegerNoise(i*j + dir.x* 200 + dir.y * 30 + dir.z * 10));
				if(dotProduct(newDir, objects[index].normal[0]) > 0)
				{
					dir = normalize(newDir);
					hitStart = hitPos;
					currentIndex = index;
					goto PHOTON_CAST;
				}
				else
				{
					dir = normalize(newDir);
					dir.x =-dir.x; dir.y =-dir.y; dir.z =-dir.z;
					hitStart = hitPos;
					currentIndex = index;
					goto PHOTON_CAST;
				}
			}
			else
			{
				photons[i*PHOTON_SQR + j].pos = hitPos;
				photons[i * PHOTON_SQR + j].power.x = 255;
				photons[i * PHOTON_SQR + j].power.y = 255;
				photons[i * PHOTON_SQR + j].power.z = 255;
			}
		}
		if (Kni > 0.5)
		{
			float3 edge1, edge2, realN;
			edge1.x = objects[index].vertex[1].x - objects[index].vertex[0].x;
			edge1.y = objects[index].vertex[1].y - objects[index].vertex[0].y;
			edge1.z = objects[index].vertex[1].z - objects[index].vertex[0].z;

			edge2.x = objects[index].vertex[2].x - objects[index].vertex[0].x;
			edge2.y = objects[index].vertex[2].y - objects[index].vertex[0].y;
			edge2.z = objects[index].vertex[2].z - objects[index].vertex[0].z;

			realN = normalize(crossProduct(edge1, edge2));


			float Ni = hitMat.Ni;

			float3 curVex[3] = { objects[index].vertex[0], objects[index].vertex[1], objects[index].vertex[2] };
			float3 curFNormal[3] = { objects[index].normal[0], objects[index].normal[1], objects[index].normal[2] };
			float3 lerpNormal = lerp(index,curVex,curFNormal, hitPos);
			lerpNormal = normalize(lerpNormal);

			float3 outDir;
			float3 n;
			if (isFront)
			{
				n.x = lerpNormal.x;
				n.y = lerpNormal.y;
				n.z = lerpNormal.z;
			}
			else
			{
				n.x = -lerpNormal.x;
				n.y = -lerpNormal.y;
				n.z = -lerpNormal.z;
				Ni = 1/Ni;
			}

			if (!getR(&outDir,Ni,dir,n))
			{
				Ks = 1.0;
			}
			else
			{
				dir = normalize(outDir);
				hitStart = hitPos;
				goto PHOTON_CAST;
			}

		}
		if (Ks > 0.5)
		{
			float3 curVex[3] = { objects[index].vertex[0], objects[index].vertex[1], objects[index].vertex[2] };
			float3 curFNormal[3] = { objects[index].normal[0], objects[index].normal[1], objects[index].normal[2] };
			float3 lerpNormal = lerp(index,curVex,curFNormal, hitPos);
			lerpNormal = normalize(lerpNormal);

			float NdotDir = -dotProduct(lerpNormal, dir);
			float3 reflectDir;
			reflectDir.x =lerpNormal.x * 2 * NdotDir + dir.x;
			reflectDir.y =lerpNormal.y * 2 * NdotDir + dir.y;
			reflectDir.z =lerpNormal.z * 2 * NdotDir + dir.z;

			if(i == 14 && j == 8 && NdotDir < 0)
			{
				printf("wrong dir %d, %d(%d, %d), (%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f) = %.2f  ???\n", 
				loopcount, index ,i,j, dir.x, dir.y, dir.z, lerpNormal.x, lerpNormal.y, lerpNormal.z, NdotDir);
				hitStart = hitPos;
				currentIndex = index;
				goto PHOTON_CAST;
			}
			else
			{
				dir = reflectDir;
				hitStart = hitPos;
				currentIndex = index;
				goto PHOTON_CAST;
			}
		}

	}
	else
	{
		photons[i*PHOTON_SQR + j].pos = hitStart;
		photons[i * PHOTON_SQR + j].power.x = 0;
		photons[i * PHOTON_SQR + j].power.y = 0;
		photons[i * PHOTON_SQR + j].power.z = 0;
	}
END_CAST:

}

__device__ void PrintTree(KDNode_Photon_GPU* KDNodePhotonArrayTree_GPU, int cur, Photon* photons)
{
	
	//printf("Node: (%d,%d)\n", KDNodePhotonArrayTree_GPU[cur].stIndex, KDNodePhotonArrayTree_GPU[cur].edIndex);
	//printf("Node: (%f,%f,%f) split: %d\n", KDNodePhotonArrayTree_GPU[cur].photon.pos.x, KDNodePhotonArrayTree_GPU[cur].photon.pos.y, KDNodePhotonArrayTree_GPU[cur].photon.pos.z, KDNodePhotonArrayTree_GPU[cur].split);
	//printf("Node %d, Left %d, Right %d\n", KDNodePhotonArrayTree_GPU[cur].index, KDNodePhotonArrayTree_GPU[cur].left, KDNodePhotonArrayTree_GPU[cur].right);
	printf("Node %d Cood: (%f,%f,%f)\n", KDNodePhotonArrayTree_GPU[cur].index, KDNodePhotonArrayTree_GPU[cur].photon.pos.x, KDNodePhotonArrayTree_GPU[cur].photon.pos.y, KDNodePhotonArrayTree_GPU[cur].photon.pos.z);
	printf("Node %d, Parent: %d, Left: %d, Right: %d\n", KDNodePhotonArrayTree_GPU[cur].index, KDNodePhotonArrayTree_GPU[cur].parent, KDNodePhotonArrayTree_GPU[cur].left, KDNodePhotonArrayTree_GPU[cur].right);
	if (KDNodePhotonArrayTree_GPU[cur].left != -1)
	{
		PrintTree(KDNodePhotonArrayTree_GPU, KDNodePhotonArrayTree_GPU[cur].left, photons);
	}
	if (KDNodePhotonArrayTree_GPU[cur].right != -1)
	{
		PrintTree(KDNodePhotonArrayTree_GPU, KDNodePhotonArrayTree_GPU[cur].right, photons);
	}
}

__global__ void test(KDNode_Photon_GPU* KDNodePhotonArrayTree_GPU, Photon* photons, int* pqtop)
{
	//printf("Print Tree:%d\n", KDNodePhotonArrayTree_GPU[0].left);
	//PrintTree(KDNodePhotonArrayTree_GPU,0, photons);
	////printf("\n");

	//float3 target = make_float3(100,55.248020,-0.000001);
	////float3 target = make_float3(100, 55.248020, 10);
	//int radius = PHOTON_RADIUS;
	//float kdDistance[PHOTON_NUM] = { 0 };
	//float distances[PHOTON_NUM] = { 0 };
	//int searched[PHOTON_NUM] = { 0 };
	//Photon pDistance[PHOTON_NUM] = { 0 };
	//float3 near;
	//float nearestDis = -1;
	//pqtop = 0;
	
	float3 hitpoint = make_float3(100,44.060032,100);
	int radius = PHOTON_RADIUS;
	//float kdDistance[PHOTON_RADIUS] = { 0 };
	//float distances[PHOTON_NUM] = { 0 };
	int searched[PHOTON_NUM] = { -1 };
	Photon pDistance[PHOTON_NUM] = { 0 };
	float3 near;
	float nearestDis = -1;
	
	//printf("%d, %d, %d \n", i, j, pqtop[0]);
	pqtop[0] = 0;
	printf("KNN\n");
	//printf("%d, %d \n", i, j);
	bool found = false;
	KDTreeKNNSearch(KDNodePhotonArrayTree_GPU, 0, hitpoint, &near, nearestDis, pDistance, radius, searched, 0, 0, pqtop);
	//
	//printf("Target: (%f,%f,%f)\n", target.x, target.y, target.z);
	//printf("Nearest: (%f,%f,%f)\n", near.x, near.y, near.z);
	//printf("Nearest Distance: %f\n", nearestDis);

	for (int i = 0; i < radius; ++i)
	{
		//printf("Nearest Points: (%f,%f,%f)\n", pDistance[i].pos.x, pDistance[i].pos.y, pDistance[i].pos.z);
		printf("Distance: %f\n", GetDistanceSquare(pDistance[i].pos, hitpoint));
	}
		

	//nearestDis = INT_MAX;
	//for (int i = 0; i < PHOTON_NUM; ++i)
	//{
	//	float dis = GetDistanceSquare(photons[i].pos, target);
	//	if (nearestDis > dis)
	//	{
	//		nearestDis = dis;
	//		near = photons[i].pos;
	//	}
	//}
	//printf("Brute Force Nearest Distance: %f\n", nearestDis);
	//printf("Brute Force Nearest: (%f,%f,%f)\n", near.x, near.y, near.z);
	//printf("%f\n", KDPhotonArray_GPU[0].pos.x);
	//printf("test: %d\n", KDNodePhotonArrayTree_GPU[0].stIndex);
}


__global__ void KDTree_Show(KDNode_CUDA * KDTree_GPU, int* TriangleIndexArray_GPU, Object* objects, int currentIndex)
{
	//displayKDTree(0, KDTree_GPU);
	//float3 pos = make_float3(55, 65, 70);
	float3 pos = make_float3(63.034805, 65, 26.537350);
	//float3 dir = make_float3(30 - pos.x, 50 - pos.y, 30 - pos.z);
	float3 dir = make_float3(0.295529, 0.793530, -0.531952);
	dir = normalize(dir);
	float3 hitPos;
	KDTriangle hitTriangle;

	float dis = INT_MAX;
	bool isFront;
	KDTreeHit(0, objects, pos, dir, &hitPos, &hitTriangle, &dis, KDTree_GPU, TriangleIndexArray_GPU, &isFront, currentIndex);
	//printf("Compare times:%d\n", visit_node);
	//printf("Check triangle:%d\n", visit_triangle);
	//objects[hitTriangle.index].color[0] = make_uchar4(0, 0, 0, 0);
	printf("Hit Position: %f,%f,%f\n", hitPos.x, hitPos.y, hitPos.z);
	printf("Hit Triangle: %d\n\n", hitTriangle.index);

	float minDis = INT_MAX;
	int index;
	float3 hitpoint;

	for (int k = 0; k<22; k++)
	{
		if (k == currentIndex)
			continue;

		float3 hitPos;
		bool isCurrentFront = true;
		float distance = hitSurface(objects[k].vertex, pos, dir, &hitPos, &isCurrentFront);
		if (distance < minDis && distance > 0.001)
		{
			isFront = isCurrentFront;
			minDis = distance;
			index = k;
			hitpoint.x = hitPos.x; hitpoint.y = hitPos.y; hitpoint.z = hitPos.z;
		}
	}

	printf("Hit Position: %f,%f,%f\n", hitpoint.x, hitpoint.y, hitpoint.z);
	printf("Hit Triangle: %d\n\n", index);
}

__global__ void KDTree_Test(KDNode_CUDA * KDTree_GPU, int* TriangleIndexArray_GPU, Object* objects, int currentIndex)
{
	//displayKDTree(0, KDTree_GPU);

	float3 pos = make_float3(20, 30, 90);
	//float3 dir = make_float3(30 - pos.x, 50 - pos.y, 30 - pos.z);
	float3 dir = make_float3(0, 0, -1);
	dir = normalize(dir);
	float3 hitPos;
	KDTriangle hitTriangle;
	hitTriangle.index = -1;

	float dis = INT_MAX;
	bool isFront;
	KDTreeHit(0, objects, pos, dir, &hitPos, &hitTriangle, &dis, KDTree_GPU, TriangleIndexArray_GPU, &isFront, currentIndex);
	//printf("Compare times:%d\n", visit_node);
	//printf("Check triangle:%d\n", visit_triangle);
	//objects[hitTriangle.index].color[0] = make_uchar4(0, 0, 0, 0);
	printf("Hit Position: %f,%f,%f\n", hitPos.x, hitPos.y, hitPos.z);
	printf("Hit Triangle: %d\n\n", hitTriangle.index);

	float minDis = MAX_DIS;
	int index;
	float3 hitpoint;
	for (int k = 0; k<22; k++)
	{
		float3 hitPos;
		bool isCurrentFront = true;
		float distance = hitSurface(objects[k].vertex, pos, dir, &hitPos, &isCurrentFront);
		if (distance < minDis && distance > 0.001)
		{
			isFront = isCurrentFront;
			minDis = distance;
			index = k;
			hitpoint.x = hitPos.x; hitpoint.y = hitPos.y; hitpoint.z = hitPos.z;
		}
	}
	printf("Hit Position: %f,%f,%f\n", hitpoint.x, hitpoint.y, hitpoint.z);
	printf("Hit Triangle: %d\n\n", index);
}


// Helper function for using CUDA to add vectors in parallel.
//void rayTracingCuda(uchar4 * pixels, int count, Object* objects, Photon* photons, Material* materials, KDNode_CUDA * KDTree_GPU, int* TriangleIndexArray_GPU, KDNode_Photon_GPU* KDNodePhotonArrayTree_GPU, Photon* KDPhotonArray_GPU, KDNode_Photon_GPU* KDNodePhotonArrayTree_CPU)
void rayTracingCuda(uchar4 * pixels, int count, Object* objects, Photon* photons, Material* materials, KDNode_CUDA * KDTree_GPU, int* TriangleIndexArray_GPU)
{
	cudaMalloc(&pqtop, UNIT_X * UNIT_Y * sizeof(int));
	dim3 photonBlock(PHOTON_SQR);
	dim3 photonThread(PHOTON_SQR);
	// compute light photons
	CastPhoton << <photonBlock, photonThread >> >(pixels, count, objects, photons, LIGHT_POS, KDTree_GPU, TriangleIndexArray_GPU, materials);
	cudaThreadSynchronize();  
	
	//Photon* photonBuffer = (Photon*)malloc(100 * sizeof(Photon));
	//cudaMemcpy(photonBuffer,photons,100 * sizeof(Photon),cudaMemcpyDeviceToHost);
	//
	//for(int i =0;i< 100;i++)
	//{
	//	std::cout<<" "<<photonBuffer[i].pos.x<<" "<<photonBuffer[i].pos.y<<" "<<photonBuffer[i].pos.z<<"\t";
	//}

	//Photon_KDTree_Init(photons, KDNodePhotonArrayTree_GPU, KDPhotonArray_GPU);


	/* KDTree For Photon */
	//
	//photonArray = (Photon*)malloc(PHOTON_NUM * sizeof(Photon));
	//memset(photonArray, 0, PHOTON_NUM * sizeof(Photon));
	//
	//cudaMemcpy(photonArray, photons, PHOTON_NUM * sizeof(Photon), cudaMemcpyDeviceToHost);
	//
	//vector<Photon*> phos;
	//for (int i = 0; i < PHOTON_NUM; ++i)
	//	phos.push_back(&photonArray[i]);
	//
	//KDNodePhoton_CPU = Photon_KDTreeBuild(phos, 0);
	//
	//KDNodePhotonArrayTree_CPU = (KDNode_Photon_GPU*)malloc(PHOTON_NUM * sizeof(KDNode_Photon_GPU));
	//memset(KDNodePhotonArrayTree_CPU, 0, PHOTON_NUM * sizeof(KDNode_Photon_GPU));
	//
	////setKDNodeIndex(KDNodePhoton_CPU, 0);
	//copyKDTreeToArray(KDNodePhotonArrayTree_CPU, KDNodePhoton_CPU, 0);
	//KDNodePhotonArrayTree_CPU[0].parent = -1;
	////copyKDTreeToArray(KDNodePhotonArrayTree_CPU, 0, KDNodePhoton_CPU);
	//
	//cudaMalloc(&KDNodePhotonArrayTree_GPU, PHOTON_NUM * sizeof(KDNode_Photon_GPU));
	//cudaMemcpy(KDNodePhotonArrayTree_GPU, KDNodePhotonArrayTree_CPU, PHOTON_NUM * sizeof(KDNode_Photon_GPU), cudaMemcpyHostToDevice);
	//
	////printf("CPU: %d\n", KDNodePhotonArrayTree_CPU[0].stIndex);
	//
	//free(photonArray);
	//free(KDNodePhotonArrayTree_CPU);

	
	//printf("%d\n",KDNodePhotonArrayTree_CPU[0].stIndex);

	//dim3 testBlock(1);
	//dim3 testThread(1);

	//test << <testBlock, testThread >> >(KDNodePhotonArrayTree_GPU, photons, pqtop);
	//cudaThreadSynchronize();

	//dim3 kdBlock(1);
	//dim3 kdThread(1);
	//KDTree_Test << <kdBlock, kdThread >> >(KDTree_GPU, TriangleIndexArray_GPU, objects, 11);
	//cudaThreadSynchronize();

	Camera* cam = (Camera*)malloc(sizeof(Camera));
	cam->pos = CAM_POS;
	cam->lookat = CAM_LOOKAT;
	cam->up = CAM_LOOKUP;
	cam->right = CAM_LOOKRIGHT;
	cam->fov = CAM_FOV;
	cam->tan_fov_2 = tan(cam->fov * PI /2 / 180);
	cudaMalloc((void**)&mainCamera_CUDA,sizeof(Camera));
	cudaMemcpy(mainCamera_CUDA,cam,sizeof(Camera),cudaMemcpyHostToDevice);

	int width = SCR_WIDTH;
	int indexX = 0;
	while (width != 0)
	{
		int x;
		int height = SCR_HEIGHT;
		int indexY = 0;

		if (width > UNIT_X)
			x = UNIT_X;
		else
			x = width;

		while (height != 0)
		{
			int y;
			if (height > UNIT_Y)
				y = UNIT_Y;
			else
				y = height;
			
			dim3 dimblock(x);
			dim3 dimthread(y);

			// Launch a kernel on the GPU with one thread for each element.
			
			kernel << <dimblock, dimthread >> >(indexX, indexY, UNIT_X, UNIT_Y, pixels, count, objects, materials, SCR_WIDTH, SCR_HEIGHT, mainCamera_CUDA, photons, KDTree_GPU, TriangleIndexArray_GPU, KDNodePhotonArrayTree_GPU, pqtop);

			cudaThreadSynchronize();

			height -= y;
			indexY++;
		}
		width -= x;
		indexX++;
	}
}

__device__ bool ClipLine(int d, const BoundingBox& aabbBox, const float3& v0, const float3& v1, float& f_low, float& f_high)
{
	float f_dim_low, f_dim_high;

	switch (d)
	{
	case 0:
		f_dim_low = (aabbBox.min.x - v0.x) / (v1.x - v0.x);
		f_dim_high = (aabbBox.max.x - v0.x) / (v1.x - v0.x);
		break;
	case 1:
		f_dim_low = (aabbBox.min.y - v0.y) / (v1.y - v0.y);
		f_dim_high = (aabbBox.max.y - v0.y) / (v1.y - v0.y);
		break;
	case 2:
		f_dim_low = (aabbBox.min.z - v0.z) / (v1.z - v0.z);
		f_dim_high = (aabbBox.max.z - v0.z) / (v1.z - v0.z);
		break;
	}

	if (f_dim_high < f_dim_low)
	{
		float tmp = f_dim_high;
		f_dim_high = f_dim_low;
		f_dim_low = tmp;
	}

	if (f_dim_high < f_low)
		return false;

	if (f_dim_low > f_high)
		return false;

	f_low = max(f_dim_low, f_low);
	f_high = min(f_dim_high, f_high);

	if (f_low > f_high)
		return false;

	return true;
}

__device__ bool LineAABBIntersection(const BoundingBox& aabbBox, const float3& v0, const float3& dir, float3& vecIntersection, float& flFraction)
{
	float f_low = 0;
	float f_high = 1;
	float3 v1 = v0 + dir * 2000;

	if (!ClipLine(0, aabbBox, v0, v1, f_low, f_high))
		return false;

	if (!ClipLine(1, aabbBox, v0, v1, f_low, f_high))
		return false;

	if (!ClipLine(2, aabbBox, v0, v1, f_low, f_high))
		return false;

	float3 b = v1 - v0;
	vecIntersection = v0 + b * f_low;

	flFraction = f_low;

	return true;
}

__device__ bool KDTreeHit(int cur_node, Object* objects, float3 pos, float3 dir, float3* hitPos, KDTriangle* hitTriangle, float* tmin, KDNode_CUDA * KDTree_GPU, int* TriangleIndexArray_GPU, bool* isFront, int currentIndex)
{
	float3 interSect;
	float fra;
	if (LineAABBIntersection(KDTree_GPU[cur_node].bbox, pos, dir, interSect, fra))
	{
		//if (cur_node == 1211)
		//{
		//	printf("1211 %d\n",KDTree_GPU[cur_node].triangle_sz);
		//	printf("Box Hit:%d Left: %d Right: %d\n", cur_node, KDTree_GPU[cur_node].left, KDTree_GPU[cur_node].right);
		//}
		bool hit_tri = false;
		if (KDTree_GPU[cur_node].left != -1 || KDTree_GPU[cur_node].right != -1)
		{
			//if (cur_node == 1211)printf("Box Hit:%d Left: %d Right: %d\n",cur_node, KDTree_GPU[cur_node].left, KDTree_GPU[cur_node].right);
			bool hitleft = false, hitright = false;
			if (KDTree_GPU[cur_node].left != -1)hitleft = KDTreeHit(KDTree_GPU[cur_node].left, objects, pos, dir, hitPos, hitTriangle, tmin, KDTree_GPU, TriangleIndexArray_GPU, isFront, currentIndex);
			if (KDTree_GPU[cur_node].right != -1)bool hitright = KDTreeHit(KDTree_GPU[cur_node].right, objects, pos, dir, hitPos, hitTriangle, tmin, KDTree_GPU, TriangleIndexArray_GPU, isFront, currentIndex);
			return hitleft || hitright;
		}
		else
		{
			float t = MAX_DIS;
			float3 hit;
			for (int i = 0; i < KDTree_GPU[cur_node].triangle_sz; ++i)
			{
				int index = TriangleIndexArray_GPU[KDTree_GPU[cur_node].stIndex + i];
				t = hitSurface(objects[index].vertex, pos, dir, &hit, isFront);
				if (t != MAX_DIS && t > 0.001)
				{
					if (currentIndex != TriangleIndexArray_GPU[KDTree_GPU[cur_node].stIndex + i] && t < *tmin)
					{
						*tmin = t;
						(*hitTriangle).index = TriangleIndexArray_GPU[KDTree_GPU[cur_node].stIndex + i];
						*hitPos = hit;
						hit_tri = true;
					}
				}
			}
			if (hit_tri)
			{
				return true;
			}
			return false;
		}
	}
	return false;
}

__device__ float GetDistanceSquare(float3 pointA, float3 pointB)
{
	return (pointA.x - pointB.x) * (pointA.x - pointB.x) + (pointA.y - pointB.y) * (pointA.y - pointB.y) + (pointA.z - pointB.z) * (pointA.z - pointB.z);
}

__device__ bool CheckAlreadAdded(KDNode_Photon_GPU* close_set, int top, KDNode_Photon_GPU* node)
{
	for (int i = 0; i < top; ++i)
	{
		if (close_set[i].photon.pos == node->photon.pos && close_set[i].photon.phi == node->photon.phi)
		{
			return true;
		}
	}
	return false;
}

__device__ void KDTreeKNNSearch(KDNode_Photon_GPU* pNode, int cur, float3 point, float3* res, float& nMinDis, Photon* distance, int rad, int* searched, int index_i, int index_j, int* pqtop)
{
	if (cur == -1)
		return;
	//printf("KNN: %d\n", cur);
	float nCurDis = GetDistanceSquare(pNode[cur].photon.pos, point);
	if (nMinDis < 0 || nCurDis < nMinDis)
	{
		nMinDis = nCurDis;
		*res = pNode[cur].photon.pos;
	}

	bool isIn = false;
	for (int i = 0; i < pqtop[index_i*UNIT_Y + index_j]; ++i)
	{
		if (searched[i] == pNode[cur].index)
		{
			isIn = true;
			break;
		}
	}
	isIn = false;
	if (pqtop[index_i*UNIT_Y + index_j] < rad)
	{
		if (!isIn)
		{
			searched[pqtop[index_i*UNIT_Y + index_j]] = pNode[cur].index;
			distance[pqtop[index_i*UNIT_Y + index_j]] = pNode[cur].photon;
			pqtop[index_i*UNIT_Y + index_j] = pqtop[index_i*UNIT_Y + index_j] + 1;
			//printf("add %d\n", pqtop[index_i*UNIT_Y + index_j]);
		}
	}
	else
	{
		if (!isIn)
		{
			float max = 0;
			int maxIndex = 0;
			for (int k = 0; k < pqtop[index_i*UNIT_Y + index_j]; ++k)
			{
				float dis = GetDistanceSquare(distance[k].pos, point);
				if (max < dis)
				{
					max = dis;
					maxIndex = k;
				}
			}
			if (max > nCurDis)
			{
				distance[maxIndex] = pNode[cur].photon;
				searched[maxIndex] = pNode[cur].index;
			}
		}
	}

	float minInQueue = 0;
	for (int i = 0; i < pqtop[index_i*UNIT_Y + index_j]; ++i)
	{
		float dis = GetDistanceSquare(distance[i].pos, point);
		if (minInQueue < dis)
		{
			minInQueue = dis;
		}
	}



	//if (pNode[cur].split == -1)return;
	if (pNode[cur].split == 0 && point.x <= pNode[cur].photon.pos.x || pNode[cur].split == 1 && point.y <= pNode[cur].photon.pos.y || pNode[cur].split == 2 && point.z <= pNode[cur].photon.pos.z)
	{
		if (pNode[cur].left != -1)KDTreeKNNSearch(pNode, pNode[cur].left, point, res, nMinDis, distance, rad, searched, index_i, index_j, pqtop);
	}
		
	else
	{
		if (pNode[cur].right != -1)KDTreeKNNSearch(pNode, pNode[cur].right, point, res, nMinDis, distance, rad, searched, index_i, index_j, pqtop);
	}
		
	float maxInQueue = 0;
	for (int i = 0; i < pqtop[index_i*UNIT_Y + index_j]; ++i)
	{
		float dis = GetDistanceSquare(distance[i].pos, point);
		if (maxInQueue < dis)
		{
			maxInQueue = dis;
		}
	}
	float rang = 0;
	switch (pNode[cur].split)
	{
	case 0:
		rang = abs(point.x - pNode[cur].photon.pos.x);
		break;
	case 1:
		rang = abs(point.y - pNode[cur].photon.pos.y);
		break;
	case 2:
		rang = abs(point.z - pNode[cur].photon.pos.z);
		break;
	}
	if (rang >= maxInQueue && pqtop[index_i*UNIT_Y + index_j] == rad)
		return;

	int axis = pNode[cur].split;
	int pGoInto = pNode[cur].right;
	switch (axis)
	{
	case 0:
		if (point.x > pNode[cur].photon.pos.x)pGoInto = pNode[cur].left;
		break;
	case 1:
		if (point.y > pNode[cur].photon.pos.y)pGoInto = pNode[cur].left;
		break;
	case 2:
		if (point.z > pNode[cur].photon.pos.z)pGoInto = pNode[cur].left;
		break;
	}
	if (pGoInto != -1)KDTreeKNNSearch(pNode, pGoInto, point, res, nMinDis, distance, rad, searched, index_i, index_j, pqtop);
}

//__device__ float KDTreeKNNSearch(KDNode_Photon_GPU* KDNodePhotonArrayTree_GPU, int cur, float3 target, float3* distance, int rad, float3* near)
//{
//	KDNode_Photon_GPU kd_point = KDNodePhotonArrayTree_GPU[cur];
//	float3 nearest = kd_point.node_pos;
//	KDNode_Photon_GPU back_point;
//
//	KDNode_Photon_GPU search_path[PHOTON_NUM];
//	KDNode_Photon_GPU close_set[PHOTON_NUM];
//	
//	int cltop = 0;
//	int top = 0;
//	int pqtop = 0;
//
//	float Min_dist = GetDistanceSquare(nearest, target);
//	int axis = 0;
//
//	while (kd_point.isValidate && axis != -1)
//	{
//		printf("Node %d\n", kd_point.Index);
//		search_path[top++] = kd_point;
//		if (GetDistanceSquare(nearest, target) > GetDistanceSquare(kd_point.node_pos, target))
//		{
//			nearest = kd_point.node_pos;
//			Min_dist = GetDistanceSquare(kd_point.node_pos, target);
//			printf("Min_dist: %f\n", Min_dist);
//		}
//
//		axis = kd_point.split;
//		//printf("axis:%d\n", axis);
//		switch (axis)
//		{
//		case 0:
//			if (target.x <= kd_point.node_pos.x)
//			{
//				cur = cur * 2 + 1;
//				kd_point = KDNodePhotonArrayTree_GPU[cur];
//			}
//			else
//			{
//				cur = cur * 2 + 2;
//				kd_point = KDNodePhotonArrayTree_GPU[cur];
//			}
//			break;
//		case 1:
//			if (target.y <= kd_point.node_pos.y)
//			{
//				cur = cur * 2 + 1;
//				kd_point = KDNodePhotonArrayTree_GPU[cur];
//			}
//			else
//			{
//				cur = cur * 2 + 2;
//				kd_point = KDNodePhotonArrayTree_GPU[cur];
//			}
//			break;
//		case 2:
//			if (target.z <= kd_point.node_pos.z)
//			{
//				cur = cur * 2 + 1;
//				kd_point = KDNodePhotonArrayTree_GPU[cur];
//			}
//			else
//			{
//				cur = cur * 2 + 2;
//				kd_point = KDNodePhotonArrayTree_GPU[cur];
//			}
//			break;
//		}
//
//		if (pqtop < rad)
//		{
//			//printf("1.Find Nearest: %d\n", pqtop);
//			distance[pqtop++] = search_path[top-1].node_pos;
//		}
//		else
//		{
//			//printf("2.Find Nearest: %d\n", pqtop);
//			float max = 0;
//			int maxIndex = 0;
//			int flag = 0;
//			for (int i = 0; i < pqtop; ++i)
//			{
//				//if (distance[i] == back_point.node_pos)continue;
//				float dis = GetDistanceSquare(distance[i], target);
//				if (max < dis)
//				{
//					max = dis;
//					maxIndex = i;
//					flag = 1;
//				}
//			}
//			if (flag)distance[maxIndex] = search_path[top-1].node_pos;
//		}
//
//		if (pqtop < rad)
//		{
//			distance[pqtop++] = search_path[top - 1].node_pos;
//		}
//		else
//		{
//			float max = 0;
//			int maxIndex = 0;
//			for (int i = 0; i < pqtop; ++i)
//			{
//				float dis = GetDistanceSquare(distance[i], target);
//				if (max < dis)
//				{
//					max = dis;
//					maxIndex = i;
//				}
//			}
//			if (GetDistanceSquare(distance[maxIndex], target) > GetDistanceSquare(search_path[top-1].node_pos,target))distance[maxIndex] = search_path[top - 1].node_pos;
//		}
//	}
//
//	while (top > 0)
//	{
//		//printf("Dead loop?%d\n%f\n",top, Min_dist);
//		
//		back_point = search_path[--top];
//		axis = back_point.split;
//		close_set[cltop++] = back_point;
//		printf("Update Node: %d\n", back_point.Index);
//		printf("Axis: %d\n", axis);
//		switch (axis)
//		{
//		case 0:
//			if ((target.x - back_point.node_pos.x) < Min_dist)
//				{
//					if (target.x <= back_point.node_pos.x)
//					{
//						kd_point = KDNodePhotonArrayTree_GPU[back_point.Index * 2 + 2];
//					}
//					else
//					{
//						kd_point = KDNodePhotonArrayTree_GPU[back_point.Index * 2 + 1];
//					}
//					if (kd_point.isValidate && !CheckAlreadAdded(close_set, cltop, &kd_point))
//					{
//						search_path[top++] = kd_point;
//						printf("Push Node %d, Top: %d\n", kd_point.Index, top);
//					}
//				}
//			break;
//		case 1:
//			if ((target.y - back_point.node_pos.y) < Min_dist)
//				{
//					if (target.y <= back_point.node_pos.y)
//					{
//						kd_point = KDNodePhotonArrayTree_GPU[back_point.Index * 2 + 2];
//					}
//					else
//					{
//						kd_point = KDNodePhotonArrayTree_GPU[back_point.Index * 2 + 1];
//					}
//					if (kd_point.isValidate && !CheckAlreadAdded(close_set, cltop, &kd_point))search_path[top++] = kd_point;
//				}
//			break;
//		case 2:
//			if ((target.z - back_point.node_pos.z) < Min_dist)
//				{
//					if (target.z <= back_point.node_pos.z)
//					{
//						kd_point = KDNodePhotonArrayTree_GPU[back_point.Index * 2 + 2];
//					}
//					else
//					{
//						kd_point = KDNodePhotonArrayTree_GPU[back_point.Index * 2 + 1];
//					}
//					if (kd_point.isValidate && !CheckAlreadAdded(close_set, cltop, &kd_point))search_path[top++] = kd_point;
//				}
//			break;
//		}
//
//		if (kd_point.isValidate && GetDistanceSquare(nearest, target) > GetDistanceSquare(kd_point.node_pos, target))
//		{
//			nearest = kd_point.node_pos;
//			Min_dist = GetDistanceSquare(kd_point.node_pos, target);
//
//			if (pqtop < rad)
//			{
//				distance[pqtop++] = kd_point.node_pos;
//			}
//			else
//			{
//				float max = 0;
//				int maxIndex = 0;
//				for (int i = 0; i < pqtop; ++i)
//				{
//					float dis = GetDistanceSquare(distance[i], target);
//					if (max < dis)
//					{
//						max = dis;
//						maxIndex = i;
//					}
//				}
//				if (GetDistanceSquare(distance[maxIndex], target) > Min_dist)distance[maxIndex] = kd_point.node_pos;
//			}
//		}
//		
//	}
//
//	*near = nearest;
//	return Min_dist;
//}