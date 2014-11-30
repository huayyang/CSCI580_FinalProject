#include "rayTracingProcessor.cuh"
#include "KDTree.h"
#include "defines.h"
#include "math_functions.h"
#include "global.h"
#include <cuda.h>
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

__device__ uchar4 getColor(int depth, int currentIndex, uchar4 * pixels, int count, Object* objects, Material* materials, float3 pos, float3 dir, Photon* photons, KDNode_CUDA * KDTree_GPU, int* TriangleIndexArray_GPU)
{
	uchar4 resultColor;

	resultColor.x = 0;
	resultColor.y = 0;
	resultColor.z = 0;

	if (depth > 30)
	{
		printf("aaa!\n");
		return make_uchar4(255,255,255,255);
	}

	float minDis = MAX_DIS;
	int index = -1;
	float3 hitpoint;
	bool isFront = true;
	KDTriangle hitTriangle;
	float dis = INT_MAX;
	bool kdIsFront;

	int KDIndex = -1;
	float3 kdHit;
	hitTriangle.index = -1;
	
	KDTreeHit(0, objects, pos, dir, &hitpoint, &hitTriangle, &minDis, KDTree_GPU, TriangleIndexArray_GPU, &kdIsFront, currentIndex);
	
	index = hitTriangle.index;
	//printf("KD Index:%d\n", index);
	//for (int k = 0; k<count; k++)
	//{
	//	if (k == currentIndex)
	//		continue;

	//	float3 hitPos;
	//	bool isCurrentFront = true;
	//	float distance = hitSurface(objects[k].vertex, pos, dir, &hitPos,&isCurrentFront);
	//	if (distance < minDis && distance > 0.001)
	//	{
	//		isFront = isCurrentFront;
	//		minDis = distance;
	//		index = k;
	//		hitpoint.x = hitPos.x; hitpoint.y = hitPos.y; hitpoint.z = hitPos.z;
	//	}
	//}
	//isFront = kdIsFront;

	if (index != -1)
	{
		Material hitMat = materials[(int)(objects[index].materialIndex.x)];
		float Kd = hitMat.Kd;
		float Ks = hitMat.Ks;
		float Kni = hitMat.Kni;
		if (Kd > 0.001 && isFront)
		{
			int radius = 50;
			float distances[100] = { 0 };
			for (int k = 0; k<100; k++)
			{
				float3 temp;
				temp.x = hitpoint.x - photons[k].pos.x;
				temp.y = hitpoint.y - photons[k].pos.y;
				temp.z = hitpoint.z - photons[k].pos.z;
				float dis = dotProduct(temp, temp);
				distances[k] = dis;
			}
			
			int3 colorInt;
			colorInt.x = resultColor.x + Kd * objects[index].color[0].x / distances[radius] * 3000;
			colorInt.y = resultColor.y + Kd * objects[index].color[0].y / distances[radius] * 3000;
			colorInt.z = resultColor.z + Kd * objects[index].color[0].z / distances[radius] * 3000;
			resultColor.x = colorInt.x > 255 ? 255 : colorInt.x;
			resultColor.y = colorInt.y > 255 ? 255 : colorInt.y;
			resultColor.z = colorInt.z > 255 ? 255 : colorInt.z;
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

				uchar4 refractColor = getColor(depth + 1, index, pixels, count,objects, materials, hitpoint, outDir, photons, KDTree_GPU, TriangleIndexArray_GPU);
			
				int3 colorInt;
				colorInt.x = resultColor.x + Kni * refractColor.x;
				colorInt.y = resultColor.y + Kni * refractColor.y;
				colorInt.z = resultColor.z + Kni * refractColor.z;
				resultColor.x = colorInt.x > 255 ? 255 : colorInt.x;
				resultColor.y = colorInt.y > 255 ? 255 : colorInt.y;
				resultColor.z = colorInt.z > 255 ? 255 : colorInt.z;
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
			float3 reflectDir;
			reflectDir.x =lerpNormal.x * 2 * NdotDir + dir.x;
			reflectDir.y =lerpNormal.y * 2 * NdotDir + dir.y;
			reflectDir.z =lerpNormal.z * 2 * NdotDir + dir.z;

			uchar4 speculateColor = getColor(depth + 1, index, pixels, count, objects, materials, hitpoint, reflectDir, photons, KDTree_GPU, TriangleIndexArray_GPU);
			
			int3 colorInt;
			colorInt.x = resultColor.x + (Ks + tF) * speculateColor.x;
			colorInt.y = resultColor.y + (Ks + tF) * speculateColor.y;
			colorInt.z = resultColor.z + (Ks + tF) * speculateColor.z;
			resultColor.x = colorInt.x > 255 ? 255 : colorInt.x;
			resultColor.y = colorInt.y > 255 ? 255 : colorInt.y;
			resultColor.z = colorInt.z > 255 ? 255 : colorInt.z;

		}

	}

	return resultColor;
}

__global__ void kernel(int indexX, int indexY, int unitX, int unitY, uchar4 * pixels, int count, Object* objects, Material* materials, unsigned int width, unsigned int height, Camera* cam, Photon* photons, KDNode_CUDA * KDTree_GPU, int* TriangleIndexArray_GPU)
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

	pixels[id] = getColor(0, -1, pixels, count, objects, materials, cam->pos, dir, photons, KDTree_GPU, TriangleIndexArray_GPU);
	
}

__global__ void CastPhoton(uchar4 * pixels, int count, float3* vertex, Photon* photons, float3 lightPos)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	float3 dir;

	if (i >= 10 || j >= 10)
		return;

	dir.x = photons[i * 10 + j].pos.x;
	dir.y = photons[i * 10 + j].pos.y;
	dir.z = -10;
	photons[i * 10 + j].pos.x = photons[i * 10 + j].pos.y = photons[i * 10 + j].pos.z = -100;
	dir = normalize(dir);



	float minDis = MAX_DIS;
	int index = -1;
	bool isFront = true;
	for (int k = 0; k<count; k++)
	{
		float3 temp;
		float distance = hitSurface(vertex + k * 3, lightPos, dir, &temp,&isFront);
		if (distance < minDis)
		{
			minDis = distance;
			index = k;
			photons[i * 10 + j].pos.x = temp.x;
			photons[i * 10 + j].pos.y = temp.y;
			photons[i * 10 + j].pos.z = temp.z;
		}
	}
	if (index != -1)
	{
		photons[i * 10 + j].power.x = 255;
		photons[i * 10 + j].power.y = 255;
		photons[i * 10 + j].power.z = 255;
	}
	else
	{
		photons[i * 10 + j].power.x = 0;
		photons[i * 10 + j].power.y = 0;
		photons[i * 10 + j].power.z = 0;
	}
}

// Helper function for using CUDA to add vectors in parallel.
void rayTracingCuda(uchar4 * pixels, int count, Object* objects, Photon* photons, Material* materials, KDNode_CUDA * KDTree_GPU, int* TriangleIndexArray_GPU)
{
	dim3 photonBlock(10);
	dim3 photonThread(10);
	// compute light photons
	CastPhoton << <photonBlock, photonThread >> >(pixels, count, objects->vertex, photons, LIGHT_POS);
	cudaThreadSynchronize();  
	
	//Photon* photonBuffer = (Photon*)malloc(100 * sizeof(Photon));
	//cudaMemcpy(photonBuffer,photons,100 * sizeof(Photon),cudaMemcpyDeviceToHost);
	//
	//for(int i =0;i< 100;i++)
	//{
	//	std::cout<<" "<<photonBuffer[i].pos.x<<" "<<photonBuffer[i].pos.y<<" "<<photonBuffer[i].pos.z<<"\t";
	//}


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
			
			kernel << <dimblock, dimthread >> >(indexX, indexY, UNIT_X, UNIT_Y, pixels, count, objects, materials, SCR_WIDTH, SCR_HEIGHT, mainCamera_CUDA, photons, KDTree_GPU, TriangleIndexArray_GPU);

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
		bool hit_tri = false;
		if (!KDTree_GPU[cur_node].isRoot)
		{
			bool hitleft = KDTreeHit(cur_node * 2 + 1, objects, pos, dir, hitPos, hitTriangle, tmin, KDTree_GPU, TriangleIndexArray_GPU, isFront, currentIndex);
			bool hitright = KDTreeHit(cur_node * 2 + 2, objects, pos, dir, hitPos, hitTriangle, tmin, KDTree_GPU, TriangleIndexArray_GPU, isFront, currentIndex);
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