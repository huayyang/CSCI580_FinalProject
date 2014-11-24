#include "rayTracingProcessor.cuh"
#include "defines.h"
#include "math_functions.h"
#include <cuda.h>
#include <curand.h>
#include <iostream>

__device__ unsigned int x = 123456789,
y = 362436000,
z = 521288629,
c = 7654321; /* Seed variables */

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

__device__ float3 getR(float ni, float3 I, float3 N)
{
	float3 planeNormal = crossProduct(N,I);
	float3 biNormal = crossProduct(planeNormal,N);
	float sinI = dotProduct(biNormal,I);
	float sinR = sinI / ni;
	float cosR = sqrtf(1 - sinR *sinR);

	float3 R;
	R.x = biNormal.x * sinR - N.x * cosR;
	R.y = biNormal.y * sinR - N.y * cosR;
	R.z = biNormal.z * sinR - N.z * cosR;

	return R;
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

__device__ uchar4 getColor(int depth,int currentIndex, uchar4 * pixels, int count, float3* vertex, float3* normal, uchar4* color, Material* materials, uchar1* materialIndex, float3 pos, float3 dir, Photon* photons)
{
	uchar4 resultColor;

	resultColor.x = 0;
	resultColor.y = 0;
	resultColor.z = 0;

	if (depth > 10)
		return resultColor;

	float minDis = MAX_DIS;
	int index = -1;
	float3 hitpoint;
	bool isFront = true;

	for (int k = 0; k<count; k++)
	{
		if (k == currentIndex)
			continue;

		float3 hitPos;
		bool isCurrentFront = true;
		float distance = hitSurface(vertex + k * 3, pos, dir, &hitPos,&isCurrentFront);
		if (distance < minDis && distance > 0.001)
		{
			isFront = isCurrentFront;
			minDis = distance;
			index = k;
			hitpoint.x = hitPos.x; hitpoint.y = hitPos.y; hitpoint.z = hitPos.z;
		}
	}

	if (index != -1)
	{
		//printf("%d\n",(int)(materialIndex[index].x));
		Material hitMat = materials[(int)(materialIndex[index].x)];
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
			// sort and get the middle distance
			//splitSort(distances,100,0,99);

			//if (currentIndex != -1)
			//	printf("%d \n",currentIndex);
			
			int3 colorInt;
			colorInt.x = resultColor.x + Kd * color[index * 3].x / distances[radius] * 3000;
			colorInt.y = resultColor.y + Kd * color[index * 3].y / distances[radius] * 3000;
			colorInt.z = resultColor.z + Kd * color[index * 3].z / distances[radius] * 3000;
			resultColor.x = colorInt.x > 255 ? 255 : colorInt.x;
			resultColor.y = colorInt.y > 255 ? 255 : colorInt.y;
			resultColor.z = colorInt.z > 255 ? 255 : colorInt.z;
		}

		// sort and get the middle distance
		//splitSort(distances,100,0,99);

		if (Ks > 0.001 && isFront)
		{
			float NdotDir = -dotProduct(normal[index * 3], dir);
			float3 reflectDir;
			reflectDir.x = normal[index * 3].x * 2 * NdotDir + dir.x;
			reflectDir.y = normal[index * 3].y * 2 * NdotDir + dir.y;
			reflectDir.z = normal[index * 3].z * 2 * NdotDir + dir.z;
			//printf("%d %f %f %f \n",currentIndex,reflectDir.x,reflectDir.y,reflectDir.z);
			//printf("%d %f %f %f \n",currentIndex,normal[index * 3].x,normal[index * 3].y,normal[index * 3].z);
			//printf("%d %f %f %f \n\n",currentIndex,dir.x,dir.y,dir.z);

			uchar4 speculateColor = getColor(depth+1,index, pixels, count, vertex, normal, color, materials, materialIndex, hitpoint, reflectDir, photons);
			
			int3 colorInt;
			colorInt.x = resultColor.x + Ks * speculateColor.x;
			colorInt.y = resultColor.y + Ks * speculateColor.y;
			colorInt.z = resultColor.z + Ks * speculateColor.z;
			resultColor.x = colorInt.x > 255 ? 255 : colorInt.x;
			resultColor.y = colorInt.y > 255 ? 255 : colorInt.y;
			resultColor.z = colorInt.z > 255 ? 255 : colorInt.z;

		}

		if (Kni > 0.001)
		{
			float Ni = hitMat.Ni;
			float3 outDir;
			float3 n;
			n.x = -normal[index * 3].x;
			n.y = -normal[index * 3].y;
			n.z = -normal[index * 3].z;
			if (isFront)
			{
				outDir = getR(Ni,dir,normal[index * 3]);
				//printf("%d %f %f %f\t\t %f %f %f \t\t %f %f %f\n",depth,dir.x,dir.y,dir.z,normal[index * 3].x,normal[index * 3].y,normal[index * 3].z,outDir.x,outDir.y,outDir.z);
			}
			else
			{
				outDir = getR(1/Ni,dir,n);
				printf("%d %f %f %f\t\t %f %f %f \t\t %f %f %f\n",depth,dir.x,dir.y,dir.z,normal[index * 3].x,normal[index * 3].y,normal[index * 3].z,outDir.x,outDir.y,outDir.z);
			}
			//printf("%d %f %f %f \n",currentIndex,reflectDir.x,reflectDir.y,reflectDir.z);
			//printf("%d %f %f %f \n",currentIndex,normal[index * 3].x,normal[index * 3].y,normal[index * 3].z);
			//printf("%d %f %f %f \n\n",currentIndex,dir.x,dir.y,dir.z);

			uchar4 refractColor = getColor(depth+1,index, pixels, count, vertex, normal, color, materials, materialIndex, hitpoint, outDir, photons);
			
			int3 colorInt;
			colorInt.x = resultColor.x + Kni * refractColor.x;
			colorInt.y = resultColor.y + Kni * refractColor.y;
			colorInt.z = resultColor.z + Kni * refractColor.z;
			resultColor.x = colorInt.x > 255 ? 255 : colorInt.x;
			resultColor.y = colorInt.y > 255 ? 255 : colorInt.y;
			resultColor.z = colorInt.z > 255 ? 255 : colorInt.z;

		}
	}

	return resultColor;
}

__global__ void kernel(int indexX,int indexY,int unitX,int unitY,uchar4 * pixels,int count,float3* vertex,float3* normal,uchar4* color,Material* materials,uchar1* materialIndex,unsigned int width,unsigned int height,Camera* cam,Photon* photons)
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
	
	pixels[id] = getColor(0,-1,pixels,count,vertex,normal,color,materials,materialIndex,cam->pos,dir,photons);
	
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
void rayTracingCuda(uchar4 * pixels, int count, float3* vertex, float3* normal, uchar4* color, Photon* photons, Material* materials, uchar1* materialIndex)
{
	dim3 photonBlock(10);
	dim3 photonThread(10);
	// compute light photons
	CastPhoton<<<photonBlock,photonThread>>>(pixels,count,vertex,photons,LIGHT_POS);
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
			
			kernel<<<dimblock,dimthread>>>(indexX,indexY,UNIT_X,UNIT_Y,pixels,count,vertex,normal,color,materials,materialIndex,SCR_WIDTH,SCR_HEIGHT,mainCamera_CUDA,photons);

			cudaThreadSynchronize();

			height -= y;
			indexY++;
		}
		width -= x;
		indexX++;
	}

}
