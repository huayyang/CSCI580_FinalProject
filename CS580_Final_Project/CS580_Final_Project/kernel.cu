
#include "rayTracingProcessor.cuh"
#include "defines.h"
#include "math_functions.h"
#include <cuda.h>
#include <curand.h>

__device__ unsigned int x = 123456789,
	y = 362436000,
	z = 521288629,
	c = 7654321; /* Seed variables */  

__device__ unsigned int KISS()
{   
	unsigned long long t, A = 698769069ULL;   

	x = 69069*x+12345;   

	y ^= (y<<13); y ^= (y>>17); y ^= (y<<5);   

	t = (A*z + c);
	c = (t >> 32);
	z = t;
	//return cuRAND();
	return x+y+z;   
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
	float3 pointTo0,pointTo1,pointTo2;
	float3 edge0to1,edge1to2,edge2to0;
	float3 cross0,cross1,cross2;
	float value0,value1,value2;
	
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
	
	cross0 = normalize(crossProduct(pointTo0,edge0to1));
	cross1 = normalize(crossProduct(pointTo1,edge1to2));
	cross2 = normalize(crossProduct(pointTo2,edge2to0));
	value0 = dotProduct(cross0,cross1);
	value1 = dotProduct(cross1,cross2);
	value2 = dotProduct(cross2,cross0);
	
	if ((value0 >= -0.001 && value1 >= -0.001 && value2 >= -0.001) )
		return true;
	else
		return false;
}

__device__ float checkDis(float3* vertex,float3 pos, float3 dir)
{
	//step1 calculate normal
	float3 edge1,edge2,normal;
	edge1.x = vertex[1].x - vertex[0].x;
	edge1.y = vertex[1].y - vertex[0].y;
	edge1.z = vertex[1].z - vertex[0].z;

	edge2.x = vertex[2].x - vertex[1].x;
	edge2.y = vertex[2].y - vertex[1].y;
	edge2.z = vertex[2].z - vertex[1].z;
	
	normal = normalize(crossProduct(edge1,edge2));

	//step2 calculate the projected vector
	float3 linkEdge,projectedVector;
	linkEdge.x = vertex[0].x - pos.x;
	linkEdge.y = vertex[0].y - pos.y;
	linkEdge.z = vertex[0].z - pos.z;

	float projectedValue = - dotProduct(linkEdge,normal);
	projectedVector.x = - projectedValue * normal.x;
	projectedVector.y = - projectedValue * normal.y;
	projectedVector.z = - projectedValue * normal.z;

	//step3 calculate the intersected point
	float3 intersected;
	float projectedValueOntoLine = dotProduct(projectedVector,dir);
	if (projectedValueOntoLine <= 0)
		return MAX_DIS;

	float distance = projectedValue * projectedValue / projectedValueOntoLine;
	intersected.x = pos.x + distance * dir.x;
	intersected.y = pos.y + distance * dir.y;
	intersected.z = pos.z + distance * dir.z;

	//step4 check if intersected
	if (isInside(intersected,vertex))
		return distance;
	else
		return MAX_DIS;
}


// 
__device__ float hitSurface(float3* vertex,float3 pos, float3 dir, float3* pho)
{
	//step1 calculate normal
	float3 edge1,edge2,normal;
	edge1.x = vertex[1].x - vertex[0].x;
	edge1.y = vertex[1].y - vertex[0].y;
	edge1.z = vertex[1].z - vertex[0].z;

	edge2.x = vertex[2].x - vertex[1].x;
	edge2.y = vertex[2].y - vertex[1].y;
	edge2.z = vertex[2].z - vertex[1].z;
	
	normal = normalize(crossProduct(edge1,edge2));

	//step2 calculate the projected vector
	float3 linkEdge,projectedVector;
	linkEdge.x = vertex[0].x - pos.x;
	linkEdge.y = vertex[0].y - pos.y;
	linkEdge.z = vertex[0].z - pos.z;

	float projectedValue = - dotProduct(linkEdge,normal);
	projectedVector.x = - projectedValue * normal.x;
	projectedVector.y = - projectedValue * normal.y;
	projectedVector.z = - projectedValue * normal.z;

	//step3 calculate the intersected point
	float3 intersected;
	float projectedValueOntoLine = dotProduct(projectedVector,dir);
	if (projectedValueOntoLine <= 0)
		return MAX_DIS;

	float distance = projectedValue * projectedValue / projectedValueOntoLine;
	intersected.x = pos.x + distance * dir.x;
	intersected.y = pos.y + distance * dir.y;
	intersected.z = pos.z + distance * dir.z;

	//step4 check if intersected
	if (isInside(intersected,vertex))
	{
		pho->x = intersected.x;
		pho->y = intersected.y;
		pho->z = intersected.z;
		return distance;
	}
	else
		return MAX_DIS;
}



__global__ void kernel(int indexX,int indexY,int unitX,int unitY,uchar4 * pixels,int count,float3* vertex,float3* normal,uchar4* color,unsigned int width,unsigned int height,Camera cam,Photon* photons)
{
    int i = blockIdx.x + indexX * unitX;
	int j = blockIdx.y + indexY * unitY;
	int offsetX = i - width / 2;
	int offsetY = height / 2 - j;
	float3 dir;
	dir.x = cam.lookat.x + (cam.tan_fov_2 * 2 * offsetY / height) * cam.up.x + (cam.tan_fov_2 * 2 * offsetX / height) * cam.right.x;
	dir.y = cam.lookat.y + (cam.tan_fov_2 * 2 * offsetY / height) * cam.up.y + (cam.tan_fov_2 * 2 * offsetX / height) * cam.right.y;
	dir.z = cam.lookat.z + (cam.tan_fov_2 * 2 * offsetY / height) * cam.up.z + (cam.tan_fov_2 * 2 * offsetX / height) * cam.right.z;
	
	dir = normalize(dir);

	float minDis = MAX_DIS;
	int index = -1;
	float3 hitpoint;
	for(int k =0;k<count;k++)
	{
		//float distance = checkDis(vertex + k * 3,cam.pos,dir);
		float3 temp;
		float distance = hitSurface(vertex + k * 3,cam.pos,dir,&temp);
		if (distance < minDis)
		{
			minDis = distance;
			index = k;
			hitpoint.x = temp.x;hitpoint.y = temp.y;hitpoint.z = temp.z;
		}
	}
	if (index != -1)
	{
		//for(int i = 0;i<10;i++)
		//{
		//	float3 temp ;
		//	temp.x = hitpoint.x - photons[i].pos.x;
		//	temp.y = hitpoint.y - photons[i].pos.y;
		//	temp.z = hitpoint.z - photons[i].pos.z;
		//	if(dotProduct(temp, temp) < 0.1)
		//	{
		//		color[index * 3].x = color[index * 3].y = color[index * 3].z = 120;
		//		break;
		//	}
		//}
		pixels[i + j * width].x = color[index * 3].x;
		pixels[i + j * width].y = color[index * 3].y;
		pixels[i + j * width].z = color[index * 3].z;
	}
	else
	{
		pixels[i + j * width].x = 0;
		pixels[i + j * width].y = 0;
		pixels[i + j * width].z = 0;
	}
}

// Helper function for using CUDA to add vectors in parallel.
void rayTracingCuda(uchar4 * pixels,int count,float3* vertex,float3* normal,uchar4* color)
{
	Camera cam;
	cam.pos = CAM_POS;
	cam.lookat = CAM_LOOKAT;
	cam.up = CAM_LOOKUP;
	cam.right = CAM_LOOKRIGHT;
	cam.fov = CAM_FOV;
	cam.tan_fov_2 = tan(cam.fov * PI /2 / 180);
	
	int width = SCR_WIDTH;
	int indexX = 0;
	while( width != 0)
	{
		int x;
		int height = SCR_HEIGHT;
		int indexY = 0;

		if (width > UNIT_X)
			x = UNIT_X;
		else
			x = width;

		while(height != 0)
		{
			int y;
			if (height > UNIT_Y)
				y = UNIT_Y;
			else
				y = height;

			dim3 dimblock(x,y);
			// Launch a kernel on the GPU with one thread for each element.
			//kernel<<<dimblock,1>>>(indexX,indexY,UNIT_X,UNIT_Y,pixels,count,vertex,normal,color,SCR_WIDTH,SCR_HEIGHT,cam);

			cudaThreadSynchronize();  

			height -= y;
			indexY++;
		}
		width -= x;
		indexX++;
	}

}



__global__ void kernel2(int indexX,int indexY,int unitX,int unitY,uchar4 * pixels,int count,float3* vertex,float3* normal,uchar4* color,unsigned int width,unsigned int height,Camera cam, Photon* photons, float3 lightPos)
{
    int i = blockIdx.x + indexX * unitX;
	int j = blockIdx.y + indexY * unitY;
	int offsetX = i - width / 2;
	int offsetY = height / 2 - j;
	float3 dir;
	if(i*10+j >= 100)
		return;
	dir.x = photons[i*10+j].pos.x - lightPos.x;
	dir.y = photons[i*10+j].pos.y - lightPos.y;
	dir.z = photons[i*10+j].pos.z - lightPos.z;
	dir = normalize(dir);

	float minDis = MAX_DIS;
	int index = -1;
	for(int k =0;k<count;k++)
	{
		float3 temp;
		float distance = hitSurface(vertex + k * 3,lightPos,dir,&temp);
		if (distance < minDis)
		{
			minDis = distance;
			index = k;
			photons[i*10+j].pos = temp;
		}
	}
	if (index != -1)
	{
		photons[i*10+j].power.x = 255;
		photons[i*10+j].power.y = 255;
		photons[i*10+j].power.z = 255;
	}
	else
	{
		photons[i*10+j].power.x = 0;
		photons[i*10+j].power.y = 0;
		photons[i*10+j].power.z = 0;
	}
}

// Helper function for using CUDA to add vectors in parallel.
void rayTracingCuda2(uchar4 * pixels,int count,float3* vertex,float3* normal,uchar4* color, Photon* photons)
{
	Camera cam;
	cam.pos = CAM_POS;
	cam.lookat = CAM_LOOKAT;
	cam.up = CAM_LOOKUP;
	cam.right = CAM_LOOKRIGHT;
	cam.fov = CAM_FOV;
	cam.tan_fov_2 = tan(cam.fov * PI /2 / 180);

	int width = SCR_WIDTH;
	int indexX = 0;
	while( width != 0)
	{
		int x;
		int height = SCR_HEIGHT;
		int indexY = 0;

		if (width > UNIT_X)
			x = UNIT_X;
		else
			x = width;

		while(height != 0)
		{
			int y;
			if (height > UNIT_Y)
				y = UNIT_Y;
			else
				y = height;

			dim3 dimblock(x,y);

			// compute light photons
			kernel2<<<dimblock,1>>>(indexX,indexY,UNIT_X,UNIT_Y,pixels,count,vertex,normal,color,SCR_WIDTH,SCR_HEIGHT,cam,photons,LIGHT_POS);
			// Launch a kernel on the GPU with one thread for each element.
			kernel<<<dimblock,1>>>(indexX,indexY,UNIT_X,UNIT_Y,pixels,count,vertex,normal,color,SCR_WIDTH,SCR_HEIGHT,cam,photons);

			cudaThreadSynchronize();  

			height -= y;
			indexY++;
		}
		width -= x;
		indexX++;
	}

}

