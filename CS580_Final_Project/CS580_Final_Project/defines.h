#ifndef DEFINES_H
#define DEFINES_H

#include <cuda_runtime.h> 

#define PI 3.14159265358979323846
#define MAX_DIS 1000000

#define UNIT_X 64
#define UNIT_Y 64

static int SCR_WIDTH = 800;
static int SCR_HEIGHT = 800;
//static float3 CAM_POS = make_float3(10,100,50);
//static float3 CAM_LOOKAT = make_float3(0,-1,0);
//static float3 CAM_LOOKUP = make_float3(0,0,1);
//static float3 CAM_LOOKRIGHT = make_float3(1,0,0);
static float3 CAM_POS = make_float3(50, 140, 40);
static float3 CAM_LOOKAT = make_float3(0, -1, 0);
static float3 CAM_LOOKUP = make_float3(0, 0, 1);
static float3 CAM_LOOKRIGHT = make_float3(1, 0, 0);
static float CAM_FOV = 90;

static float3 LIGHT_POS = make_float3(50,50,80);

typedef struct {
	float3 pos,lookat,up,right;
	float fov;
	float tan_fov_2;
}Camera;

typedef struct {
	float Kd; //diffuse reflection
	float Ks; //speculate reflection
	float Kni; //refraction
	float Ni; //refraction
}Material;

typedef struct {
	float3 pos;
	uchar4 power;
	char phi, theta;
	float distance;
}Photon;

typedef struct {
	float3 vertex[3];
	float3 normal[3];
	uchar4 color[3];
	uchar1 materialIndex;
}Object;

#endif