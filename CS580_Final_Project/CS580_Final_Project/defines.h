#ifndef DEFINES_H
#define DEFINES_H

#include <cuda_runtime.h> 

#define PI 3.14159265358979323846
#define MAX_DIS 1000000

static int UNIT_X = 128;
static int UNIT_Y = 128;

static int SCR_WIDTH = 1024;
static int SCR_HEIGHT = 1024;
static float3 CAM_POS = make_float3(10,100,50);
static float3 CAM_LOOKAT = make_float3(0,-1,0);
static float3 CAM_LOOKUP = make_float3(0,0,1);
static float3 CAM_LOOKRIGHT = make_float3(1,0,0);
//static float3 CAM_POS = make_float3(50, 100, 50);
//static float3 CAM_LOOKAT = make_float3(0, -1, 0);
//static float3 CAM_LOOKUP = make_float3(0, 0, 1);
//static float3 CAM_LOOKRIGHT = make_float3(1, 0, 0);
static float CAM_FOV = 100;

static float3 LIGHT_POS = make_float3(50,30,100);

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
}Photon;

typedef struct {
	float3 vertex[3];
	float3 normal[3];
	uchar4 color[3];
	uchar1 materialIndex;
}Object;

#endif