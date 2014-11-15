#ifndef DEFINES_H
#define DEFINES_H

#include <cuda_runtime.h> 

#define PI 3.14159265358979323846
#define MAX_DIS 1000000

static int SCR_WIDTH = 256;
static int SCR_HEIGHT = 256;
static float3 CAM_POS = make_float3(50,200,50);
static float3 CAM_LOOKAT = make_float3(0,-1,0);
static float3 CAM_LOOKUP = make_float3(0,0,1);
static float3 CAM_LOOKRIGHT = make_float3(1,0,0);
static float CAM_FOV = 60;

typedef struct {
	float3 pos,lookat,up,right;
	float fov;
	float tan_fov_2;
}Camera;

#endif