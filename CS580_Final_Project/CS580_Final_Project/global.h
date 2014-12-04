#ifndef GLOBAL_H
#define GLOBAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "defines.h"
#include "ReadObj.h"
#include "rayTracingProcessor.cuh"
#pragma comment(lib, "glew32.lib")

#include <gl\glew.h>
#include <gl\glut.h>
#include <cuda_runtime.h> 
#include <cuda_gl_interop.h>


#pragma comment(lib, "glew32.lib")

#define PHOTON_NUM 900
#define PHOTON_SQR 30
#define PHOTON_RADIUS 10
#define PHOTON_FORCE 50
#define ADD_FORCE 1000
#define PHOTON_ANGLE 1.0
#define PHOTON_DIFFUSE_RATE 1
#define PHOTON_SHOW false

#define BALL_SHOW false
#define BALL_POS_X 20
#define BALL_POS_Y 40
#define BALL_POS_Z 20
#define BALL_MAT 2
#define BALL_R 15
#define BALL_COLOR_R 255
#define BALL_COLOR_G 255
#define BALL_COLOR_B 255
#define BALL_COLOR_A 255

extern GLuint screenBufferPBO;
extern GLuint screenTexture2D;
extern Object *objects;
extern Material *materialBuffer;

extern int kd_size;
extern KDTriangle *kdTriangles;

extern GLuint photonBufferPBO;
extern Photon *photonBuffer;
extern Photon *photonArray;


extern int totalNum;
extern bool rendered;
extern struct cudaGraphicsResource* screenBufferPBO_CUDA;
extern struct cudaGraphicsResource* photonBufferPBO_CUDA;

//float3 operator+(const float3 &a, const float3 &b);
//float3 operator-(const float3 &a, const float3 &b);
//float3 operator*(const float3 &a, const float3 &b);
//float3 operator*(const float3 &a, const float &b);
//float3 operator/(const float3 &a, const float3 &b);
//bool operator==(const float3 &a, const float3 &b);

#endif

