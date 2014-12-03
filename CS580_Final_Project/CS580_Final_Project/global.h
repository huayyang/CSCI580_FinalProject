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

#define PHOTON_NUM 3600
#define PHOTON_SQR 60
#define PHOTON_RADIUS 30
#define PHOTON_FORCE 100
#define PHOTON_ANGLE 1.0
#define PHOTON_DIFFUSE_RATE 1
#define PHOTON_SHOW true

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

