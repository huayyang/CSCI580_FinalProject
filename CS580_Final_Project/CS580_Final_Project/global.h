#ifndef GLOBAL_H
#define GLOBAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "defines.h"
#include "KDTree.h"
#include "ReadObj.h"
#pragma comment(lib, "glew32.lib")

#include <gl\glew.h>
#include <gl\glut.h>
#include <cuda_runtime.h> 
#include <cuda_gl_interop.h>


#pragma comment(lib, "glew32.lib")

extern GLuint screenBufferPBO;
extern GLuint screenTexture2D;
extern float3 *vertexBuffer, *normalBuffer;
extern uchar4 *colorBuffer;

extern KDTree::KDTriangle *kdTriangles;

extern GLuint photonBufferPBO;
extern Photon *photonBuffer;

extern int totalNum;
extern bool rendered;
extern struct cudaGraphicsResource* screenBufferPBO_CUDA;
extern struct cudaGraphicsResource* photonBufferPBO_CUDA;

float3 operator+(const float3 &a, const float3 &b);
float3 operator-(const float3 &a, const float3 &b);
float3 operator*(const float3 &a, const float3 &b);
float3 operator*(const float3 &a, const float &b);
float3 operator/(const float3 &a, const float3 &b);
bool operator==(const float3 &a, const float3 &b);
#endif
