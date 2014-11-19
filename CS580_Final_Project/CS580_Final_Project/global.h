#ifndef GLOBAL_H
#define GLOBAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#pragma comment(lib, "glew32.lib")

#include <gl\glew.h>
#include <gl\glut.h>
#include <cuda_runtime.h> 
#include <cuda_gl_interop.h>

#include "defines.h"

GLuint screenBufferPBO;
GLuint screenTexture2D;
float3 *vertexBuffer,*normalBuffer;
uchar4 *colorBuffer;

GLuint photonBufferPBO;
float3 *photonDirBuffer;
int totalNum;
bool rendered;
struct cudaGraphicsResource* screenBufferPBO_CUDA;
struct cudaGraphicsResource* photonBufferPBO_CUDA;
#endif