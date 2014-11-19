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
<<<<<<< HEAD

#pragma comment(lib, "glew32.lib")
=======
>>>>>>> b5d83a7e61ae665906dd895bf1a0beae3738e62c

#include "defines.h"

GLuint screenBufferPBO;
GLuint screenTexture2D;
float3 *vertexBuffer,*normalBuffer;
uchar4 *colorBuffer;
<<<<<<< HEAD
KDTriangle *kdTriangles;
=======

GLuint photonBufferPBO;
float3 *photonDirBuffer;
>>>>>>> b5d83a7e61ae665906dd895bf1a0beae3738e62c
int totalNum;
bool rendered;
struct cudaGraphicsResource* screenBufferPBO_CUDA;
struct cudaGraphicsResource* photonBufferPBO_CUDA;
#endif