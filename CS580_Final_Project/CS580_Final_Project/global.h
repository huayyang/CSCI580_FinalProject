#ifndef GLOBAL_H
#define GLOBAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gl\glew.h>
#include <gl\glut.h>
#include <cuda_runtime.h> 
#include <cuda_gl_interop.h>
#pragma comment(lib, "glew32.lib")

#include "defines.h"

GLuint screenBufferPBO;
GLuint screenTexture2D;
float4 *vertexBuffer,*normalBuffer;
uchar4 *colorBuffer;
int totalNum;
struct cudaGraphicsResource* screenBufferPBO_CUDA;

#endif