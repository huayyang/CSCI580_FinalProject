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



static GLuint screenBufferPBO;
static GLuint screenTexture2D;
static float3 *vertexBuffer, *normalBuffer;
static uchar4 *colorBuffer;
static Material *materialBuffer;
static uchar1 *materialIndexBuffer;

static KDTree::KDTriangle *kdTriangles;

static GLuint photonBufferPBO;
static Photon *photonBuffer;

static int totalNum;
static bool rendered;
static struct cudaGraphicsResource* screenBufferPBO_CUDA;
static struct cudaGraphicsResource* photonBufferPBO_CUDA;
#endif