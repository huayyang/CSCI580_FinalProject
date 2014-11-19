#include "global.h"

GLuint screenBufferPBO;
GLuint screenTexture2D;
float3 *vertexBuffer, *normalBuffer;
uchar4 *colorBuffer;

KDTree::KDTriangle *kdTriangles;

GLuint photonBufferPBO;
float3 *photonDirBuffer;

int totalNum;
bool rendered;
struct cudaGraphicsResource* screenBufferPBO_CUDA;
struct cudaGraphicsResource* photonBufferPBO_CUDA;