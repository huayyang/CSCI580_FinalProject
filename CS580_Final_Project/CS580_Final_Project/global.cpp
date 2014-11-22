#include "global.h"

GLuint screenBufferPBO;
GLuint screenTexture2D;
float3 *vertexBuffer, *normalBuffer;
uchar4 *colorBuffer;
Material *materialBuffer;
uchar1 *materialIndexBuffer;

KDTree::KDTriangle *kdTriangles;

Photon *photonBuffer;
GLuint photonBufferPBO;
float3 *photonDirBuffer;

int totalNum;
bool rendered;
struct cudaGraphicsResource* screenBufferPBO_CUDA;
struct cudaGraphicsResource* photonBufferPBO_CUDA;