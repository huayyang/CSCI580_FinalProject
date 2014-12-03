#include "global.h"

GLuint screenBufferPBO;
GLuint screenTexture2D;
Object *objects;
Material *materialBuffer;

int kd_size;
KDTriangle *kdTriangles;

Photon *photonBuffer;
GLuint photonBufferPBO;
float3 *photonDirBuffer;
Photon *photonArray;


int totalNum;
bool rendered;
struct cudaGraphicsResource* screenBufferPBO_CUDA;
struct cudaGraphicsResource* photonBufferPBO_CUDA;
