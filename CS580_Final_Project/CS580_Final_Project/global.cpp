#include "global.h"

GLuint screenBufferPBO;
GLuint screenTexture2D;
float3 *vertexBuffer, *normalBuffer;
uchar4 *colorBuffer;
Material *materialBuffer;
uchar1 *materialIndexBuffer;

KDTree::KDTriangle *kdTriangles;
KDTree::KDNode* KDTreeRoot;

Photon *photonBuffer;
GLuint photonBufferPBO;
float3 *photonDirBuffer;

int totalNum;
bool rendered;
struct cudaGraphicsResource* screenBufferPBO_CUDA;
struct cudaGraphicsResource* photonBufferPBO_CUDA;

//float3 operator+(const float3 &a, const float3 &b) {
//
//	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
//
//}
//
//float3 operator-(const float3 &a, const float3 &b) {
//
//	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
//
//}
//
//float3 operator*(const float3 &a, const float3 &b) {
//
//	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
//
//}
//
//float3 operator*(const float3 &a, const float &b) {
//
//	return make_float3(a.x * b, a.y * b, a.z * b);
//
//}
//
//bool operator==(const float3 &a, const float3 &b) {
//
//	return ((a.x == b.x) && (a.y == b.y) && (a.z == b.z));
//
//}
//
//float3 operator/(const float3 &a, const float3 &b) {
//
//	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
//
//}