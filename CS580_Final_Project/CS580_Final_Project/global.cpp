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

int totalNum;
bool rendered;
struct cudaGraphicsResource* screenBufferPBO_CUDA;
struct cudaGraphicsResource* photonBufferPBO_CUDA;

//
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
