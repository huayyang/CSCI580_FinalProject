#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<algorithm>
#include <vector>
#include <map>
#include<iostream>
#include<fstream>
#include "defines.h"
using namespace std;

struct FaceInfo{
	int vertexIndex[3];
	int uvIndex[3];
	int normalIndex[3];
	char *usemtl;
};

struct MtlInfo{

	char mtlname[30];
	float Ks[3];
	float Kd[3];
	float Ka[3];
	float Tf[3];
	float Ni;
};

class ObjInfo
{
public:
	std::vector<float3> v;
	std::vector<float3> vn;
	std::vector<float2> vt;
	std::vector<FaceInfo> f;
	std::map<char *,vector<MtlInfo>> mltMap;
	void  readObj(char *objName);
	float3 worldPos;
	float3 rotate;
	float scale;
};