#include "rayTracingProcessor.cuh"
#include "global.h"

#include <iostream>
#include <ctime>

using namespace std;

int initCornellBox()
{
	//temp
	objects[0].vertex[0] = make_float3(0, 0, 0);
	objects[0].vertex[1] = make_float3(100, 0, 100);
	objects[0].vertex[2] = make_float3(100, 0, 0);

	objects[1].vertex[0] = make_float3(0, 0, 0);
	objects[1].vertex[1] = make_float3(0, 0, 100);
	objects[1].vertex[2] = make_float3(100, 0, 100);

	objects[2].vertex[0] = make_float3(0, 0, 0);
	objects[2].vertex[1] = make_float3(0, 150, 0);
	objects[2].vertex[2] = make_float3(0, 150, 100);

	objects[3].vertex[0] = make_float3(0, 0, 0);
	objects[3].vertex[1] = make_float3(0, 150, 100);
	objects[3].vertex[2] = make_float3(0, 0, 100);

	objects[4].vertex[0] = make_float3(0, 0, 0);
	objects[4].vertex[1] = make_float3(100, 0, 0);
	objects[4].vertex[2] = make_float3(100, 150, 0);

	objects[5].vertex[0] = make_float3(0, 0, 0);
	objects[5].vertex[1] = make_float3(100, 150, 0);
	objects[5].vertex[2] = make_float3(0, 150, 0);

	objects[6].vertex[0] = make_float3(100, 0, 0);
	objects[6].vertex[1] = make_float3(100, 150, 100);
	objects[6].vertex[2] = make_float3(100, 150, 0);

	objects[7].vertex[0] = make_float3(100, 0, 0);
	objects[7].vertex[1] = make_float3(100, 0, 100);
	objects[7].vertex[2] = make_float3(100, 150, 100);

	objects[8].vertex[0] = make_float3(0, 0, 100);
	objects[8].vertex[1] = make_float3(100, 150, 100);
	objects[8].vertex[2] = make_float3(100, 0, 100);

	objects[9].vertex[0] = make_float3(0, 0, 100);
	objects[9].vertex[1] = make_float3(0, 150, 100);
	objects[9].vertex[2] = make_float3(100, 150, 100);

	objects[10].vertex[0] = make_float3(0, 150, 0);
	objects[10].vertex[1] = make_float3(100, 150, 0);
	objects[10].vertex[2] = make_float3(100, 150, 100);

	objects[11].vertex[0] = make_float3(0, 150, 0);
	objects[11].vertex[1] = make_float3(100, 150, 100);
	objects[11].vertex[2] = make_float3(0, 150, 100);

	objects[0].normal[0] = make_float3(0, 1, 0);
	objects[0].normal[1] = make_float3(0, 1, 0);
	objects[0].normal[2] = make_float3(0, 1, 0);
	objects[1].normal[0] = make_float3(0, 1, 0);
	objects[1].normal[1] = make_float3(0, 1, 0);
	objects[1].normal[2] = make_float3(0, 1, 0);

	objects[2].normal[0] = make_float3(1, 0, 0);
	objects[2].normal[1] = make_float3(1, 0, 0);
	objects[2].normal[2] = make_float3(1, 0, 0);
	objects[3].normal[0] = make_float3(1, 0, 0);
	objects[3].normal[1] = make_float3(1, 0, 0);
	objects[3].normal[2] = make_float3(1, 0, 0);

	objects[4].normal[0] = make_float3(0, 0, 1);
	objects[4].normal[1] = make_float3(0, 0, 1);
	objects[4].normal[2] = make_float3(0, 0, 1);
	objects[5].normal[0] = make_float3(0, 0, 1);
	objects[5].normal[1] = make_float3(0, 0, 1);
	objects[5].normal[2] = make_float3(0, 0, 1);

	objects[6].normal[0] = make_float3(-1, 0, 0);
	objects[6].normal[1] = make_float3(-1, 0, 0);
	objects[6].normal[2] = make_float3(-1, 0, 0);
	objects[7].normal[0] = make_float3(-1, 0, 0);
	objects[7].normal[1] = make_float3(-1, 0, 0);
	objects[7].normal[2] = make_float3(-1, 0, 0);

	objects[8].normal[0] = make_float3(0, 0, -1);
	objects[8].normal[1] = make_float3(0, 0, -1);
	objects[8].normal[2] = make_float3(0, 0, -1);
	objects[9].normal[0] = make_float3(0, 0, -1);
	objects[9].normal[1] = make_float3(0, 0, -1);
	objects[9].normal[2] = make_float3(0, 0, -1);

	objects[10].normal[0] = make_float3(0, -1, 0);
	objects[10].normal[1] = make_float3(0, -1, 0);
	objects[10].normal[2] = make_float3(0, -1, 0);
	objects[11].normal[0] = make_float3(0, -1, 0);
	objects[11].normal[1] = make_float3(0, -1, 0);
	objects[11].normal[2] = make_float3(0, -1, 0);

	unsigned char r = (155) & 0xff;
	unsigned char g = (155) & 0xff;
	unsigned char b = (155) & 0xff;
	unsigned char a = (155) & 0xff;
	objects[0].color[0] = make_uchar4(r, g, b, a);
	objects[0].color[1] = make_uchar4(r, g, b, a);
	objects[0].color[2] = make_uchar4(r, g, b, a);
	objects[1].color[0] = make_uchar4(r, g, b, a);
	objects[1].color[1] = make_uchar4(r, g, b, a);
	objects[1].color[2] = make_uchar4(r, g, b, a);

	objects[2].color[0] = make_uchar4(0, g, 0, a);
	objects[2].color[1] = make_uchar4(0, g, 0, a);
	objects[2].color[2] = make_uchar4(0, g, 0, a);
	objects[3].color[0] = make_uchar4(0, g, 0, a);
	objects[3].color[1] = make_uchar4(0, g, 0, a);
	objects[3].color[2] = make_uchar4(0, g, 0, a);

	objects[4].color[0] = make_uchar4(r, g, 0, a);
	objects[4].color[1] = make_uchar4(r, g, 0, a);
	objects[4].color[2] = make_uchar4(r, g, 0, a);
	objects[5].color[0] = make_uchar4(r, g, 0, a);
	objects[5].color[1] = make_uchar4(r, g, 0, a);
	objects[5].color[2] = make_uchar4(r, g, 0, a);

	objects[6].color[0] = make_uchar4(r, 0, b, a);
	objects[6].color[1] = make_uchar4(r, 0, b, a);
	objects[6].color[2] = make_uchar4(r, 0, b, a);
	objects[7].color[0] = make_uchar4(r, 0, b, a);
	objects[7].color[1] = make_uchar4(r, 0, b, a);
	objects[7].color[2] = make_uchar4(r, 0, b, a);

	objects[8].color[0] = make_uchar4(0, g, b, a);
	objects[8].color[1] = make_uchar4(0, g, b, a);
	objects[8].color[2] = make_uchar4(0, g, b, a);
	objects[9].color[0] = make_uchar4(0, g, b, a);
	objects[9].color[1] = make_uchar4(0, g, b, a);
	objects[9].color[2] = make_uchar4(0, g, b, a);

	objects[10].color[0] = make_uchar4(0, 0, b, a);
	objects[10].color[1] = make_uchar4(0, 0, b, a);
	objects[10].color[2] = make_uchar4(0, 0, b, a);
	objects[11].color[0] = make_uchar4(0, 0, b, a);
	objects[11].color[1] = make_uchar4(0, 0, b, a);
	objects[11].color[2] = make_uchar4(0, 0, b, a);

	objects[0].materialIndex = make_uchar1(0);
	objects[1].materialIndex = make_uchar1(0);
	objects[2].materialIndex = make_uchar1(0);
	objects[3].materialIndex = make_uchar1(0);
	objects[4].materialIndex = make_uchar1(0);
	objects[5].materialIndex = make_uchar1(0);
	objects[6].materialIndex = make_uchar1(0);
	objects[7].materialIndex = make_uchar1(0);
	objects[8].materialIndex = make_uchar1(0);
	objects[9].materialIndex = make_uchar1(0);

	objects[10].materialIndex = make_uchar1(0);
	objects[11].materialIndex = make_uchar1(0);
	//

	return 12;
}

void initMaterials()
{
	int materialNum = 3;
	materialBuffer = (Material*)malloc(materialNum * sizeof(Material));
	memset(materialBuffer, 0, materialNum * sizeof(Material));
	cudaMalloc((void**)&materialBuffer_CUDA, materialNum * sizeof(Material));

	materialBuffer[0].Kd = 1.0f;
	materialBuffer[0].Ks = 0.0f;
	materialBuffer[0].Kni = 0.0f;
	materialBuffer[0].Ni = 1.0f;

	materialBuffer[1].Kd = 0.0f;
	materialBuffer[1].Ks = 1.0f;
	materialBuffer[1].Kni = 0.0f;
	materialBuffer[1].Ni = 1.00f;

	materialBuffer[2].Kd = 0.0f;
	materialBuffer[2].Ks = 0.0f;
	materialBuffer[2].Kni = 1.1f;
	materialBuffer[2].Ni = 1.5f;

	cudaMemcpy(materialBuffer_CUDA, materialBuffer, materialNum * sizeof(Material), cudaMemcpyHostToDevice);
}


float3 float3Plusfloat3(float3 a, float3 b)
{
	float3 res;
	res.x = a.x + b.x;
	res.y = a.y + b.y;
	res.z = a.z + b.z;
	return res;
}


//yating
int inputModel(ObjInfo obj, int existFaceNum, float3 offset, int materialIndex, uchar4 objColors)
{
	int faceOrder = existFaceNum;
	int faceSize = obj.f.size();
	int testnum = obj.f[0].vertexIndex[0];

	for (int i = 0; i<faceSize; i++)
	{
		objects[faceOrder].vertex[0] = float3Plusfloat3(obj.v[obj.f[i].vertexIndex[1] - 1], offset);
		objects[faceOrder].vertex[1] = float3Plusfloat3(obj.v[obj.f[i].vertexIndex[0] - 1], offset);
		objects[faceOrder].vertex[2] = float3Plusfloat3(obj.v[obj.f[i].vertexIndex[2] - 1], offset);
		
		objects[faceOrder].normal[0] = obj.vn[(obj.f[i].normalIndex[1]) - 1];
		objects[faceOrder].normal[1] = obj.vn[(obj.f[i].normalIndex[0]) - 1];
		objects[faceOrder].normal[2] = obj.vn[(obj.f[i].normalIndex[2]) - 1];

		objects[faceOrder].color[0] = objColors;
		objects[faceOrder].color[1] = objColors;
		objects[faceOrder].color[2] = objColors;

		objects[faceOrder].materialIndex = make_uchar1(materialIndex);
		faceOrder++;
	}
	return  faceSize;
}

void readFile()  // currently i s premade
{
	//totalNum =  10+4968;
	//totalNum = 22 + 4968;
	totalNum = 772;
	//totalNum = 10 + 871168;
	//totalNum = 12;
	size_t size = totalNum;
	objects = (Object*)malloc(size * sizeof(Object));
	memset(objects, 0, size * sizeof(Object));
	cudaMalloc((void**)&objects_CUDA, size * sizeof(Object));

	kdTriangles = (KDTriangle*)malloc(totalNum * sizeof(KDTriangle));
	memset(kdTriangles, 0, totalNum * sizeof(KDTriangle));

	photonBuffer = (Photon*)malloc(PHOTON_NUM * sizeof(Photon));
	memset(photonBuffer, 0, PHOTON_NUM * sizeof(Photon));
	cudaMalloc((void**)&photonBuffer_CUDA, PHOTON_NUM * sizeof(Photon));

	//yating edit
	ObjInfo objBox;
	ObjInfo objBox2;
	ObjInfo objBox3;
	ObjInfo objBox4;
	//objBox.readObj("dragon.obj"); //sphere: "sphere10.obj"  "sphere20.obj"  rab.obj
	//objBox.readObj("sphere20.obj");
	//objBox2.readObj("sphere20.obj");
	//objBox3.readObj("sphere20.obj");
	objBox4.readObj("sphere20.obj");
	int curTotalTriFace = initCornellBox();
	
	uchar4 boxColor = make_uchar4(90,90,90,255);
	//curTotalTriFace+= inputModel( objBox2,curTotalTriFace,make_float3(20,15, 13), 1, boxColor);
	//curTotalTriFace+= inputModel( objBox,curTotalTriFace,make_float3(50, 15, 13 ), 1, boxColor);
	//curTotalTriFace+= inputModel( objBox3,curTotalTriFace,make_float3(80, 15, 13 ), 1, boxColor);
	curTotalTriFace+= inputModel( objBox4,curTotalTriFace,make_float3(20, 40, 20 ), 2, boxColor);
	//srand((unsigned)time(NULL));
	//for (int i = 0; i<PHOTON_NUM; i++)
	//{
	//	float randx = 1; 
	//	float randy = 1;
	//	float randz = 1;
	//	while((randx * randx + randy * randy + randz*randz) > 1)
	//	{
	//		randx = (rand() % 10000 - 5000) / 5000.0;
	//		randy = (rand() % 10000 - 5000) / 5000.0;
	//		randz = (rand() % 10000 - 5000) / 5000.0;
	//	}
	//	//cout<<randx<<" "<<randy<<" "<<randz<<endl;
	//	//randx = (PHOTON_SQR/2.0 - i/PHOTON_SQR) / (PHOTON_SQR/2);
	//	//randy = (PHOTON_SQR/2.0 - i%PHOTON_SQR) / (PHOTON_SQR/2);
	//	
	//	if(randz > -0.3)
	//	{
	//		i--;
	//		continue;
	//	}
	//	photonBuffer[i].pos = make_float3(randx, randy, randz);
	//	photonBuffer[i].power = make_uchar4(255, 255, 255, 255);
	//}
	cout<<endl;
	float total = 0;
	for (int i = 0;i<PHOTON_SQR;i++)
	{
		double tha = - 90.0 + i * 120.0 / PHOTON_SQR;tha = tha / 180 * PI;
		total += cos(tha);
		cout<<tha<<"\t"<<i<<endl;
	}
	cout<<total<<endl;
	int curindex = 0;
	//for(int i = 0;i<PHOTON_SQR;i++)
	//{
	//	double tha = - 90.0 + i * 120.0 / PHOTON_SQR;tha = tha / 180 * PI;
	//
	//	for(int j = 0;j < (PHOTON_NUM * cos(tha) / total + 10) ;j++)
	//	{
	//		double the = 180.0 - j * 360.0 / (PHOTON_NUM * cos(tha) / total + 10);
	//		the = the / 180 * PI;
	//		float randx = cos(the) * cos(tha);
	//		float randy = sin(the) * cos(tha);
	//		float randz = sin(tha);
	//
	//		photonBuffer[curindex].pos = make_float3(randx, randy, randz);
	//		cout<<the<<"\t"<<tha<<"\t"<<randx<<"\t"<<randy<<"\t"<<randz<<"\t"<<curindex<<endl;
	//		photonBuffer[curindex++].power = make_uchar4(255, 255, 255, 255);
	//		if(curindex >= PHOTON_NUM)
	//			break;
	//	}
	//	if(curindex >= PHOTON_NUM)
	//		break;
	//}

	int globalp = PHOTON_NUM/2;
	for(int i = 0; i<globalp;i++)
	{
		float k = 1-(2.0*i-1)/(globalp);
		float tha = asin(k);
		float the = tha * sqrt(globalp * PI);

		float randx = cos(the) * cos(tha);
		float randy = sin(the) * cos(tha);
		float randz = sin(tha);

		photonBuffer[i].pos = make_float3(randx, randy, randz);
		photonBuffer[i].power = make_uchar4(255, 255, 255, 255);
	}
	int directionp =  PHOTON_NUM/2 ;
	for(int i = 0;i<directionp;i++)
	{

		float k = 1-(2.0*i-1)/(directionp*3);
		float tha = asin(k);
		float the = tha * sqrt((directionp*3) * PI);

		float randx = cos(the) * cos(tha);
		float randy = sin(the) * cos(tha);
		float randz = -sin(tha);
		//randz = randz > 0?-randz:randz;

		photonBuffer[i + globalp].pos = make_float3(randx, randy, randz);
		photonBuffer[i + globalp].power = make_uchar4(255, 255, 255, 255);
	}

	cudaMemcpy(objects_CUDA, objects, size * sizeof(Object), cudaMemcpyHostToDevice);

	cudaMemcpy(photonBuffer_CUDA, photonBuffer, PHOTON_NUM * sizeof(Photon), cudaMemcpyHostToDevice);

	/* KDTree For Triangles */
	vector<KDTriangle*>tris;
	for (int i = 0; i < totalNum; ++i)
	{
		kdTriangles[i].index = i;
		kdTriangles[i].generate_bounding_box();
		tris.push_back(&kdTriangles[i]);
	}

	TriangleIndexArray_CPU = (int*)malloc(sizeof(int) * totalNum);
	TI_cur = 0;
	KDTreeRoot_CPU = KDTreeRoot_CPU->build(tris, 0, TriangleIndexArray_CPU);

	cudaMalloc(&TriangleIndexArray_GPU, sizeof(int) * totalNum);
	cudaMemcpy(TriangleIndexArray_GPU, TriangleIndexArray_CPU, sizeof(int) * totalNum, cudaMemcpyHostToDevice);

	int node_num = treeSize(KDTreeRoot_CPU);
	int arraySize = node_num * sizeof(KDNode_CUDA);
	
	KDTree_CPU = (KDNode_CUDA*)malloc(arraySize);
	memset(KDTree_CPU, 0, _msize(KDTree_CPU));

	setKDNodeIndex(KDTreeRoot_CPU, 0);

	copyKDTreeToArray(KDTree_CPU, KDTreeRoot_CPU, 0);
	cudaMalloc(&KDTree_GPU, arraySize);
	cudaMemcpy(KDTree_GPU, KDTree_CPU, arraySize, cudaMemcpyHostToDevice);
}

void init()
{
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glShadeModel(GL_SMOOTH);

	screenBufferPBO = 0;
	photonBufferPBO = 0;
	glewInit();
	glGenBuffersARB(1, &screenBufferPBO);//生成一个缓冲区句柄  
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, screenBufferPBO);//将句柄绑定到像素缓冲区（即缓冲区存放的数据类型为：PBO）  
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, SCR_WIDTH * SCR_HEIGHT * 4 * sizeof(GLubyte), NULL, GL_DYNAMIC_COPY);//申请内存空间并设置相关属性以及初始值 
	cudaGraphicsGLRegisterBuffer(&screenBufferPBO_CUDA, screenBufferPBO, cudaGraphicsMapFlagsNone);

	glGenBuffersARB(1, &photonBufferPBO);//生成一个缓冲区句柄  
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, photonBufferPBO);//将句柄绑定到像素缓冲区（即缓冲区存放的数据类型为：PBO）  
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, SCR_WIDTH * SCR_HEIGHT * 4 * sizeof(GLubyte), NULL, GL_DYNAMIC_COPY);//申请内存空间并设置相关属性以及初始值 
	cudaGraphicsGLRegisterBuffer(&photonBufferPBO_CUDA, photonBufferPBO, cudaGraphicsMapFlagsNone);


	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &screenTexture2D);
	glBindTexture(GL_TEXTURE_2D, screenTexture2D);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	initMaterials();
}

void draw()
{
	glBindTexture(GL_TEXTURE_2D, screenTexture2D);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, screenBufferPBO);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SCR_WIDTH, SCR_HEIGHT,/*window_width, window_height,*/
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f, 0.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 0.0f, 0.0f);
	glEnd();

	glutSwapBuffers();
}

void display()
{
	if (rendered)
		return;
	rendered = true;
	cout << "rendering....\n";
	uchar4 *pixelPtr = NULL;
	size_t num_bytes;

	cudaGraphicsMapResources(1, &screenBufferPBO_CUDA, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&pixelPtr, &num_bytes, screenBufferPBO_CUDA);

	//rayTracingCuda(pixelPtr,totalNum,vertexBuffer_CUDA,normalBuffer_CUDA,colorBuffer_CUDA);
	//rayTracingCuda(pixelPtr, totalNum, objects_CUDA, photonBuffer_CUDA, materialBuffer_CUDA, KDTree_GPU, TriangleIndexArray_GPU, KDNodePhotonArrayTree_GPU, KDPhotonArray_GPU, KDNodePhotonArrayTree_CPU);
	rayTracingCuda(pixelPtr, totalNum, objects_CUDA, photonBuffer_CUDA, materialBuffer_CUDA, KDTree_GPU, TriangleIndexArray_GPU);

	uchar4 * tmp = (uchar4*)malloc(sizeof(uchar4) * SCR_WIDTH * SCR_HEIGHT);
	for (int i = 0; i<100; i++)
		tmp[i].x = 0;
	cudaMemcpy(tmp, pixelPtr, sizeof(uchar4) * SCR_WIDTH * SCR_HEIGHT, cudaMemcpyDeviceToHost);

	cudaGraphicsUnmapResources(1, &screenBufferPBO_CUDA, 0);

	draw();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
}

//***********************************************
// entrance function
//***********************************************
int main(int argc, char **argv)
{
	size_t value;
	cudaDeviceGetLimit(&value, cudaLimitStackSize);
	cudaDeviceSetLimit(cudaLimitStackSize, value * 256);

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(SCR_WIDTH, SCR_HEIGHT);
	glutCreateWindow("OpenGL Hello World");
	init();
	readFile();
	glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);
	glutDisplayFunc(display);
	//glutReshapeFunc(reshape);  
	rendered = false;
	glutMainLoop();

	return 0;
}

