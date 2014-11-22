//#pragma comment(linker,"/subsystem:windows /entry:mainCRTStartup")

#include "rayTracingProcessor.cuh"
#include "global.h"
#include <iostream>

using namespace std;
using namespace KDTree;
#define PHOTON_NUM 100

int initCornellBox()
{
	//temp
	vertexBuffer[0] = make_float3(0,0,0);
	vertexBuffer[1] = make_float3(100,0,100);
	vertexBuffer[2] = make_float3(100,0,0);
	
	vertexBuffer[3] = make_float3(0,0,0);
	vertexBuffer[4] = make_float3(0,0,100);
	vertexBuffer[5] = make_float3(100,0,100);
	
	vertexBuffer[6] = make_float3(0,0,0);
	vertexBuffer[7] = make_float3(0,100,100);
	vertexBuffer[8] = make_float3(0,100,0);
	
	vertexBuffer[9] = make_float3(0,0,0);
	vertexBuffer[10] = make_float3(0,0,100);
	vertexBuffer[11] = make_float3(0,100,100);
	
	vertexBuffer[12] = make_float3(0,0,0);
	vertexBuffer[13] = make_float3(100,0,0);
	vertexBuffer[14] = make_float3(100,100,0);
	
	vertexBuffer[15] = make_float3(0,0,0);
	vertexBuffer[16] = make_float3(100,100,0);
	vertexBuffer[17] = make_float3(0,100,0);
	
	vertexBuffer[18] = make_float3(100,0,0);
	vertexBuffer[19] = make_float3(100,100,100);
	vertexBuffer[20] = make_float3(100,100,0);
	
	vertexBuffer[21] = make_float3(100,0,0);
	vertexBuffer[22] = make_float3(100,0,100);
	vertexBuffer[23] = make_float3(100,100,100);
	
	vertexBuffer[24] = make_float3(0,0,100);
	vertexBuffer[25] = make_float3(100,0,100);
	vertexBuffer[26] = make_float3(100,100,100);
	
	vertexBuffer[27] = make_float3(0,0,100);
	vertexBuffer[28] = make_float3(100,100,100);
	vertexBuffer[29] = make_float3(0,100,100);

	normalBuffer[0] = make_float3(0,1,0);
	normalBuffer[1] = make_float3(0,1,0);
	normalBuffer[2] = make_float3(0,1,0);
	normalBuffer[3] = make_float3(0,1,0);
	normalBuffer[4] = make_float3(0,1,0);
	normalBuffer[5] = make_float3(0,1,0);
	
	normalBuffer[6] = make_float3(1,0,0);
	normalBuffer[7] = make_float3(1,0,0);
	normalBuffer[8] = make_float3(1,0,0);
	normalBuffer[9] = make_float3(1,0,0);
	normalBuffer[10] = make_float3(1,0,0);
	normalBuffer[11] = make_float3(1,0,0);
	
	normalBuffer[12] = make_float3(0,0,1);
	normalBuffer[13] = make_float3(0,0,1);
	normalBuffer[14] = make_float3(0,0,1);
	normalBuffer[15] = make_float3(0,0,1);
	normalBuffer[16] = make_float3(0,0,1);
	normalBuffer[17] = make_float3(0,0,1);
	
	normalBuffer[18] = make_float3(-1,0,0);
	normalBuffer[19] = make_float3(-1,0,0);
	normalBuffer[20] = make_float3(-1,0,0);
	normalBuffer[21] = make_float3(-1,0,0);
	normalBuffer[22] = make_float3(-1,0,0);
	normalBuffer[23] = make_float3(-1,0,0);
	
	normalBuffer[24] = make_float3(0,0,-1);
	normalBuffer[25] = make_float3(0,0,-1);
	normalBuffer[26] = make_float3(0,0,-1);
	normalBuffer[27] = make_float3(0,0,-1);
	normalBuffer[28] = make_float3(0,0,-1);
	normalBuffer[29] = make_float3(0,0,-1);
	
	unsigned char r = (255) & 0xff;  
	unsigned char g = (255) & 0xff;  
	unsigned char b = (255) & 0xff;  
	unsigned char a = (255) & 0xff;  
	colorBuffer[0] = make_uchar4(r,g,b,a);
	colorBuffer[1] = make_uchar4(r,g,b,a);
	colorBuffer[2] = make_uchar4(r,g,b,a);
	colorBuffer[3] = make_uchar4(r,g,b,a);
	colorBuffer[4] = make_uchar4(r,g,b,a);
	colorBuffer[5] = make_uchar4(r,g,b,a);
	
	colorBuffer[6] = make_uchar4(0,g,0,a);
	colorBuffer[7] = make_uchar4(0,g,0,a);
	colorBuffer[8] = make_uchar4(0,g,0,a);
	colorBuffer[9] = make_uchar4(0,g,0,a);
	colorBuffer[10] = make_uchar4(0,g,0,a);
	colorBuffer[11] = make_uchar4(0,g,0,a);
	
	colorBuffer[12] = make_uchar4(r,g,0,a);
	colorBuffer[13] = make_uchar4(r,g,0,a);
	colorBuffer[14] = make_uchar4(r,g,0,a);
	colorBuffer[15] = make_uchar4(r,g,0,a);
	colorBuffer[16] = make_uchar4(r,g,0,a);
	colorBuffer[17] = make_uchar4(r,g,0,a);
	
	colorBuffer[18] = make_uchar4(r,0,b,a);
	colorBuffer[19] = make_uchar4(r,0,b,a);
	colorBuffer[20] = make_uchar4(r,0,b,a);
	colorBuffer[21] = make_uchar4(r,0,b,a);
	colorBuffer[22] = make_uchar4(r,0,b,a);
	colorBuffer[23] = make_uchar4(r,0,b,a);
	
	colorBuffer[24] = make_uchar4(0,g,b,a);
	colorBuffer[25] = make_uchar4(0,g,b,a);
	colorBuffer[26] = make_uchar4(0,g,b,a);
	colorBuffer[27] = make_uchar4(0,g,b,a);
	colorBuffer[28] = make_uchar4(0,g,b,a);
	colorBuffer[29] = make_uchar4(0,g,b,a);


	materialIndexBuffer[0] = make_uchar1(1);
	materialIndexBuffer[1] = make_uchar1(1);
	materialIndexBuffer[2] = make_uchar1(0);
	materialIndexBuffer[3] = make_uchar1(0);
	materialIndexBuffer[4] = make_uchar1(0);
	materialIndexBuffer[5] = make_uchar1(0);
	materialIndexBuffer[6] = make_uchar1(0);
	materialIndexBuffer[7] = make_uchar1(0);
	materialIndexBuffer[8] = make_uchar1(0);
	materialIndexBuffer[9] = make_uchar1(0);
	//

	return 10;
}

void initMaterials()
{
	int materialNum = 2;
	materialBuffer = (Material*)malloc(materialNum * sizeof(Material));
	memset(materialBuffer, 0, materialNum * sizeof(Material));
	cudaMalloc((void**)&materialBuffer_CUDA, materialNum * sizeof(Material));
	
	materialBuffer[0].Kd = 1.0f;
	materialBuffer[0].Ks = 0.0f;
	materialBuffer[0].Ni = 1.0f;

	materialBuffer[1].Kd = 0.4f;
	materialBuffer[1].Ks = 0.6f;
	materialBuffer[1].Ni = 1.0f;

	cudaMemcpy(materialBuffer_CUDA,materialBuffer, materialNum * sizeof(Material),cudaMemcpyHostToDevice);
}

float3 float3Plusfloat3(float3 a, float3 b)
{
	float3 res;
	res.x = a.x+b.x;
	res.y = a.y+b.y;
	res.z = a.z+b.z;
	return res;
}

//yating
int inputModel(ObjInfo obj, int existFaceNum, float3 offset, int materialIndex)
{

	unsigned char r = (255) & 0xff;  
	unsigned char g = (255) & 0xff;  
	unsigned char b = (255) & 0xff;  
	unsigned char a = (255) & 0xff;  

	int order = existFaceNum*3; 
	int faceOrder = existFaceNum;
	int faceSize  = obj.f.size();
	int testnum = obj.f[0].vertexIndex[0];
 
	for(int i=0;i<faceSize;i++)
	{
		vertexBuffer[order] =float3Plusfloat3( obj.v[ obj.f[i].vertexIndex[0]-1],offset);
		normalBuffer[order] = obj.vn[(obj.f[i].normalIndex[0])-1];
		colorBuffer[order] = make_uchar4(r,g,b,a);

		vertexBuffer[order+1] =float3Plusfloat3( obj.v[ obj.f[i].vertexIndex[1]-1],offset);
		normalBuffer[order+1] = obj.vn[(obj.f[i].normalIndex[1])-1];
		colorBuffer[order+1] = make_uchar4(r,g,b,a);

		vertexBuffer[order+2] =float3Plusfloat3 (obj.v[ obj.f[i].vertexIndex[2]-1],offset);
		normalBuffer[order+2] = obj.vn[(obj.f[i].normalIndex[2])-1];
		colorBuffer[order+2] = make_uchar4(r,g,b,a);
		order+=3;

		materialIndexBuffer[faceOrder] = make_uchar1(materialIndex);
		faceOrder++;
	}
	return  faceSize;
}

void readFile()  // currently is premade
{ 


	totalNum =22;//10+760;
	size_t size = totalNum * 3;
	vertexBuffer = (float3*)malloc(size * sizeof(float3));
	memset(vertexBuffer, 0, size * sizeof(float3));
	cudaMalloc((void**)&vertexBuffer_CUDA,size * sizeof(float3));

	normalBuffer = (float3*)malloc(size * sizeof(float3));
	memset(normalBuffer, 0, size * sizeof(float3));
	cudaMalloc((void**)&normalBuffer_CUDA,size * sizeof(float3));
	
	colorBuffer = (uchar4*)malloc(size * sizeof(uchar4));
	memset(colorBuffer, 0, size * sizeof(uchar4));
	cudaMalloc((void**)&colorBuffer_CUDA,size * sizeof(uchar4));
	
	materialIndexBuffer = (uchar1*)malloc(totalNum * sizeof(uchar1));
	memset(materialIndexBuffer, 0, totalNum * sizeof(uchar1));
	cudaMalloc((void**)&materialIndexBuffer_CUDA,totalNum * sizeof(uchar1));


	kdTriangles = (KDTriangle*)malloc(size * sizeof(KDTriangle));
	memset(kdTriangles, 0, size * sizeof(KDTriangle));

	photonBuffer = (Photon*)malloc(PHOTON_NUM * sizeof(Photon));
	memset(photonBuffer, 0, PHOTON_NUM * sizeof(Photon));
	cudaMalloc((void**)&photonBuffer_CUDA, PHOTON_NUM * sizeof(Photon));

	//yating edit
	ObjInfo objBox;
	objBox.readObj("box30.obj"); //sphere: "sphere10.obj"  "sphere20.obj"
	int curTotalTriFace = initCornellBox();
 
	curTotalTriFace+= inputModel( objBox,curTotalTriFace,make_float3(15,15,15 ),0);
 
	for(int i = 0;i<PHOTON_NUM;i++)
	{
		photonBuffer[i].pos = make_float3(5-i/10,5-i%10,0);
		photonBuffer[i].power = make_uchar4(255,255,255,255);
	}
	
	cudaMemcpy(vertexBuffer_CUDA,vertexBuffer,size * sizeof(float3),cudaMemcpyHostToDevice);
	cudaMemcpy(normalBuffer_CUDA,normalBuffer,size * sizeof(float3),cudaMemcpyHostToDevice);
	cudaMemcpy(colorBuffer_CUDA,colorBuffer,size * sizeof(uchar4),cudaMemcpyHostToDevice);
	cudaMemcpy(materialIndexBuffer_CUDA,materialIndexBuffer,totalNum * sizeof(uchar1),cudaMemcpyHostToDevice);

	cudaMemcpy(photonBuffer_CUDA,photonBuffer, PHOTON_NUM * sizeof(Photon),cudaMemcpyHostToDevice);

	vector<KDTriangle*> tris;
	for (int i = 0; i < totalNum; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			kdTriangles[i].index[j] = 3*i+j;
		}
		kdTriangles[i].generate_bounding_box();
		tris.push_back(&kdTriangles[i]);
	}

	KDNode* KDTreeRoot = (KDNode*) malloc(sizeof(KDNode));
	KDTreeRoot = KDTreeRoot->build(tris, 0);
	float3 pos = make_float3(50, 100, 50);
	float3 dir = make_float3(-1,-1.5,0);
	dir = normalize(dir);
	float3 hitPos;
	KDTriangle hitTriangle;
	KDTreeRoot->hit(KDTreeRoot,pos,dir,&hitPos,&hitTriangle);
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
	glTexCoord2f(0.0f,1.0f); glVertex3f(0.0f,0.0f,0.0f);  
	glTexCoord2f(0.0f,0.0f); glVertex3f(0.0f,1.0f,0.0f);  
	glTexCoord2f(1.0f,0.0f); glVertex3f(1.0f,1.0f,0.0f);  
	glTexCoord2f(1.0f,1.0f); glVertex3f(1.0f,0.0f,0.0f);  
	glEnd(); 

	glutSwapBuffers();  
}  

void display()  
{  
	if(rendered)
		return;
	rendered = true;
	cout<<"rendering....\n";
	uchar4 *pixelPtr = NULL;
    size_t num_bytes;  

    cudaGraphicsMapResources(1, &screenBufferPBO_CUDA, 0);  
    cudaGraphicsResourceGetMappedPointer((void**)&pixelPtr, &num_bytes, screenBufferPBO_CUDA);  

	//rayTracingCuda(pixelPtr,totalNum,vertexBuffer_CUDA,normalBuffer_CUDA,colorBuffer_CUDA);
	rayTracingCuda(pixelPtr,totalNum,vertexBuffer_CUDA,normalBuffer_CUDA,colorBuffer_CUDA, photonBuffer_CUDA,materialBuffer_CUDA,materialIndexBuffer_CUDA);

	uchar4 * tmp = (uchar4*)malloc(sizeof(uchar4) * SCR_WIDTH * SCR_HEIGHT);
	for(int i = 0;i<100;i++)
		tmp[i].x = 0;
	cudaMemcpy(tmp,pixelPtr,sizeof(uchar4) * SCR_WIDTH * SCR_HEIGHT,cudaMemcpyDeviceToHost);

	cudaGraphicsUnmapResources(1, &screenBufferPBO_CUDA, 0);  

    draw();  
}  

void reshape(int w, int h)  
{  
    glViewport(0, 0, (GLsizei) w, (GLsizei) h);  
}

//***********************************************
// entrance function
//***********************************************
int main(int argc, char **argv)
{
	size_t value;
	cudaDeviceGetLimit(&value, cudaLimitStackSize);
	cudaDeviceSetLimit(cudaLimitStackSize, value * 2);

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