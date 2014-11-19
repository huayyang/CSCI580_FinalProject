//#pragma comment(linker,"/subsystem:windows /entry:mainCRTStartup")

#include "rayTracingProcessor.cuh"
#include "global.h"
#include <iostream>

using namespace std;
#define PHOTON_NUM 100

void readFile()  // currently is premade
{
	totalNum = 4;
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

<<<<<<< HEAD
	kdTriangles = (KDTriangle*)malloc(size * sizeof(KDTriangle));
	memset(kdTriangles, 0, size * sizeof(KDTriangle));
=======
	photonDirBuffer = (float3*)malloc(PHOTON_NUM * sizeof(float3));
	memset(photonDirBuffer, 0, PHOTON_NUM * sizeof(float3));
	cudaMalloc((void**)&photonDirBuffer_CUDA, PHOTON_NUM * sizeof(float3));
>>>>>>> b5d83a7e61ae665906dd895bf1a0beae3738e62c

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
	
	unsigned char r = (255) & 0xff;  
	unsigned char g = (255) & 0xff;  
	unsigned char b = (255) & 0xff;  
	unsigned char a = (255) & 0xff;  
	colorBuffer[0] = make_uchar4(r,0,0,a);
	colorBuffer[1] = make_uchar4(r,0,0,a);
	colorBuffer[2] = make_uchar4(r,0,0,a);
	
	colorBuffer[3] = make_uchar4(r,0,0,a);
	colorBuffer[4] = make_uchar4(r,0,0,a);
	colorBuffer[5] = make_uchar4(r,0,0,a);
	
	colorBuffer[6] = make_uchar4(0,g,0,a);
	colorBuffer[7] = make_uchar4(0,g,0,a);
	colorBuffer[8] = make_uchar4(0,g,0,a);
	
	colorBuffer[9] = make_uchar4(0,g,0,a);
	colorBuffer[10] = make_uchar4(0,g,0,a);
	colorBuffer[11] = make_uchar4(0,g,0,a);
	//

	for(int i = 0;i<PHOTON_NUM;i++)
	{
		photonDirBuffer[i] = make_float3(5-i/10,0,5-i%10);
	}
	
	cudaMemcpy(vertexBuffer_CUDA,vertexBuffer,size * sizeof(float3),cudaMemcpyHostToDevice);
	cudaMemcpy(normalBuffer_CUDA,normalBuffer,size * sizeof(float3),cudaMemcpyHostToDevice);
	cudaMemcpy(colorBuffer_CUDA,colorBuffer,size * sizeof(uchar4),cudaMemcpyHostToDevice);

<<<<<<< HEAD

	for (int i = 0; i < size; ++i)
	{
		kdTriangles[i].index = i;
		kdTriangles[i].generate_bounding_box(i);
	}
	//KDNode* KDTreeRoot = new KDNode();
	//KDTreeRoot->build();
=======
	cudaMemcpy(photonDirBuffer_CUDA,photonDirBuffer, PHOTON_NUM * sizeof(float3),cudaMemcpyHostToDevice);
>>>>>>> b5d83a7e61ae665906dd895bf1a0beae3738e62c
}

void init()  
{  
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);  
    glShadeModel(GL_SMOOTH);  

	screenBufferPBO = 0;
	photonBufferPBO = 0;
	glewInit();
    glGenBuffersARB(1, &screenBufferPBO);//����һ�����������  
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, screenBufferPBO);//������󶨵����ػ�����������������ŵ���������Ϊ��PBO��  
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, SCR_WIDTH * SCR_HEIGHT * 4 * sizeof(GLubyte), NULL, GL_DYNAMIC_COPY);//�����ڴ�ռ䲢������������Լ���ʼֵ 
	cudaGraphicsGLRegisterBuffer(&screenBufferPBO_CUDA, screenBufferPBO, cudaGraphicsMapFlagsNone);   

	glGenBuffersARB(1, &photonBufferPBO);//����һ�����������  
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, photonBufferPBO);//������󶨵����ػ�����������������ŵ���������Ϊ��PBO��  
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, SCR_WIDTH * SCR_HEIGHT * 4 * sizeof(GLubyte), NULL, GL_DYNAMIC_COPY);//�����ڴ�ռ䲢������������Լ���ʼֵ 
	cudaGraphicsGLRegisterBuffer(&photonBufferPBO_CUDA, photonBufferPBO, cudaGraphicsMapFlagsNone);   


	glEnable(GL_TEXTURE_2D);  
    glGenTextures(1, &screenTexture2D);  
    glBindTexture(GL_TEXTURE_2D, screenTexture2D);  
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);  
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);  
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


	// ==============================
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// �����Ļ����Ȼ���
	//glLoadIdentity();// ���õ�ǰ��ģ�͹۲����
	//glColor3f(1.0,1.0,0.0);

	//glEnable( GL_POINT_SMOOTH );
	//glEnable( GL_BLEND );
	//glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	//glClearDepth(1.0f);
	//glEnable(GL_DEPTH_TEST);
	//glDepthFunc(GL_LEQUAL);
	//glPointSize(20);
	//
	//glBegin(GL_POINTS);
	//	
	//	glVertex3f(0,0,0.1);
	//glEnd();
	//glFlush(); 
	// ============================

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

	rayTracingCuda(pixelPtr,totalNum,vertexBuffer_CUDA,normalBuffer_CUDA,colorBuffer_CUDA);

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
    glutReshapeFunc(reshape);  
	rendered = false;
    glutMainLoop();  

    return 0;
}