#include <GL/gl.h>
#include <GL/glut.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string.h>
#include <assert.h>
#include <map>
#define WIDTH 2000
#define HEIGHT 2000
#define TRIANGLES 10000
#define BUNNYHEIGHT 400
#define BUNNYWIDTH 400

using namespace std;
extern float image[WIDTH][HEIGHT][3];
float parImage[WIDTH][HEIGHT][3];

GLfloat light_diffuse[] = {0.5, 0.5, 0.5, 1.0};  /* Red diffuse light. */
GLfloat light_position[] = {-0.2, 1.0, -1, 0.0};  /* Infinite light location. */


typedef struct Tri
{
    float x1, y1, z1,
          x2, y2, z2, 
          x3, y3, z3;

    float x1_2d, y1_2d,
          x2_2d, y2_2d,
          x3_2d, y3_2d;  

    float bboxLeft, bboxTop, bboxWidth, bboxHeight;

    float R1, G1, B1, 
          R2, G2, B2,
          R3, G3, B3;
        
    float normalX, normalY, normalZ;

} Tri;

extern void Parse(int type);
extern Tri Triangles[TRIANGLES];

Tri *cudaTri;
float *Depthbuffer;
float *img;
int *TriIndex;

void imageWrite()
{

  int x, y, w = WIDTH, h = HEIGHT, r, g, b;
  FILE *f;

  unsigned char *img = NULL;
  //int yres = HEIGHT;
  int filesize = 54 + 3*w*h;  //w is your image width, h is image height, both int

  if( img )
      free( img );
  img = (unsigned char *)malloc(3*w*h);
  memset(img,0,sizeof(img));

  int i = 0;

  for(i=0; i<w; i++)
  {
    for(int j=0; j<h; j++)
    {
      x=i; y = j;
      r = (int)(image[i][j][0]*255);
      g = (int)(image[i][j][1]*255);
      b = (int)(image[i][j][2]*255);

      if (r > 255) r=255;
      if (g > 255) g=255;
      if (b > 255) b=255;

      img[(x+y*w)*3+2] = (unsigned char)(r);
      img[(x+y*w)*3+1] = (unsigned char)(g);
      img[(x+y*w)*3+0] = (unsigned char)(b);

    }
  }

  unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
  unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
  unsigned char bmppad[3] = {0,0,0};

  bmpfileheader[ 2] = (unsigned char)(filesize    );
  bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
  bmpfileheader[ 4] = (unsigned char)(filesize>>16);
  bmpfileheader[ 5] = (unsigned char)(filesize>>24);

  bmpinfoheader[ 4] = (unsigned char)(       w    );
  bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
  bmpinfoheader[ 6] = (unsigned char)(       w>>16);
  bmpinfoheader[ 7] = (unsigned char)(       w>>24);
  bmpinfoheader[ 8] = (unsigned char)(       h    );
  bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
  bmpinfoheader[10] = (unsigned char)(       h>>16);
  bmpinfoheader[11] = (unsigned char)(       h>>24);

  f = fopen("img.bmp","wb");
  fwrite(bmpfileheader,1,14,f);
  fwrite(bmpinfoheader,1,40,f);
  for(i=0; i<h; i++)
  {
      fwrite(img+(w*(h-i-1)*3),3,w,f);
      fwrite(bmppad,1,(4-(w*3)%4)%4,f);
  }
  fclose(f);
}

__forceinline__ __device__ float computeColor(float x, float y, float z, float diffuseColor) {
   //Color of light
   float lightPosX = 3;
   float lightPosY = 15.0;
   float lightPosZ = 13.0;

   float size = sqrt(lightPosX * lightPosX + lightPosY * lightPosY + lightPosZ + lightPosZ);
   lightPosX /= size;   
   lightPosY /= size;
   lightPosZ /= size;

   size = sqrt(x * x + y * y + z * z);
   x /= size;
   y /= size; 
   z /= size;

   float diffuseIntensity = 0.75;
   float diffuseLight = diffuseIntensity * (x * lightPosX + y * lightPosY + z * lightPosZ);

   float ambientLight = 0;
   float color = (diffuseLight + ambientLight) * diffuseColor;

   return color;
}



__global__ void Rasterizer(Tri *cudaTri, float *img, float *Depthbuffer, int triCounter)
{
     int i = blockIdx.x * 512 + threadIdx.x;

     if(i > triCounter)
	return;
   
     float xa = cudaTri[i].x1_2d;
     float xb = cudaTri[i].x2_2d;
     float xc = cudaTri[i].x3_2d;
				
     float ya = cudaTri[i].y1_2d;
     float yb = cudaTri[i].y2_2d; 
     float yc = cudaTri[i].y3_2d;

     for(int j = cudaTri[i].bboxLeft; 
          j <= cudaTri[i].bboxLeft + cudaTri[i].bboxWidth; 
          ++j) {
         for(int k = cudaTri[i].bboxTop; 
             k <= cudaTri[i].bboxTop + cudaTri[i].bboxHeight; 
             ++k) {  
            float x = (float)j;
            float y = (float)k;

            float beta = (((xa-xc) * (y - yc)) - ((x-xc) * (ya - yc))) / 
                         (((xb - xa) * (yc - ya)) - ((xc - xa) * (yb-ya)));
            float gamma = (((xb-xa) * (y - ya)) - ((x-xa) * (yb - ya))) / 
                          (((xb - xa) * (yc - ya)) - ((xc - xa) * (yb-ya)));
            float alpha = 1 - beta - gamma;

            if(alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1 && gamma >= 0 
                  && gamma <= 1) {
               float depthP = alpha * cudaTri[i].z1 + 
                              beta * cudaTri[i].z2 + 
                              gamma * cudaTri[i].z3;
               // 1 is camera z position
               float distancefromeye = 1 - depthP;

               if(distancefromeye <= Depthbuffer[(j * HEIGHT) + k]) {
                  float R = 
                        computeColor(cudaTri[i].x1, cudaTri[i].y1, cudaTri[i].z1, cudaTri[i].R1) * alpha + 
                        computeColor(cudaTri[i].x2, cudaTri[i].y2, cudaTri[i].z2, cudaTri[i].R2) * beta + 
                        computeColor(cudaTri[i].x3, cudaTri[i].y3, cudaTri[i].z3, cudaTri[i].R3) * gamma;

                  float G = 
                        computeColor(cudaTri[i].x1, cudaTri[i].y1, cudaTri[i].z1, cudaTri[i].G1) * alpha + 
                        computeColor(cudaTri[i].x2, cudaTri[i].y2, cudaTri[i].z2, cudaTri[i].G2) * beta + 
                        computeColor(cudaTri[i].x3, cudaTri[i].y3, cudaTri[i].z3, cudaTri[i].G3) * gamma;

                  float B = 
                        computeColor(cudaTri[i].x1, cudaTri[i].y1, cudaTri[i].z1, cudaTri[i].B1) * alpha + 
                        computeColor(cudaTri[i].x2, cudaTri[i].y2, cudaTri[i].z2, cudaTri[i].B2) * beta + 
                        computeColor(cudaTri[i].x3, cudaTri[i].y3, cudaTri[i].z3, cudaTri[i].B3) * gamma;

              for(int p = 0; p < 5; p++)
              {
                 for(int l = 0; l < 5; l++)
                 {
                    img[((j + (p*BUNNYWIDTH)) * HEIGHT) + (k + (l*BUNNYHEIGHT))] = R;
                    img[((j + (p*BUNNYWIDTH)) * HEIGHT) + (k + (l*BUNNYHEIGHT)) + (WIDTH * HEIGHT)] = G;
                    img[((j + (p*BUNNYWIDTH)) * HEIGHT) + (k + (l*BUNNYHEIGHT)) + 2 * (WIDTH * HEIGHT)] = B;
                 }
              }

                 
                  Depthbuffer[(j * HEIGHT) + k] = distancefromeye;
               } 
            }
         }
      }

  __syncthreads();    
}

float *img1;
int type = 0; // 0 for sequential, 1 for parralelized

int main (int argc, char **argv) {

  Parse(type);

  dim3 grid, block;

  grid.x = (TRIANGLES / 512) + (TRIANGLES % 512 ? 1 : 0);
  block.x = 512;
  
  img1 = (float*)malloc(sizeof(float) * WIDTH * HEIGHT * 3);

  int counter = TRIANGLES;
  
  float* zDepth;
  zDepth = (float*)malloc(sizeof(float) * WIDTH * HEIGHT);
   for(int i = 0; i < WIDTH; i++)
      for(int j = 0; j < HEIGHT; j++)
         zDepth[i * WIDTH + j] = 1000000;

  cudaMalloc(&cudaTri, sizeof(Tri) * TRIANGLES);
  cudaMalloc(&Depthbuffer, sizeof(float) * WIDTH * HEIGHT);
  cudaMalloc(&img, sizeof(float) * WIDTH * HEIGHT * 3);

  cudaMemcpy(cudaTri, Triangles, sizeof(Tri) * TRIANGLES, cudaMemcpyHostToDevice);
  cudaMemcpy(Depthbuffer, zDepth, sizeof(float) * WIDTH * HEIGHT , cudaMemcpyHostToDevice);
  
  Rasterizer<<<grid, block>>>(cudaTri, img, Depthbuffer, counter); 

  cudaMemcpy(img1, img, sizeof(float) * WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);

  cudaFree(cudaTri);
  cudaFree(Depthbuffer);
  cudaFree(img);

  if(type)
  {
     for(int i = 0; i < WIDTH; i++)
     {
        for(int j = 0; j < HEIGHT; j++)
        {
           image[i][j][0] = img1[i * WIDTH + j];
           image[i][j][1] = img1[i * WIDTH + j + 1 * (WIDTH * HEIGHT)];
           image[i][j][2] = img1[i * WIDTH + j + 2 * (WIDTH * HEIGHT)];
        }
     }
  }

  imageWrite();
  return 0;
}

