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


using namespace std;
extern float image[WIDTH][HEIGHT][3];

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
extern void Parse();
extern Tri Triangles[TRIANGLES];

float *z_buffer;
__constant__ Tri Triangles[TRIANGLES];

void imageWrite()
{

  int x, y, w = WIDTH, h = HEIGHT, r, g, b;
  FILE *f;

  unsigned char *img = NULL;
  int yres = HEIGHT;
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

__global__ void Rasterizer()
{
for(int i = 0; i < TRIANGLES; i++) {
      float min = 11100000;
      float max = 0;

      float xa = Triangles[i].x1_2d;
      float xb = Triangles[i].x2_2d;
      float xc = Triangles[i].x3_2d;

      float ya = Triangles[i].y1_2d;
      float yb = Triangles[i].y2_2d; 
      float yc = Triangles[i].y3_2d;


      for(int j = Triangles[i].bboxLeft; 
          j <= Triangles[i].bboxLeft + Triangles[i].bboxWidth; 
          ++j) {
         for(int k = Triangles[i].bboxTop; 
             k <= Triangles[i].bboxTop + Triangles[i].bboxHeight; 
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
               float depthP = alpha * Triangles[i].z1 + 
                              beta * Triangles[i].z2 + 
                              gamma * Triangles[i].z3;
               // 1 is camera z position
               float distancefromeye = 1 - depthP;
               if(distancefromeye <= z_buffer[j][k]) {
                  float R = 
                        computeColor(Vector3(Triangles[i].x1, Triangles[i].y1, Triangles[i].z1) , Triangles[i].R1) * alpha + 
                        computeColor(Vector3(Triangles[i].x2, Triangles[i].y2, Triangles[i].z2) , Triangles[i].R2) * beta + 
                        computeColor(Vector3(Triangles[i].x3, Triangles[i].y3, Triangles[i].z3) , Triangles[i].R3) * gamma;

                  float G = 
                        computeColor(Vector3(Triangles[i].x1, Triangles[i].y1, Triangles[i].z1) , Triangles[i].G1) * alpha + 
                        computeColor(Vector3(Triangles[i].x2, Triangles[i].y2, Triangles[i].z2) , Triangles[i].G2) * beta + 
                        computeColor(Vector3(Triangles[i].x3, Triangles[i].y3, Triangles[i].z3) , Triangles[i].G3) * gamma;

                  float B = 
                        computeColor(Vector3(Triangles[i].x1, Triangles[i].y1, Triangles[i].z1) , Triangles[i].B1) * alpha + 
                        computeColor(Vector3(Triangles[i].x2, Triangles[i].y2, Triangles[i].z2) , Triangles[i].B2) * beta + 
                        computeColor(Vector3(Triangles[i].x3, Triangles[i].y3, Triangles[i].z3) , Triangles[i].B3) * gamma;

                  image[j + (xAlign*WIDTH)][k + (yAlign*HEIGHT)][0] = R;
                  image[j + (xAlign*WIDTH)][k + (yAlign*HEIGHT)][1] = G;
                  image[j + (xAlign*WIDTH)][k + (yAlign*HEIGHT)][2] = B;
                  z_buffer[j][k] = distancefromeye;
               } 
            } 
         } // inner for
      } // outer for
   } // for each triangle
}

int main (int argc, char **argv) {
  Parse();

  cudaMemcpyToSymbol(cudaTri, Triangles, sizeof(data));
  cudaMalloc(&z_buffer, TRIANGLES);
  Rasterizer<<<1000, 128>>>(d_A, d_B, d_C);
  cudaFree(z_buffer);
  
  imageWrite();
  return 0;
}

