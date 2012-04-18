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

using namespace std;
extern float image[WIDTH][HEIGHT][3];

GLfloat light_diffuse[] = {0.5, 0.5, 0.5, 1.0};  /* Red diffuse light. */
GLfloat light_position[] = {-0.2, 1.0, -1, 0.0};  /* Infinite light location. */

typedef struct Vector3
{
    float x;
    float y;
    float z;
    
    int R;
    int G;
    int B;

    Vector3(float in_x, float in_y, float in_z) : x(in_x), y(in_y), z(in_z) {}
    Vector3() {}
} Vector3;

typedef struct BoundingBox
{
  int top;
  int left;
  int width;
  int height;

  BoundingBox(int in_top, int in_left, int in_width, int in_height) : top(in_top), left(in_left), width(in_width), height(in_height)	 {}
  BoundingBox() {}

} BoundingBox;

typedef struct Vector2
{
    float x;
    float y;
    float z_depth;
    Vector2(float in_x, float in_y) : x(in_x), y(in_y) {}
    Vector2() {}
} Vector2;

typedef struct Triangle2d
{
    Vector2 v1;
    Vector2 v2;
    Vector2 v3;
    Triangle2d() {}
} Triangle2d;

//data structure to store triangle -
//note that v1, v2, and v3 are indexes into the vertex array
typedef struct Tri
{
    int v1;
    int v2;
    int v3;

    BoundingBox bbx;
    Triangle2d tri2d;  
  
    Vector3 normal;
    Vector3 color;
    Tri(int in_v1, int in_v2, int in_v3) : v1(in_v1), v2(in_v2), v3(in_v3), normal(0, 1, 0) {}
    Tri() : normal(0, 1, 0) {}
} Tri;

extern void Parse();
extern vector<Tri*> Triangles;
extern vector<Vector3 *> Vertices;
extern Vector3 center;

using namespace std;

double angle = 0;
typedef struct 
{
  int X;
  int Y; 
  int Z;
  double U;
  double V;
  bool isInside;
  float normalX;
  float normalY;
  float normalZ;
} VERTICES;

const double PI = 3.1415926535897;
const int space = 10;
const int VertexCount = (90 / space) * (360 / space) * 4;

VERTICES VERTEX[VertexCount];
VERTICES CUBEVERTEX[20000];
void displayCube(int);
GLuint LoadTextureRAW( const char * filename );
GLUquadricObj *quadric;

void displayShape()
{
  glColor3f(0.1,0.7,0.2); 

  for(int i = 0; i < Triangles.size(); i++)
  {
    glBegin(GL_TRIANGLES);    
    //top
    glColor3f(1.0f, 1.0f, 0.0f);
    glNormal3f(0.0, 1.0f, 0.0f);

    int v1 = Triangles[i]->v1 - 1;
    int v2 = Triangles[i]->v2 - 1;
    int v3 = Triangles[i]->v3 - 1;
    
    glColor3f(1.0f, 1.0f, 0.0f);
    //glColor3f(Vertices[v1]->z * 15, Vertices[v1]->z * 15, Vertices[v1]->z * 15);
    glVertex3f(Vertices[v1]->x, Vertices[v1]->y, Vertices[v1]->z);

    //glColor3f(Vertices[v2]->z * 15, Vertices[v2]->z * 15, Vertices[v2]->z * 15);
    glVertex3f(Vertices[v2]->x, Vertices[v2]->y, Vertices[v2]->z);

    //glColor3f(Vertices[v3]->z * 15, Vertices[v3]->z * 15, Vertices[v3]->z * 15);
    glVertex3f(Vertices[v3]->x, Vertices[v3]->y, Vertices[v3]->z);

    glEnd();
  }  
}

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

int main (int argc, char **argv) {
  Parse();
  imageWrite();
  return 0;
}

