//basic program to read in a mesh file (of .m format from H. Hoppe)
//Hh code modified by ZJW for csc 471

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string.h>
#include <assert.h>
#include <map>

using namespace std;

#define FLT_MIN 1.1754E-38F
#define FLT_MAX 1.1754E+38F
#define WIDTH 400
#define HEIGHT 400
#define SCALEFACTOR 13.122

//very simple data structure to store 3d points
typedef struct Vector3
{
    float x;
    float y;
    float z;
    
    float R;
    float G;
    float B;

    Vector3(float in_x, float in_y, float in_z) : x(in_x), y(in_y), z(in_z) {}
    Vector3() {}
} Vector3;

typedef struct BoundingBox
{
  float top;
  float left;
  float width;
  float height;

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

//stl vector to store all the triangles in the mesh
vector<Tri *> Triangles;
//stl vector to store all the vertices in the mesh
vector<Vector3 *> Vertices;

//for computing the center point and extent of the model
Vector3 center;
float max_x, max_y, max_z, min_x, min_y, min_z;
float max_extent;
float image[5 * WIDTH][5 * HEIGHT][3];
float z_buffer[WIDTH][HEIGHT];

//other globals
int GW;
int GH;
int display_mode;
int view_mode;

//forward declarations of functions
void readLine(char* str);
void readStream(istream& is);
void drawTri(Tri * t);
void drawObjects();
void display();

