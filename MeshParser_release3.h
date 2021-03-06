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
#define SCALEFACTOR 10
#define TRIANGLES 10000

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

//stl vector to store all the triangles in the mesh
Tri Triangles[TRIANGLES];
//stl vector to store all the vertices in the mesh
vector<Vector3 *> Vertices;

//for computing the center point and extent of the model
float image[5 * WIDTH][5 * HEIGHT][3];
float z_buffer[WIDTH][HEIGHT];


//forward declarations of functions
void readLine(char* str);
void readStream(istream& is);
void drawTri(Tri * t);
void drawObjects();
void display();

