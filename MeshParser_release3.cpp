#include "MeshParser_release3.h"

void calculateBoundingBox(Tri *Tiangles);

//open the file for reading
void ReadFile(char* filename)
{
    printf("Reading coordinates from %s\n", filename);

    ifstream in_f(filename);
    if (!in_f)
      printf("Could not open file %s\n", filename);
    else
      readStream(in_f);

}

//process the input stream from the file
int counter = 0;
void readStream(istream& is)
{
   char str[256];
   while (is) {
      is >> ws;
      is.get(str,sizeof(str));
      if (!is) break;
      is.ignore(9999,'\n');
      readLine(str);
   }
}

//process each line of input save vertices and faces appropriately
int limited_rand(int limit) {
   int r, d = RAND_MAX / limit;
   limit *= d;
   do { r = rand(); } while (r >= limit);
   return r / d;
}

void readLine(char* str) {
   int indx = 0, vi;
   float x, y, z;
   float r, g, b;
   int mat;


   if (str[0]=='#') return;
   //read a vertex or face
   if (str[0]=='V' && !strncmp(str,"Vertex ",7)) {
      Vector3* v;
      if (sscanf(str,"Vertex %d %g %g %g",&vi,&x,&y,&z) !=4) {
         printf("an error occurred in reading vertices\n");
#ifdef _DEBUG
         exit(EXIT_FAILURE);
#endif
      }
      v = new Vector3(x * SCALEFACTOR, y * SCALEFACTOR, z * SCALEFACTOR);
      /*v->R = ((float)limited_rand(255)) / 255;
        v->G = ((float)limited_rand(255)) / 255;
        v->B = ((float)limited_rand(255)) / 255;*/
      v->R = 0.5; 
      v->G = 0.5;
      v->B = 0.5;
      Vertices.push_back(v);

   }
   else if (str[0]=='F' && !strncmp(str,"Face ",5)) {
      char* s=str+4;
      int fi=-1;
      int v1, v2, v3;
      for (int t_i = 0;; t_i++) {
         while (*s && isspace(*s)) s++;
         //if we reach the end of the line break out of the loop
         if (!*s) break;
         //save the position of the current character
         char* beg=s;
         //advance to next space
         while (*s && isdigit(*s)) s++;
         //covert the character to an integer
         int j=atoi(beg);
         //the first number we encounter will be the face index, don't store it
         if (fi<0) {
            fi=j;
            continue;
         }
         //otherwise process the digit we've grabbed in j as a vertex index
         //the first number will be the face id the following are vertex ids
         if (t_i == 1)
            v1 = j;
         else if (t_i == 2)
            v2 = j;
         else if (t_i == 3)
            v3 = j;
         //if there is more data to process break out
         if (*s =='{') break;
      }
      //possibly process colors if the mesh has colors
      if (*s && *s =='{') {
         char *s1 = s+1;
         cout << "trying to parse color " << !strncmp(s1,"rgb",3) << endl;
         //if we're reading off a color
         if (!strncmp(s1,"rgb=",4)) {
            //grab the values of the string
            if (sscanf(s1,"rgb=(%g %g %g) matid=%d",&r,&g,&b,&mat)!=4) {
               printf("error during reading rgb values\n");
#ifdef _DEBUG
               exit(EXIT_FAILURE);
#endif
            }

            cout << "set color to: " << r << " " << g << " " << b << endl;
         }
      }
      //store the triangle read in
      Triangles[counter].x1 = Vertices[v1-1]->x;
      Triangles[counter].x2 = Vertices[v2-1]->x;
      Triangles[counter].x3 = Vertices[v3-1]->x;

      Triangles[counter].y1 = Vertices[v1-1]->y;
      Triangles[counter].y2 = Vertices[v2-1]->y;
      Triangles[counter].y3 = Vertices[v3-1]->y;

      Triangles[counter].z1 = Vertices[v1-1]->z;
      Triangles[counter].z2 = Vertices[v2-1]->z;
      Triangles[counter].z3 = Vertices[v3-1]->z;

      Vector3 AB = Vector3((Vertices)[v2 - 1]->x - (Vertices)[v1 - 1]->x, 
	 		 (Vertices)[v2 - 1]->y - (Vertices)[v1 - 1]->y, 
                         (Vertices)[v2 - 1]->z - (Vertices)[v1 - 1]->z);

      Vector3 BC = Vector3((Vertices)[v2 - 1]->x - (Vertices)[v3 - 1]->x, 
			 (Vertices)[v2 - 1]->y - (Vertices)[v3 - 1]->y, 
                         (Vertices)[v2 - 1]->z - (Vertices)[v3 - 1]->z);

      Vector3 normal = Vector3(AB.y * BC.z - AB.z * BC.y, AB.x * BC.z - AB.z * BC.x, AB.x * BC.y - AB.y * BC.x);

      Triangles[counter].normalX = normal.x;
      Triangles[counter].normalY = normal.y;
      Triangles[counter].normalZ = normal.z;

      Triangles[counter].R1 = Vertices[v1 - 1]->R;
      Triangles[counter].G1 = Vertices[v1 - 1]->G;
      Triangles[counter].B1 = Vertices[v1 - 1]->B;

      Triangles[counter].R2 = Vertices[v2 - 1]->R;
      Triangles[counter].G2 = Vertices[v2 - 1]->G;
      Triangles[counter].B2 = Vertices[v2 - 1]->B;

      Triangles[counter].R3 = Vertices[v3 - 1]->R;
      Triangles[counter].G3 = Vertices[v3 - 1]->G;
      Triangles[counter].B3 = Vertices[v3 - 1]->B;


      calculateBoundingBox(&Triangles[counter]);
      counter++;

   }
}

Vector3 convertPointTo2d(Vector3 vertex, Vector3 camera) {
   float x = vertex.x;
   float y = vertex.y;
   float z = vertex.z;

   x = x - camera.x;
   y = y - camera.y;
   z = z - camera.z;

   float x_screen = WIDTH/2 + (WIDTH/2) * x;
   float y_screen = HEIGHT/2 - (HEIGHT/2) * y;

   return (Vector3(x_screen, y_screen, z));
}

	
void calculateBoundingBox(Tri *Trian) {
      Vector3 vect1 = convertPointTo2d(Vector3((*Trian).x1, (*Trian).y1, (*Trian).z1), 
            Vector3(-0.2,1,0));
      Vector3 vect2 = convertPointTo2d(Vector3((*Trian).x2, (*Trian).y2, (*Trian).z2), 
            Vector3(-0.2,1,0));
      Vector3 vect3 = convertPointTo2d(Vector3((*Trian).x3, (*Trian).y3, (*Trian).z3), 
            Vector3(-0.2,1,0));

      (*Trian).x1_2d = vect1.x;
      (*Trian).y1_2d = vect1.y;

      (*Trian).x2_2d = vect2.x;
      (*Trian).y2_2d = vect2.y;

      (*Trian).x3_2d = vect3.x;
      (*Trian).y3_2d = vect3.y;


      float min_x = vect1.x < vect2.x ? vect1.x : vect2.x;
      min_x = min_x < vect3.x ? min_x : vect3.x;

      float max_x = vect1.x > vect2.x ? vect1.x : vect2.x;
      max_x = max_x > vect3.x ? max_x : vect3.x;

      float min_y = vect1.y < vect2.y ? vect1.y : vect2.y;
      min_y = min_y < vect3.y ? min_y : vect3.y;

      float max_y = vect1.y > vect2.y ? vect1.y : vect2.y;
      max_y = max_y > vect3.y ? max_y : vect3.y;

      (*Trian).bboxLeft = (int)min_x;
      (*Trian).bboxTop = (int)min_y;
      (*Trian).bboxWidth = (int)(max_x - min_x);
      (*Trian).bboxHeight = (int)(max_y - min_y);
}

Vector3 normalize(Vector3* vt) {
   float sizeit = vt->x * vt->x + vt->y * vt->y + vt->z * vt->z;
   sizeit = sqrt(sizeit);
   return (Vector3(vt->x / sizeit, vt->y/sizeit, vt->z/sizeit));
}

float dotProduct(Vector3 v1, Vector3 v2) {
   return (v1.x * v2.x + v1.y * v2.y + v1.z*v2.z);
}

float computeColor(Vector3 vt, float diffuseColor) {
   //Color of light
   Vector3 lightColor = Vector3(0.8, 0.8, 0.8);

   //Light position
   Vector3 uLight = Vector3(3, 15.0, 13.0);

   Vector3 normalVt = normalize(&vt);
   Vector3 normalLight = normalize(&uLight);

   float diffuseIntensity = 0.75;
   float diffuseLight = diffuseIntensity * dotProduct(normalVt, normalLight);

   float ambientLight = 0;
   float color = (diffuseLight + ambientLight) * diffuseColor;

   return color;
}

void calculateBarycentric(int xAlign, int yAlign) {
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
                  Vector3 vt = convertPointTo2d(Vector3(j, k, depthP), 
                                                Vector3(0,0,0));

                  Vector3 point = Vector3(j, k, depthP);


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

                  image[j + (xAlign*WIDTH)][k + (yAlign*HEIGHT)][0] = (R - 0.9) / 0.5;
                  image[j + (xAlign*WIDTH)][k + (yAlign*HEIGHT)][1] = (G - 0.9) / 0.5;
                  image[j + (xAlign*WIDTH)][k + (yAlign*HEIGHT)][2] = (B - 0.9) / 0.5;
                  z_buffer[j][k] = distancefromeye;
               } 
            } 
         } // inner for
      } // outer for
   } // for each triangle
}

void Parse(int type) {

   char str[] = "bunny10k.m";

   ReadFile(str);

   printf("Initializing the Z-Buffer\n");
   for(int i = 0; i < WIDTH; i++)
      for(int j = 0; j < HEIGHT; j++)
         z_buffer[i][j] = 10000000;

   if(type == 0)
   {
      printf("Rasterization\n");
      for(int i = 0; i < 5; i++)
         for(int j = 0; j < 5; j++)
            calculateBarycentric(i,j);
   }
}
