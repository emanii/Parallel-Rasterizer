#include "MeshParser_release3.h"

//open the file for reading
void ReadFile(char* filename)
{

    printf("Reading coordinates from %s\n", filename);

    ifstream in_f(filename);
    if (!in_f)
    {
        printf("Could not open file %s\n", filename);
    }
    else
    {
        readStream(in_f);
    }

}

//process the input stream from the file
void readStream(istream& is)
{
    char str[256];
    for (; is;)
    {
        is >> ws;
        is.get(str,sizeof(str));
        if (!is) break;
        is.ignore(9999,'\n');
        readLine(str);
    }
}

//process each line of input save vertices and faces appropriately
int limited_rand(int limit)
{
  int r, d = RAND_MAX / limit;
  limit *= d;
  do { r = rand(); } while (r >= limit);
  return r / d;
}

void readLine(char* str)
{

    int indx = 0, vi;
    float x, y, z;
    float r, g, b;
    int mat;

    if (str[0]=='#') return;
    //read a vertex or face
    if (str[0]=='V' && !strncmp(str,"Vertex ",7))
    {
        Vector3* v;
        if (sscanf(str,"Vertex %d %g %g %g",&vi,&x,&y,&z) !=4)
        {
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
        //store the vertex
        Vertices.push_back(v);
        //house keeping to display in center of the scene
        center.x += v->x;
        center.y += v->y;
        center.z += v->z;
        if (v->x > max_x) max_x = v->x;
        if (v->x < min_x) min_x = v->x;
        if (v->y > max_y) max_y = v->y;
        if (v->y < min_y) min_y = v->y;
        if (v->z > max_z) max_z = v->z;
        if (v->z < min_z) min_z = v->z;
    }
    else if (str[0]=='F' && !strncmp(str,"Face ",5))
    {
        Tri* t;
        t = new Tri();
        char* s=str+4;
        int fi=-1;
        for (int t_i = 0;; t_i++)
        {
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
            if (fi<0)
            {
                fi=j;
                continue;
            }
            //otherwise process the digit we've grabbed in j as a vertex index
            //the first number will be the face id the following are vertex ids
            if (t_i == 1)
                t->v1 = j;
            else if (t_i == 2)
                t->v2 = j;
            else if (t_i == 3)
                t->v3 = j;
            //if there is more data to process break out
            if (*s =='{') break;
        }
        //possibly process colors if the mesh has colors
        if (*s && *s =='{')
        {
            char *s1 = s+1;
            cout << "trying to parse color " << !strncmp(s1,"rgb",3) << endl;
            //if we're reading off a color
            if (!strncmp(s1,"rgb=",4))
            {
                //grab the values of the string
                if (sscanf(s1,"rgb=(%g %g %g) matid=%d",&r,&g,&b,&mat)!=4)
                {
                    printf("error during reading rgb values\n");
#ifdef _DEBUG
                    exit(EXIT_FAILURE);
#endif
                }
                t->color.x = r;
                t->color.x = g;
                t->color.x = b;
                cout << "set color to: " << r << " " << g << " " << b << endl;
            }
        }
        //store the triangle read in
        Triangles.push_back(t);
    }
}

Vector2 convertPointTo2d(Vector3* vertex, Vector3* camera)
{

   float x = vertex->x;
   float y = vertex->y;
   float z = vertex->z;

   x = x - camera->x;
   y = y - camera->y;
   z = z - camera->z;

   float x_screen = WIDTH/2 + (WIDTH/2) * x;
   float y_screen = HEIGHT/2 - (HEIGHT/2) * y;


/*
   float half_screen_width = WIDTH / 2;
   float half_screen_height = HEIGHT / 2;


    //printf("%f %f %f\n", x, y,z);
   // x_screen *= WIDTH;//(+(x / z) + half_screen_width);
   // y_screen *= HEIGHT;//(-(y / z) + half_screen_height);
    printf("%f %f\n", x_screen, y_screen);

  float window_aspect = WIDTH / HEIGHT;

  if (window_aspect > 1.0)
  {
    x_screen = x_screen / window_aspect;
  }
  else
  {
    y_screen = y_screen * window_aspect;
  }
 
/*
   float x = vertex->x;
   float y = vertex->y;
   float z = vertex->z;

   float camX = camera->x; 
   float camY = camera->y;
   float camZ = camera->z;
   float camFieldofView = 60;

   float aspectRatio = WIDTH/HEIGHT;
   
   float inputX = x - camX;
   float inputY = y - camY;
   float inputZ = z - camZ;
  
   float scrX = inputX * (-inputZ * tan(camFieldofView/2));
   float scrY = (inputY * aspectRatio) / (-inputZ * tan(camFieldofView/2));
  
   scrX = scrX * WIDTH;
   scrY = (1-scrY) * HEIGHT;

    // Additional, currently unused, projection scaling factors
    /*
    double xScale = 1 / Math.Tan(Math.PI * cam.FieldOfView / 360);
    double yScale = aspectRatio * xScale;

    double zFar = cam.FarPlaneDistance;
    double zNear = cam.NearPlaneDistance;

    double zScale = zFar == Double.PositiveInfinity ? -1 : zFar / (zNear - zFar);
    double zOffset = zNear * zScale;

    */

   return (Vector2(x_screen, y_screen));
}

//testing routine for parser - left in just as informative code about accessing data
void printFirstThree()
{
    printf("first vertex: %f %f %f \n", Vertices[0]->x, Vertices[0]->y, Vertices[0]->z);
    printf("first face: %d %d %d \n", Triangles[0]->v1, Triangles[0]->v2, Triangles[0]->v3);
    printf("second vertex: %f %f %f \n", Vertices[1]->x, Vertices[1]->y, Vertices[1]);
    printf("second face: %d %d %d \n", Triangles[1]->v1, Triangles[1]->v2, Triangles[1]->v3);
    printf("third vertex: %f %f %f \n", Vertices[2]->x, Vertices[2]->y, Vertices[2]->z);
    printf("third face: %d %d %d \n", Triangles[2]->v1, Triangles[2]->v2, Triangles[2]->v3);
}

void calculateBoundingBox()
{
   for(int i = 0; i < Triangles.size(); i++)
   {
      int vert1 = Triangles[i]->v1;
      int vert2 = Triangles[i]->v2;
      int vert3 = Triangles[i]->v3;   

      Vector2 vect1 = convertPointTo2d(Vertices[vert1 - 1], new Vector3(-0.2,1,0));
      Vector2 vect2 = convertPointTo2d(Vertices[vert2 - 1], new Vector3(-0.2,1,0));
      Vector2 vect3 = convertPointTo2d(Vertices[vert3 - 1], new Vector3(-0.2,1,0));

      Triangles[i]->tri2d.v1 = vect1;
      Triangles[i]->tri2d.v1.z_depth = Vertices[vert1 - 1]->z;

      Triangles[i]->tri2d.v2 = vect2;
      Triangles[i]->tri2d.v2.z_depth = Vertices[vert2 - 1]->z;

      Triangles[i]->tri2d.v3 = vect3;
      Triangles[i]->tri2d.v3.z_depth = Vertices[vert3 - 1]->z;

      float min_x = (vect1.x < vect2.x ? vect1.x : vect2.x);
      min_x = (min_x < vect3.x ? min_x : vect3.x);

      float max_x = (vect1.x > vect2.x ? vect1.x : vect2.x);
      max_x = (max_x > vect3.x ? max_x : vect3.x);

      float min_y = (vect1.y < vect2.y ? vect1.y : vect2.y);
      min_y = (min_y < vect3.y ? min_y : vect3.y);

      float max_y = (vect1.y > vect2.y ? vect1.y : vect2.y);
      max_y = (max_y > vect3.y ? max_y : vect3.y);

      Triangles[i]->bbx.left = (int)min_x;
      Triangles[i]->bbx.top = (int)min_y;
      Triangles[i]->bbx.width = (int)(max_x - min_x);
      Triangles[i]->bbx.height = (int)(max_y - min_y);
   }
}

Vector3 normalize(Vector3* vt)
{
  float sizeit = vt->x * vt->x + vt->y * vt->y + vt->y * vt->y;
  sizeit = sqrt(sizeit);
  return (Vector3(vt->x / sizeit, vt->y/sizeit, vt->z/sizeit));
}

float dotProduct(Vector3 v1, Vector3 v2)
{
  return (v1.x * v2.x + v1.y * v2.y + v1.z*v2.z);
}

float computeColor(Vector3 vt, float diffuseColor)
{
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


void calculateBarycentric(int xAlign, int yAlign)
{
    for(int i = 0; i < Triangles.size(); i++)
    {
       float min = 11100000;
       float max = 0;
       //printf("Bounding: %f %f %f %f\n", Triangles[i]->bbx.top, Triangles[i]->bbx.left, Triangles[i]->bbx.width, Triangles[i]->bbx.height);
       float xa = Triangles[i]->tri2d.v1.x;
       float xb = Triangles[i]->tri2d.v2.x;
       float xc = Triangles[i]->tri2d.v3.x;

       float ya = Triangles[i]->tri2d.v1.y;
       float yb = Triangles[i]->tri2d.v2.y;
       float yc = Triangles[i]->tri2d.v3.y;

       int v1 = Triangles[i]->v1 - 1;
       int v2 = Triangles[i]->v2 - 1;
       int v3 = Triangles[i]->v3 - 1;

       for(int j = Triangles[i]->bbx.left; j <= Triangles[i]->bbx.left + Triangles[i]->bbx.width; j++)
       {
          for(int k = Triangles[i]->bbx.top; k <= Triangles[i]->bbx.top + Triangles[i]->bbx.height; k++)
          {  
             float x = (float)j;
             float y = (float)k;

	     float beta = (((xa-xc) * (y - yc)) - ((x-xc) * (ya - yc))) / (((xb - xa) * (yc - ya)) - ((xc - xa) * (yb-ya)));
	     float gamma = (((xb-xa) * (y - ya)) - ((x-xa) * (yb - ya))) / (((xb - xa) * (yc - ya)) - ((xc - xa) * (yb-ya)));
             float alpha = 1 - beta - gamma;

             if(alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1 && gamma >= 0 && gamma <= 1)
             {
		 float depthP = alpha * Triangles[i]->tri2d.v1.z_depth + beta * Triangles[i]->tri2d.v2.z_depth + gamma * Triangles[i]->tri2d.v3.z_depth;
                 // 1 is camera z position
                 float distancefromeye = 1 - depthP;
                 if(distancefromeye <= z_buffer[j][k])
                 {
 		   Vector2 vt = convertPointTo2d(new Vector3(j, k, depthP), new Vector3(0,0,0));

                   Vector3 point = Vector3(j, k, depthP);


                   float R = computeColor(*Vertices[v1], Vertices[v1]->R) * alpha + computeColor(*Vertices[v2], Vertices[v2]->R)  * beta + computeColor(*Vertices[v3], Vertices[v3]->R)  * gamma;

                   float G = computeColor(*Vertices[v1], Vertices[v1]->G) * alpha + computeColor(*Vertices[v2], Vertices[v2]->G)  * beta + computeColor(*Vertices[v3], Vertices[v3]->G)  * gamma;

                   float B = computeColor(*Vertices[v1], Vertices[v1]->B) * alpha + computeColor(*Vertices[v2], Vertices[v2]->B)  * beta + computeColor(*Vertices[v3], Vertices[v3]->B)  * gamma;

    		   image[j + (xAlign*WIDTH)][k + (yAlign*HEIGHT)][0] = R;
    		   image[j + (xAlign*WIDTH)][k + (yAlign*HEIGHT)][1] = G;
    		   image[j + (xAlign*WIDTH)][k + (yAlign*HEIGHT)][2] = B;
                   z_buffer[j][k] = distancefromeye;
                 }
             }
        }
      }
    }
}

//int main( int argc, char** argv )

void Parse()
{
    //initialization
    max_x = max_y = max_z = FLT_MIN;
    min_x = min_y = min_z = FLT_MAX;
    center.x = 0;
    center.y = 0;
    center.z = 0;
    display_mode = 0;
    max_extent = 1.0;
    view_mode = 0;

    char str[] = "bunny10k.m";

        ReadFile(str);
        //once the file is parsed find out the maximum extent to center and scale mesh
        max_extent = max_x - min_x;
        if (max_y - min_y > max_extent) max_extent = max_y - min_y;

        center.x = center.x/Vertices.size();
        center.y = center.y/Vertices.size();
        center.z = center.z/Vertices.size();

	printf("Initializing the Z-Buffer\n");
        for(int i = 0; i < WIDTH; i++)
        { 
           for(int j = 0; j < HEIGHT; j++)
           {
              z_buffer[i][j] = 10000000;
           }
        }

	printf("Caclulating Bounding Box\n");
        calculateBoundingBox();
   
	printf("Rasterization\n");
 	for(int i = 0; i < 5; i++)
        {
           for(int j = 0; j < 5; j++)
           {
	        calculateBarycentric(i,j);
           }
        }

}
//}

