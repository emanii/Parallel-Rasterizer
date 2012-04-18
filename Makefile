all:
	g++ openglit.c MeshParser_release3.cpp -DGL_GLEXT_PROTOTYPES -lglut -lGL -lGLU
mac:
	g++ openglit.c MeshParser_release3.cpp -DGL_GLEXT_PROTOTYPES -framework OpenGL -framework GLUT
else:
	g++ openglit.c MeshParser_release3.cpp -DGL_GLEXT_PROTOTYPES -lglut -lGL -lGLU
