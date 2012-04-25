NVFLAGS=-g -pg -arch=compute_20 -code=sm_20 -DGL_GLEXT_PROTOTYPES -lglut -lGL -lGLU
# list .c and .cu source files here
SRCFILES=openglit.cu MeshParser_release3.cpp 

all:	mm_cuda	

mm_cuda: $(SRCFILES) 
	nvcc $(NVFLAGS) -o mm_cuda $^

clean: 
	rm -f *.o mm_cuda
