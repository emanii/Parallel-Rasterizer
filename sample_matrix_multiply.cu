/**
 * CPE 458 Spring 2012
 * Lab 1C
 * Dr. Lupo
 * Mitchell Rosen and Tyler Saadus
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "mm.h"
#include "debug.h"

__global__ void MatrixMulKernel(float* prod, float* n1, float* n2, int width, 
      int height, int inner) {
   __shared__ float d_n1_s[TILE_WIDTH][TILE_WIDTH];
   __shared__ float d_n2_s[TILE_WIDTH][TILE_WIDTH];

   int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
   int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

   // If |overflow|, the number of rows/cols that compose this partial tile
   int overflow = inner % TILE_WIDTH;

   float prod_value = 0;

   for (int i = 0; i < inner / TILE_WIDTH + (overflow ? 1 : 0); i++) {
      if (overflow) {
         if (threadIdx.x < overflow && threadIdx.y < height % TILE_WIDTH) {
            d_n1_s[threadIdx.y][threadIdx.x] = n1[row * inner + (i * TILE_WIDTH + 
                  threadIdx.x)];
         }
         if (threadIdx.x < width % TILE_WIDTH && threadIdx.y < overflow) {
            d_n2_s[threadIdx.y][threadIdx.x] = n2[col + height * (i * TILE_WIDTH + 
                  threadIdx.y)];
         }
      }
      else {
         d_n1_s[threadIdx.y][threadIdx.x] = n1[row * inner + (i * TILE_WIDTH + 
               threadIdx.x)];
         d_n2_s[threadIdx.y][threadIdx.x] = n2[col + height * (i * TILE_WIDTH + 
               threadIdx.y)];
      }
      __syncthreads();

      if (row < height && col < width)
         for (int j = 0; j < (overflow ? overflow : TILE_WIDTH); j++)
            prod_value += d_n1_s[threadIdx.y][j] * d_n2_s[j][threadIdx.x];

      __syncthreads();
   }

   if (row < height && col < width)
      prod[row * width + col] = prod_value;
}

int main(int argc, char** argv) {
   dim3 gridDim, blockDim;

   float *m1, *m2, *mprod;
   float *d_m1, *d_m2, *d_mprod;
   
   int m1rows, m1cols, m2rows, m2cols;

   check_args(argc, argv);
   
   m1 = get_matrix_from_file(argv[1], &m1rows, &m1cols);
   m2 = get_matrix_from_file(argv[2], &m2rows, &m2cols);
   if (NULL == (mprod = (float*) malloc(m1rows * m1rows * sizeof(float)))) {
      fprintf(stderr, "Malloc failed\n");
      exit(EXIT_FAILURE);
   }

   //print_matrix(m1, m1rows, m1cols);
   //print_matrix(m2, m2rows, m2cols);
   
   if (m1cols != m2rows) {
      fprintf(stderr, "Can't multiply %dx%d by %dx%d\n", m1rows, m1cols,
            m2rows, m2cols);
      exit(EXIT_FAILURE);
   }

   // allocate memory on device
   cudaMalloc((void**) &d_m1, m1rows * m1cols * sizeof(float));
   cudaMalloc((void**) &d_m2, m2rows * m2cols * sizeof(float));
   cudaMalloc((void**) &d_mprod, m1rows * m2cols * sizeof(float));

   cudaMemcpy(d_m1, m1, m1rows * m1cols * sizeof(float), 
         cudaMemcpyHostToDevice); 
   cudaMemcpy(d_m2, m2, m2rows * m2cols * sizeof(float), 
         cudaMemcpyHostToDevice); 

   // set up dimensions
   gridDim.x = m2cols / TILE_WIDTH + (m2cols % TILE_WIDTH ? 1 : 0); 
   gridDim.y = m1rows / TILE_WIDTH + (m1rows % TILE_WIDTH ? 1 : 0); 

   blockDim.x = TILE_WIDTH;
   blockDim.y = TILE_WIDTH;

   // matrix multiply
   MatrixMulKernel<<<gridDim, blockDim>>>(d_mprod, d_m1, d_m2, m1rows, m2cols,
         m1cols);

   // copy back
   cudaMemcpy(mprod, d_mprod, m1rows * m2cols * sizeof(float), 
         cudaMemcpyDeviceToHost);

   // print results
   //fprintf(stdout, "Product:\n");
   //print_matrix(mprod, m1rows, m2cols);

   // write matrix to file
   write_matrix_to_file(OUTFILE, mprod, m1rows, m2cols);

   // clean up
   free(m1);
   free(m2);
   free(mprod);
   cudaFree(d_m1);
   cudaFree(d_m2);
   cudaFree(d_mprod);
   
   return EXIT_SUCCESS;
}

void check_args(int argc, char** argv) {
   if (argc != 3) {
      fprintf(stderr, "Usage: %s matrix-one-file matrix-two-file\n",
         argv[0]);
      exit(EXIT_FAILURE);
   }
}

float* get_matrix_from_file(char* fname, int* rows, int* cols) {
   FILE* infile;
   float* matrix;

   if (NULL == (infile = fopen(fname, "r"))) {
      perror("fopen");
      exit(EXIT_FAILURE);
   }
   
   get_size(infile, rows, cols);

   if (NULL == (matrix = (float*) malloc(*rows * *cols * sizeof(float)))) {
      fprintf(stderr, "Malloc failed\n");
      exit(EXIT_FAILURE);
   }

   fclose(infile);
   if (NULL == (infile = fopen(fname, "r"))) {
      perror("fopen");
      exit(EXIT_FAILURE);
   }

   for (int i = 0; i < *rows * *cols; i++)
      fscanf(infile, "%f ", matrix + i);

   return matrix;
}

void get_size(FILE* file, int* rows, int* cols) {
   char buf[BUF_SIZE];
   int m_rows = 0, m_nums = 0;
   int rlen;
   STATE state = READING_WHITESPACE;

   while ((rlen = fread(buf, sizeof(char), BUF_SIZE, file))) {
      FATALCALL(rlen, "fread");   
      for (int i = 0; i < rlen; i++) {
         switch (state) {
            case READING_NUM:
               if (buf[i] == ' ') {
                  state = READING_WHITESPACE;
                  m_nums++;
               }
               if (buf[i] == '\n') {
                  state = READING_WHITESPACE;
                  m_rows++;
                  m_nums++;
               }
               break;
            case READING_WHITESPACE:
               if (buf[i] == '\n') {
                  m_rows++;
               }
               else if (buf[i] != ' ') {
                  state = READING_NUM;
               }
               break;
            default:
               fprintf(stderr, "Switch shouldn't get here\n");
               exit(EXIT_FAILURE);
         }
      }
   }

   *rows = m_rows;
   *cols = m_nums / m_rows;
}

void write_matrix_to_file(char* fname, float* matrix, int rows, int cols) {
   int outfd;
   char buf[BUF_SIZE];
   char* p = buf;

   int wlen;

   FATALCALL((outfd = open(fname, O_CREAT | O_WRONLY | O_TRUNC, S_IRUSR | S_IWUSR)), "open");
   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         // if there is room in the buffer to fit another number, put it in
         if (p - buf <= MAX_FLOAT_CHARS + 2) { // space, null
            wlen = snprintf(p, MAX_FLOAT_CHARS + 2, "%.2f ", matrix[i * cols + j]);
            p += wlen;
         } 
         
         // otherwise, write the whole buffer and clear it out
         else {
            FATALCALL(write(outfd, buf, p - buf), "write"); 
            p = buf;

            wlen = snprintf(p, MAX_FLOAT_CHARS + 2, "%.2f ", matrix[i * cols + j]);
            p += wlen;
         }
      }

      *p++ = '\n';
   }
   
   if (p - buf) {
      FATALCALL(write(outfd, buf, p - buf), "write");
   }
}

void print_matrix(float* matrix, int rows, int cols) {
   int i, j, k = 0;
   
   printf("\n");
   for (i = 0; i < rows; i++) {
      for (j = 0; j < cols; j++) { 
         printf("%f ", matrix[k++]);
      }
      printf("\n");
   }
   printf("\n");
}
