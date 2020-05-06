#include <cuda.h>
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

// https://gist.github.com/jefflarkin/5390993
// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                                       \
  {                                                                            \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(e));                                           \
      exit(0);                                                                 \
    }                                                                          \
  }

__global__ void imgProcessingKernel(unsigned char *d_origImg,
                                    unsigned char *d_newImg) {
  // each thread will work on pixel value of the image, a pixel is represented
  // with 3 values R, G, and B
  int col = (blockIdx.x * blockDim.x) + threadIdx.x;
  int row = (blockIdx.y * blockDim.y) + threadIdx.y;

  printf("col = %d, row = %d\n", col, row);

  if (row == 0 || row == 255) {
    return;
  }

  if (col == 0 || col == 767) {
    return;
  }

  // gaussian blur kernel
  int blurKernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

  // edge detection kernel
  int edgeDetectionKernel[3][3] = {{1, 0, -1}, {0, 0, 0}, {-1, 0, 1}};

  // emboss kernel
  int embossKernel[3][3] = {{-2, -2, 0}, {-2, 6, 0}, {0, 0, 0}};

  // matrix to hold neighbor values
  int mat[3][3];

  // calculate neighbor values and put them in a matrix
  mat[0][0] = d_origImg[(col - 3) + 768 * (row - 1)];
  mat[1][0] = d_origImg[col + 768 * (row - 1)];
  mat[2][0] = d_origImg[(col + 3) + 768 * (row - 1)];
  mat[0][1] = d_origImg[(col - 3) + 768 * row];
  mat[1][1] = d_origImg[col + 768 * row];
  mat[2][1] = d_origImg[(col + 3) + 768 * row];
  mat[0][2] = d_origImg[(col - 3) + 768 * (row + 1)];
  mat[1][2] = d_origImg[col + 768 * (row - 1)];
  mat[2][2] = d_origImg[(col + 3) + 768 * (row + 1)];

  int newRGBValue =
      ((mat[0][0] * embossKernel[0][0]) + (mat[1][0] * embossKernel[1][0]) +
       (mat[2][0] * embossKernel[2][0]) + (mat[0][1] * embossKernel[0][1]) +
       (mat[1][1] * embossKernel[1][1]) + (mat[2][1] * embossKernel[2][1]) +
       (mat[0][2] * embossKernel[0][2]) + (mat[1][2] * embossKernel[1][2]) +
       (mat[2][2] * embossKernel[2][2])) /
      2;

  d_newImg[col + 768 * row] = newRGBValue; // r, g, or b value is copied
}

__host__ void imgProcessing(unsigned char *h_origImg, unsigned char *h_newImg,
                            int imgSize) {
  unsigned char *d_origImg;
  unsigned char *d_newImg;

  // allocate memory for the original image, new image, and convolution kernel
  // on the device
  cudaMalloc((void **)&d_origImg, imgSize);
  cudaMalloc((void **)&d_newImg, imgSize);

  // copy the original image data from the host to the original image data
  // allocated on the device
  cudaMemcpy(d_origImg, h_origImg, imgSize, cudaMemcpyHostToDevice);

  // 8 x 8 is 64 threads per block
  dim3 threadsPerBlock(8, 8);
  // 96 x 32 blocks or 3,072 blocks
  dim3 numBlocks(768 / threadsPerBlock.x, 256 / threadsPerBlock.y);

  // perform image processing with 196,608 threads total, which is enough for a
  // 768 x 256 array
  imgProcessingKernel<<<numBlocks, threadsPerBlock>>>(d_origImg, d_newImg);
  cudaThreadSynchronize();
  cudaCheckError();

  // copy device image to host image
  cudaMemcpy(h_newImg, d_newImg, imgSize, cudaMemcpyDeviceToHost);

  cudaFree(d_newImg);
}

const char *IMG_PATH = "images/shell.jpg";

int main() {

  int imgWidth, imgHeight, imgChannels;

  // allocate memory for original image on host
  unsigned char *h_origImg =
      stbi_load(IMG_PATH, &imgWidth, &imgHeight, &imgChannels, 0);
  if (h_origImg == NULL) {
    printf("Error in loading the image\n");
    exit(1);
  }

  // calculate the image size
  unsigned long int imgSize = (imgWidth * imgHeight) * imgChannels;

  printf("Loaded an image with a width of %dpx, a height of %dpx and %d "
         "channels. The image size calculate is then imgChannels * imgHeight * "
         "imgChannels = %lu\n",
         imgWidth, imgHeight, imgChannels, imgSize);

  // allocate memory for the new image on host
  unsigned char *h_newImg = (unsigned char *)malloc(imgSize);

  // host function to start the image processing
  imgProcessing(h_origImg, h_newImg, imgSize);
  cudaDeviceSynchronize();

  // create the new image
  stbi_write_jpg("images/shell-copy.jpg", imgWidth, imgHeight, imgChannels,
                 h_newImg, 100);

  stbi_image_free(h_origImg);
  free(h_newImg);

  return 0;
}