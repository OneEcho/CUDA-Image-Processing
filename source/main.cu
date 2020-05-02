#include <cuda.h>
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

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
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // gaussian blur kernel
  float blurKernel[3][3] = {{0.1111, 0.1111, 0.1111},
                            {0.1111, 0.1111, 0.1111},
                            {0.1111, 0.1111, 0.1111}};

  int edgeDetection[3][3] = {{1, 0, -1}, {0, 0, 0}, {-1, 0, 1}};

  // matrix to hold neighbor values
  int mat[3][3];

  // ignore edges
  if (row != 0 && row != 255 && col != 0 && col != 767) {
    mat[0][0] = d_origImg[(col - 3) + 768 * (row - 1)];
    mat[1][0] = d_origImg[col + 768 * (row - 1)];
    mat[2][0] = d_origImg[(col + 3) + 768 * (row - 1)];
    mat[0][1] = d_origImg[(col - 3) + 768 * row];
    mat[1][1] = d_origImg[col + 768 * row];
    mat[2][1] = d_origImg[(col + 3) + 768 * row];
    mat[0][2] = d_origImg[(col - 3) + 768 * (row + 1)];
    mat[1][2] = d_origImg[col + 768 * (row - 1)];
    mat[2][2] = d_origImg[(col + 3) + 768 * (row + 1)];
  }

  int newRGBValue =
      (mat[0][0] * edgeDetection[0][0]) + (mat[1][0] * edgeDetection[1][0]) +
      (mat[2][0] * edgeDetection[2][0]) + (mat[0][1] * edgeDetection[0][1]) +
      (mat[1][1] * edgeDetection[1][1]) + (mat[2][1] * edgeDetection[2][1]) +
      (mat[0][2] * edgeDetection[0][2]) + (mat[1][2] * edgeDetection[1][2]) +
      (mat[2][2] * edgeDetection[2][2]);

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

  dim3 block(28, 28);
  dim3 grid(28, 28);

  // perform image processing
  imgProcessingKernel<<<grid, block>>>(d_origImg, d_newImg);
  cudaCheckError();
  cudaThreadSynchronize();

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