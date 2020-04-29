#include <cuda.h>
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

__global__ void imgProcessingKernel(unsigned char *d_origImg,
                                    unsigned char *d_newImg) {
  // each thread will work on pixel value of the image, a pixel is represented
  // with 3 values R, G, and B
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i % 3 != 0) {
    return;
  }

  // gaussian blur kernel
  float blurKernel[3][3] = {{0.1111, 0.1111, 0.1111},
                            {0.1111, 0.1111, 0.1111},
                            {0.1111, 0.1111, 0.1111}};

  // convert 1d to 2d coords
  int x = i % 768;
  int y = i / 768;

  // ignore edges
  if (y != 0 && y != 255 && x != 0 && x != 765) {
    d_newImg[i] = d_origImg[i];         // R
    d_newImg[i + 1] = d_origImg[i + 1]; // G
    d_newImg[i + 2] = d_origImg[i + 2]; // B
  }
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

  // max amount of threads in a block is 1024
  dim3 threadsPerBlock(1024);
  // calculate the amount of blocks needed
  dim3 numBlocks(imgSize / 1024);

  // perform image processing
  imgProcessingKernel<<<numBlocks, threadsPerBlock>>>(d_origImg, d_newImg);
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