#include <cuda.h>
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

__global__ void imgProcessingKernel(unsigned char *d_origImg,
                                    unsigned char *d_newImg, int width,
                                    int height) {
  // calculates the unique pixel coordinate for each thread to work on (1 pixel
  // for 1 thread)
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  unsigned char *origPixel = &d_origImg[x * width + y];
  unsigned char *newPixel = &d_newImg[x * width + y];

  newPixel[0] = origPixel[0]; // r
  newPixel[1] = origPixel[1]; // g
  newPixel[2] = origPixel[2]; // b
}

__host__ void imgProcessing(unsigned char *h_origImg, unsigned char *h_newImg,
                            int width, int height) {
  unsigned char *d_origImg;
  unsigned char *d_newImg;

  // allocate memory for the original image and new image on the device
  cudaMalloc((void **)&d_origImg, sizeof(unsigned char) * width * height);
  cudaMalloc((void **)&d_newImg, sizeof(unsigned char) * width * height);

  // copy the original image data from the host to the original image data
  // allocated on the device
  cudaMemcpy(d_origImg, h_origImg, sizeof(unsigned char) * width * height,
             cudaMemcpyHostToDevice);

  // setup of the block and grid
  dim3 threadsPerBlock(8, 8);
  dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
  imgProcessingKernel<<<numBlocks, threadsPerBlock>>>(d_origImg, d_newImg,
                                                      width, height);
  cudaThreadSynchronize();
  cudaMemcpy(h_newImg, d_newImg, sizeof(unsigned char) * width * height,
             cudaMemcpyDeviceToHost);
}

int main() {
  const char *IMG_PATH = "images/shell-image.jpg";
  int imgWidth, imgHeight, imgPixelComponents;

  // allocate memory for original image on host
  unsigned char *h_origImg =
      stbi_load(IMG_PATH, &imgWidth, &imgHeight, &imgPixelComponents, 0);

  // allocate memory for the new image on host
  unsigned char *h_newImg =
      (unsigned char *)malloc(sizeof(unsigned char) * imgWidth * imgHeight * 3);

  // printf("The image has a width of %d and a height of %d with %d components "
  //        "per pixel. The pixel value at (128,128) is %d\n",
  //        imgWidth, imgHeight, imgPixelComponents, h_origImg[128]);
  int size = sizeof(h_origImg);
  printf("The length of the original image array is %d.\n", size);

  // host function to start the image processing
  imgProcessing(h_origImg, h_newImg, imgWidth, imgHeight);
  cudaDeviceSynchronize();

  stbi_write_jpg("images/shell-image-copy.jpg", imgWidth, imgHeight,
                 imgPixelComponents, h_newImg, 100);

  return 0;
}