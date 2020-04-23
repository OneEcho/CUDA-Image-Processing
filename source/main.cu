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
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  d_newImg[row * width + col] = d_origImg[row * width + col];
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
  dim3 block(16, 16);
  dim3 grid(width / 16, height / 16);
  imgProcessingKernel<<<grid, block>>>(d_origImg, d_newImg, width, height);
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
      (unsigned char *)malloc(sizeof(unsigned char) * imgWidth * imgHeight);

  printf("The image has a width of %d and a height of %d with %d components "
         "per pixel. The pixel value at (128,128) is %d\n",
         imgWidth, imgHeight, imgPixelComponents, h_origImg[128]);

  // host function to start the image processing
  imgProcessing(h_origImg, h_newImg, imgWidth, imgHeight);

  printf("Orig img at (43, 65) is %d and New img at (43, 65) is %d",
         h_origImg[43 * imgWidth + 65], h_newImg[43 * imgWidth + 65]);

  stbi_write_jpg("images/shell-image-copy.jpg", imgWidth, imgHeight,
                 imgPixelComponents, h_newImg, 100);

  return 0;
}