#include <cuda.h>
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

__global__ void imgProcessingKernel(unsigned char *originalImg,
                                    unsigned char *newImg, int width,
                                    int height) {}

int main() {
  const char *IMG_PATH = "images/shell-image.jpg";
  int imgWidth, imgHeight, imgPixelComponents;

  unsigned char *origImgData =
      stbi_load(IMG_PATH, &imgWidth, &imgHeight, &imgPixelComponents, 0);
  unsigned char *newImgData =
      (unsigned char *)malloc(sizeof(unsigned char) * imgWidth * imgHeight);

  printf("The image has a width of %d and a height of %d with %d components "
         "per pixel. The pixel value at (128,128) is %d\n",
         imgWidth, imgHeight, imgPixelComponents, origImgData[128]);

  dim3 block(16, 16);
  dim3 grid(imgWidth / 16, imgHeight / 16);
  imgProcessingKernel<<<Grid, Block>>>(origImgData, newImgData, imgWidth,
                                       imgHeight);

  return 0;
}