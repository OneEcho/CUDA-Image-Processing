#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <cuda.h>
#include <iostream>
#include <string>

using namespace cv;

// // https://gist.github.com/jefflarkin/5390993
// // Macro for checking cuda errors following a cuda launch or api call
// #define cudaCheckError()                                                \
//     {                                                                    \
//       cudaError_t e = cudaGetLastError();                               \
//       if (e != cudaSuccess) {                                           \
//           printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,      \
//                  cudaGetErrorString(e));                                \
//           exit(0);                                                      \
//       }                                                                 \
//   }

__global__ void imgProcessingKernel(unsigned char *d_origImg,
                                    unsigned char *d_newImg) {
    // each thread will work on pixel value of the image, a pixel is represented
    // with 3 values R, G, and B
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (row == 0 || row == 255 || col == 0 || col == 767) {
        return;
    }

    // gaussian blur kernel
    int blurKernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

    // edge detection kernel
    int edgeDetectionKernel[3][3] = {{1, 0, -1}, {0, 0, 0}, {-1, 0, 1}};

    // emboss kernel
    int embossKernel[3][3] = {{-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}};

    // matrix to hold neighbor values
    int mat[3][3];

    // calculate neighbor values and put them in a matrix
    int colOffset = 3;
    int rowOffset = 1;

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            mat[i][j] = d_origImg[(col - colOffset) + 768 * (row - rowOffset)];
            colOffset -= 3;
        }
        // Go to next row
        rowOffset -= 1;

        // Reset colOffset after finishing a row
        colOffset = 3;
    }

    // Calculate new red, green, or blue value
    int newRGBValue = 0;

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            newRGBValue += mat[i][j] * edgeDetectionKernel[i][j];
        }
    }

    d_newImg[col + 768 * row] = newRGBValue; // r, g, or b value is copied
}

__host__ void imgProcessing(const Mat &h_origImg, const Mat &h_newImg) {
    unsigned char *d_origImg;
    unsigned char *d_newImg;
    size_t imageSize = (h_origImg.cols * h_origImg.channels()) * h_origImg.cols * sizeof(uchar);

    // allocate memory for the original image and new image
    cudaMalloc((void **)&d_origImg, imageSize);
    cudaMalloc((void **)&d_newImg, imageSize);

    // copy the original image data from the host to the original image data
    // allocated on the device
    cudaMemcpy(d_origImg, h_origImg.data, imageSize, cudaMemcpyHostToDevice);

    // 8 x 8 is 64 threads per block
    dim3 threadsPerBlock(8, 8);
    // 96 x 32 blocks or 3,072 blocks
    dim3 numBlocks((h_origImg.cols * h_origImg.channels()) / threadsPerBlock.x, h_origImg.rows / threadsPerBlock.y);

    // perform image processing with 196,608 threads total, which is enough for a
    // 768 x 256 array
    imgProcessingKernel<<<numBlocks, threadsPerBlock>>>(d_origImg, d_newImg);
    cudaThreadSynchronize();
    // cudaCheckError();

    // copy device image to host image
    cudaMemcpy(h_newImg.data, d_newImg, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_newImg);
}

int main(int argc, char ** argv) {

    std::string imagePath, newImagePath;

    if(argc < 3) {
        std::cout << "Not enough arguments provided. Please provide the path to a image and a path for the newly created image.\n";
        return 1;
    } else {
        imagePath = argv[1];
        newImagePath = argv[2];
    }

    // allocate memory for original image on host
    Mat h_origImg = imread(imagePath, IMREAD_COLOR);
    if(h_origImg.empty()) {
        std::cout << "Could not read the image: " << imagePath << std::endl;
        return 1;
    }

    // calculate the image size
    Size imageSize = h_origImg.size();
    int imgChannels = h_origImg.channels();
    int imgWidth = imageSize.width;
    int imgHeight = imageSize.height;

    std::cout << "Loaded an image with a width of " << imgWidth << " and a height of "
              << imgHeight << ". The image has " << imgChannels << " channels.\n";

    // create a new Mat for the processed image
    Mat h_newImg(imageSize, h_origImg.type());

    // host function to start the image processing
    imgProcessing(h_origImg, h_newImg);
    cudaDeviceSynchronize();

    // create the new image
    imwrite(newImagePath, h_newImg);

    return 0;
}
