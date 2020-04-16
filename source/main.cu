#include "include/CImg.h"
#include <cuda.h>
#include <stdio.h>

using namespace cimg_library;

int main() {
  CImg<unsigned char> img("images/lab_puppies.jpg");

  img.display("My first CImg code"); // Display the image in a display window.
  return 0;
}