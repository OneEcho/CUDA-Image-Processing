# the compiler we are using
CC = nvcc

# compiler flags
CFLAGS = -O2 -L/usr/X11R6/lib -lm -lpthread -lX11 -g

imageproc:
	$(CC) -o imageproc.exe source/main.cu $(CFLAGS) 