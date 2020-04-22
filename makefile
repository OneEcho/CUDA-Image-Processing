# the compiler we are using
CC = nvcc

# compiler flags
CFLAGS = -O2 -L/usr/X11R6/lib -lm -lpthread -lX11

imageproc:
	$(CC) -o image_proc.exe source/main.cu $(CFLAGS) 