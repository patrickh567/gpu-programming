NAME=gridAddKernel

build:
	nvcc $(NAME).cu -o $(NAME)
