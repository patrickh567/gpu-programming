NAME=saxpyKernel

build: gpuLib.o $(NAME).o
	nvcc -o $(NAME) $(NAME).o gpuLib.o

$(NAME).o:
	nvcc -c $(NAME).cu -o $(NAME).o

gpuLib.o:
	nvcc -c gpuLib.cu -o gpuLib.o

clean:
	-rm $(NAME) *.o

clean-all:
	-rm saxpyKernel matAddKernel gridAddKernel printfKernel *.o
