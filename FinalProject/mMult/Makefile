NAME=microBench

build: matLib.o $(NAME).o
	#nvcc -o $(NAME) $(NAME).o matLib.o
	nvcc --device-debug --source-in-ptx -keep -o $(NAME) $(NAME).o matLib.o

$(NAME).o:
	nvcc -c $(NAME).cu -o $(NAME).o

matLib.o:
	nvcc -c matLib.cu -o matLib.o

clean:
	-rm $(NAME) $(NAME)_dlink* *.o
