NVCC=nvcc
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64
CUDA_INCLUDEPATH=/software/cuda-7.5-el6-x86_64/include/

histo: main.cu reference_calc.o student.o Makefile
	nvcc -o HW5 main.cu reference_calc.o student.o $(NVCC_OPTS)

student.o: student.cu
	nvcc -c student.cu $(NVCC_OPTS)

reference_calc.o: reference_calc.cpp reference_calc.h
	g++ -c reference_calc.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

clean:
	rm -f *.o hw *.bin
