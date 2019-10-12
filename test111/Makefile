all: c p

c:
	nvcc test111.cu -lcurand -arch=sm_61 -use_fast_math --ptxas-options=-v -o test111.x

p:
	nvprof --log-file prof.txt ./test111.x 512 512 512 20; sed -i -e 's/(.*)//g' prof.txt; head -n 6 prof.txt

