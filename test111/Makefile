nx=512
ny=512
nz=512
nt=100
arch=sm_70
log=prof.txt
all: c p


c:
	nvcc test111.cu -lcurand -arch=$(arch) -use_fast_math --ptxas-options=-v -o test111.x

p:
	nvprof --log-file $(log) ./test111.x $(nx) $(ny) $(nz) $(nt); 
	sed -i -e 's/(.*)//g' $(log); head -n 12 $(log)

dram:
	nvprof --log-file $(log) --metrics dram_read_throughput,dram_write_throughput ./test111.x $(nx) $(ny) $(nz) $(nt); 
	sed -i -e 's/(.*)//g' $(log); cat $(log)

l2:
	nvprof --log-file $(log) --metrics l2_read_throughput,l2_write_throughput ./test111.x $(nx) $(ny) $(nz) $(nt); 
	sed -i -e 's/(.*)//g' $(log); cat $(log)

