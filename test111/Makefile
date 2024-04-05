#!/bin/bash
nx=512
ny=512
nz=512
nt=10
arch=sm_75
log=prof.txt
args=-lineinfo
all: c p
exe=test111.x

ifeq ($(arch), sm_61) 
# Disable L1 caching on Pascal
extra_args=-Xptxas -dlcm=cg
endif

c:
	nvcc test111.cu -arch=$(arch) $(extra_args) $(debug) $(args) -use_fast_math --ptxas-options=-v -o $(exe) -D$(arch);

p:
	nvprof --log-file $(log) ./test111.x $(nx) $(ny) $(nz) $(nt); 
	sed -i -e 's/(.*)//g' $(log); head -n 12 $(log)

dram:
	nvprof --profile-from-start off --log-file $(log) --metrics dram_read_throughput,dram_write_throughput ./test111.x $(nx) $(ny) $(nz) $(nt); 

dram-all:
	nvprof --profile-from-start off --log-file $(log) \
	--metrics dram_read_throughput\
	,dram_write_throughput\
	,dram_read_transactions\
	,dram_write_transactions\
	,dram_read_throughput\
	,dram_write_throughput\
	,dram_utilization\
	,dram_read_bytes\
	,dram_write_bytes\
	 ./test111.x $(nx) $(ny) $(nz) $(nt); 
	sed -i -e 's/(.*)//g' $(log); cat $(log)

l2:
	nvprof --profile-from-start off --log-file $(log) --metrics l2_read_throughput,l2_write_throughput ./test111.x $(nx) $(ny) $(nz) $(nt); 
	sed -i -e 's/(.*)//g' $(log); cat $(log)

sed:
	sed -i -e 's/(.*)//g' $(log); cat $(log)

clean:
	rm test111.x

