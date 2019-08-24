#!/usr/bin/bash
opt=release/tests/test_optimized_kernels
unopt=release/tests/test_unoptimized_kernels
project=profile/mapping
mkdir -p ${project}
#nvflags=--analysis-metrics
nz=128
nt=1
nx=120
ny=140
outputdir=${project}/run_${nx}_${ny}_${nz}
echo Running: nx = ${nx} ny = ${ny}
#$unopt --nx=$nx --ny=$ny --nz=$nz \
#       --px=1 --py=1 --output=${outputdir} --nt=${nt}
$opt \
        --nx=$nx --ny=$ny --nz=$nz \
        --px=1 --py=1 --input=${outputdir} --nt=${nt}
