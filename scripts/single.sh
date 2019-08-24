#!/usr/bin/bash
opt=release/tests/test_optimized_kernels
unopt=release/tests/test_unoptimized_kernels
project=profile/mapping
mkdir -p ${project}
#nvflags=--analysis-metrics
nz=256
nt=100
nx=400
ny=400
outputdir=${project}/run_${nx}_${ny}_${nz}
echo Running: nx = ${nx} ny = ${ny}
#time $unopt --nx=$nx --ny=$ny --nz=$nz \
#       --px=1 --py=1 --output=${outputdir} --nt=${nt}
#time $opt --nx=$nx --ny=$ny --nz=$nz \
#       --px=1 --py=1 --output=${outputdir} --nt=${nt}
nvprof -s --log-file ${outputdir}.log -o ${outputdir}.%p \
        ${nvflags}\
       --profile-from-start off \
$opt \
        --nx=$nx --ny=$ny --nz=$nz \
        --px=1 --py=1 --input=${outputdir} --nt=${nt}
# remove function signatures in log files
sed -i -e 's/(.*)//g' ${outputdir}.log
