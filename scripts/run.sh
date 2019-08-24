#!/usr/bin/bash
dir=$1
nvflags=$2
nx=$3
ny=$4
nz=$5
nt=100
opt=release/tests/test_optimized_kernels
unopt=release/tests/test_unoptimized_kernels

mkdir -p $dir
outputdir=$dir/run_${nx}_${ny}_${nz}

echo Running: nx = ${nx} ny = ${ny} nz = ${nz}
$unopt --nx=$nx --ny=$ny --nz=$nz \
       --px=1 --py=1 --output=${outputdir} --nt=${nt}
nvprof ${nvflags} --log-file ${outputdir}.log -o ${outputdir}.%p \
        $opt \
        --nx=$nx --ny=$ny --nz=$nz \
        --px=1 --py=1 --input=${outputdir} --nt=${nt}
# remove function signatures in log files
sed -i -e 's/(.*)//g' ${outputdir}.log
