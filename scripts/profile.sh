#!/usr/bin/bash
opt=build/tests/test_optimized_kernels
unopt=build/tests/test_unoptimized_kernels
mkdir -p logs
nz=256
nt=100
for nx in 100 200 300 400 500 600
do
        for ny in 100 200 300 400 500 600
        do
                outputdir=logs/run_${nx}_${ny}_${nz}
                echo Running: nx = ${nx} ny = ${ny}
                $unopt --nx=$nx --ny=$ny --nz=$nz \
                       --px=1 --py=1 --output=${outputdir} --nt=${nt}
                nvprof -s --log-file ${outputdir}.log -o ${outputdir}.%p $opt \
                        --nx=$nx --ny=$ny --nz=$nz \
                        --px=1 --py=1 --input=${outputdir} --nt=${nt}
                # remove function signatures in log files
                sed -i -e 's/(.*)//g' ${outputdir}.log
        done
done
