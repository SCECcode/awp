#!/usr/bin/bash
opt=release/tests/test_optimized_kernels
unopt=release/tests/test_unoptimized_kernels
dir=logs_nvvp
mkdir -p 
nz=256
nt=10
for nx in 200 400 600
do
        for ny in 200 400 600
        do
                outputdir=$dir/run_${nx}_${ny}_${nz}
                echo Running: nx = ${nx} ny = ${ny}
                $unopt --nx=$nx --ny=$ny --nz=$nz \
                       --px=1 --py=1 --output=${outputdir} --nt=${nt}
                nvprof -s --log-file ${outputdir}.log -o ${outputdir}.%p \
                        --analysis-metrics \
                        --profile-from-start off \
                         $opt \
                        --nx=$nx --ny=$ny --nz=$nz \
                        --px=1 --py=1 --input=${outputdir} --nt=${nt}
                # remove function signatures in log files
                sed -i -e 's/(.*)//g' ${outputdir}.log
        done
done
