#!/bin/bash
project=str_no_align
threads_x=( 64 32 16 )
threads_y=( 8 4 2 1 )
threads_z=( 8 4 2 1 )
unroll_x=( 1 2 4 8 )
unroll_y=( 1 2 4 8 )
arch=sm_61
let max_threads=256
let min_threads=32
max_unroll=8
mkdir -p $project
mkdir -p $project/log
set -e
echo "key: <threads.x, threads.y, threads.z, unroll.x, unroll.y>"
for tx in ${threads_x[@]}
do
for ty in ${threads_y[@]}
do
for tz in ${threads_z[@]}
do
        let mt=$tx*$ty*$tz
        if (( ${mt} > ${max_threads} ))
        then
                continue;
        fi
        if (( ${mt} < ${min_threads} ))
        then
                continue;
        fi
        
        for a in ${unroll_x[@]}
        do
                for b in ${unroll_y[@]}
                do
                        if (( $a*$b  > ${max_unroll} ))
                        then
                                continue;
                        fi
                        echo "Compiling: <$tx, $ty, $tz, $a, $b>"
                        args="-DSTRMU_NA=$a -DSTRMU_NB=$b -DSTRMU_TX=$tx -DSTRMU_TY=$ty -DSTRMU_TZ=$tz"
                        exe="str_${tx}_${ty}_${tz}_${a}_${b}"
                        nvcc test111.cu -arch=${arch} -use_fast_math --ptxas-options=-v -o $project/$exe $args &> $project/log/$exe.txt &
                done;
        done;
done;
done;
done;


