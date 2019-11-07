#!/bin/bash
# usage: launch_search.sh config.sh
# Perform brute force search for optimal launch configuration. 
# This script will produce one executable per launch configuration.
#
# The file config.sh should contain:
# out string        Output directory
# function string   Name of kernel function to vary launch configuration for

# config string     Macro name that determines the launch configuration. 
#                   If config=STRMU, then the threads in the source code should
#                   be set by STRMU_TX, STRMU_TY, STRMU_TZ, and the loops should
#                   be unrolled using the sizes STRMU_RX, STRMU_RY for the x and
#                   y directions, respectively

# skip_spill bool   Do not produce executables if any type of spillage occurs
#                   (stack frame, register reads and writes)

# max_threads int   Maximum number of threads to use
# min_threads int   Minimum number of threads to use
# max_unroll  int   Maximum loop unroll factor to use 

# threads_x ints    Array of threads for the CUDA X-dimension
# threads_y ints    Array of threads for the CUDA Y-dimension
# threads_z ints    Array of threads for the CUDA Z-dimension
#
# unroll_x ints     Array of loop unroll factors to try in the CUDA X-dimension 
# unroll_y ints     Array of loop unroll factors to try in the CUDA Y-dimension 
#
# arch string       CUDA Architecture to compile for, e.g., sm_70
if [ -z ${1} ]
then
        echo "No configuration specified. Using default configuration"
        out=default
        function=macro_unroll
        config="STRMU"
        
        skip_spill=1
        let max_threads=1024
        let min_threads=32
        max_unroll=8
        
        threads_x=( 64 32 16 )
        threads_y=( 8 4 2 1 )
        threads_z=( 8 4 2 1 )
        
        unroll_x=( 1 2 4 8 )
        unroll_y=( 1 2 4 8 )
        
        arch=sm_61
else
        source ${1}
fi

###############################################################################
mkdir -p $out
mkdir -p $out/log
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
                echo "Too many threads"
                continue;
        fi
        if (( ${mt} < ${min_threads} ))
        then
                echo "Too few threads"
                continue;
        fi
        
        for a in ${unroll_x[@]}
        do
                for b in ${unroll_y[@]}
                do
                        if (( $a*$b  > ${max_unroll} ))
                        then
                                echo "Maximum loop unrolling factor exceeded"
                                continue;
                        fi
                        echo "Compiling: <$tx, $ty, $tz, $a, $b>"
                        args="-D${config}_RX=$a -D${config}_RY=$b -D${config}_TX=$tx -D${config}_TY=$ty -D${config}_TZ=$tz"
                        exe="str_${tx}_${ty}_${tz}_${a}_${b}"
                        nvcc test111.cu -arch=${arch} -use_fast_math --ptxas-options=-v -o $out/$exe $args &> $out/log/$exe.txt
                        awk -v var="$function" 'c&&! --c; $0 ~ var {c=1}' $out/log/$exe.txt > tmp.txt
                        awk -v var="$function" 'c&&! --c; $0 ~ var {c=2}' $out/log/$exe.txt

                        # Check if spillage occurred
                        stack=`cat tmp.txt | grep -P "[1-9]+[0-9]* bytes stack" | wc -l`
                        spill=`cat tmp.txt | grep -P "[1-9]+[0-9]* bytes spill" | wc -l`
                        if (( $spill > 0 || $stack > 0 ))
                        then
                                if [[ $skip_spill == 1 ]]
                                then
                                        rm $out/$exe;
                                        rm $out/log/$exe.txt
                                        echo "Ignoring configuration due to" \
                                             "spillage."
                                fi
                        else
                                echo wrote: $out/$exe
                        fi

                done;
        done;
done;
done;
done;
#rm tmp.txt


