#!/bin/bash    
dir=logs/mapping
submit=analysis
nvflags="--profile-from-start off --analysis-metrics"
nz=256

git fetch && git fetch --tags
git checkout 1.0.1
mkdir -p release
cd release; cmake -DCMAKE_BUILD_TYPE=Release ..; make clean; make; cd -

mkdir -p ${dir}

for nx in 100 200 300 400 500 600
do
        for ny in 100 200 300 400 500 600
        do
                filename=${submit}_${nx}_${ny}.lsf
echo "#!/bin/bash
# Begin LSF Directives
#BSUB -P geo112
#BSUB -W 0:20
#BSUB -nnodes 1
#BSUB -alloc_flags gpumps
#BSUB -J profile
#BSUB -o ${dir}/profile_${nx}_${ny}.out
#BSUB -e ${dir}/profile_${nx}_${ny}.err
                
module load cuda

#cd $LS_SUBCWD
bash scripts/run.sh ${dir} \"${nvflags}\" ${nx} ${ny} ${nz}
" >    \
$dir/submit_${nx}_${ny}.lsf
        done

        #bsub $dir/submit_${nx}_${ny}.lsf
done
