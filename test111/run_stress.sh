nx=300
ny=350
nz=512
nt=100
project=str_no_align

cd $project
for f in str*
do
        log=log/${f}_${nx}_${ny}_${nz}.nvprof
        echo $f
        nvprof --log-file $log ./$f $nx $ny $nz $nt; 
        sed -i -e 's/(.*)//g' $log; grep dtopo_str $log
        echo ""
done
