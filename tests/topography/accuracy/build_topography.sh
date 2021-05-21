x=16
h=1.0
for i in 0 1 2 3;
do
python3 topography.py ../../../build/topography_$i.bin $x $x $h
let x=x*2;
h=`python3 -c "print(${h}/2)"`;
done
