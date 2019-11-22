out=macro_unroll
function=dtopo_str_111_macro_unroll
config="STRMU"

skip_spill=1
let max_threads=2048
let min_threads=32
max_unroll=8

threads_x=( 256 128 64 32 16 8 )
threads_y=( 16 8 4 2 1 )
threads_z=( 16 8 4 2 1 )
unroll_x=( 4 2 1 )
unroll_y=( 4 2 1 )

arch=sm_61

