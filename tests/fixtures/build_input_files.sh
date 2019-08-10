#!/usr/bin/bash

x=1.0
y=2.0
z=3.0

force_version=1.0.0
force_num=1
force_gpu=1
force_cpu=1
force_steps=100
force_coordsystem=0
force_degree=3
force_file=force.bin

source_version=1.0.0
source_steps=10
source_num=1
source_gpu=1
source_cpu=1
source_coordsystem=0
source_degree=3
source_file=sources.bin

recv_version=1.0.0
recv_num=2
recv_stride=1
recv_gpu=100
recv_cpu=100
recv_coordsystem=0
recv_degree=3
recv_file=recv.bin

sgt_version=1.0.0
sgt_num=1
sgt_stride=1
sgt_gpu=100
sgt_cpu=100
sgt_coordsystem=0
sgt_degree=3
sgt_file=sgt.bin

# Source file
echo "$source_version
file=$source_file 
length=$source_num 
steps=$source_steps 
gpu_buffer_size=$source_cpu
cpu_buffer_size=$source_cpu
$source_coordsystem $source_degree
$x $y $z 
$x $y $z" \
> sources.in

# Force file
echo "$force_version
$force_file
$force_num $force_steps $force_gpu $force_cpu
$force_coordsystem $force_degree
$x $y" \
> forces.in

# Receiver file
echo "$recv_version
$recv_file
$recv_num $recv_stride $recv_gpu $recv_cpu
$recv_coordsystem $recv_degree
$x $y $z
$x $y $z" \
> recvs.in

# SGT output file
echo "$sgt_version
$sgt_file
$sgt_num $sgt_stride $sgt_gpu $sgt_cpu
$sgt_coordsystem $sgt_degree
$x $y $z
$x $y $z" \
> sgt.in
