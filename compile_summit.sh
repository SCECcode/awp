#!/bin/tcsh

rm -r release
mkdir -p release
module load cmake gcc cuda

cd release
cmake ..
make
