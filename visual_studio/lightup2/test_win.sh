#!/bin/bash

if [ -z $1 ]; then
	echo "Please specify input video!"
	exit 1
fi

# Default number of threads is 4
threads=4
if [ ! -z $2 ]; then
	threads=$2
fi

printf "* Sequential\n"
x64/Release/sequential.exe $1

printf "\n* Pthread\n"
x64/Release/pthread_TDM.exe $1 $threads

printf "\n* OpenMP\n"
x64/Release/openmp_TDM.exe $1 $threads

printf "\n* Task Parallel\n"
x64/Release/task_parallel.exe $1 $threads

printf "\n* CUDA\n"
x64/Release/cuda.exe $1 $threads

printf "\n* CUDA TDM\n"
x64/Release/cuda_TDM.exe $1 $threads
