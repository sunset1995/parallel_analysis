#!/bin/bash

if [ -z $1 ]; then
	echo "Please specify input video!"
	exit 1
fi

printf "* Sequential\n"
./video_sequential $1

printf "\n* Pthread\n"
./video_pthread_TDM $1 4

printf "\n* OpenMP\n"
./video_openmp_TDM $1 4

printf "\n* Task Parallel\n"
./task_parallel $1 4
