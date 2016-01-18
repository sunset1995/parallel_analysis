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

printf "\n--- lightup\n"
printf "* Sequential\n"
./lightup_sequential $1

printf "\n* Pthread\n"
./lightup_pthread $1 $threads

printf "\n* Pthread2\n"
./lightup_pthread_2 $1 $threads


printf "\n\n--- lightup2\n"
printf "* Sequential\n"
./lightup2_sequential $1

printf "\n* Pthread\n"
./lightup2_pthread $1 $threads

printf "\n* Pthread2\n"
./lightup2_pthread_2 $1 $threads