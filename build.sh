#!/bin/bash

cvcc whiteBalance/video_sequential.cpp
cvcc whiteBalance/video_pthread_TDM.cpp -lpthread
cvcc whiteBalance/video_openmp_TDM.cpp -fopenmp
cvcc whiteBalance/task_parallel.cpp -fopenmp
