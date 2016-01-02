#!/bin/bash

mv ../../whiteBalance/sequential/sequential.cpp ../../whiteBalance/sequential/sequential.cpp.old
cp whiteBalance/video_sequential.cpp ../../whiteBalance/sequential/sequential.cpp

mv ../../whiteBalance/pthread_TDM/pthread_TDM.cpp ../../whiteBalance/pthread_TDM/pthread_TDM.cpp.old
cp whiteBalance/video_pthread_TDM.cpp ../../whiteBalance/pthread_TDM/pthread_TDM.cpp

mv ../../whiteBalance/openmp_TDM/openmp_TDM.cpp ../../whiteBalance/openmp_TDM/openmp_TDM.cpp.old
cp whiteBalance/video_openmp_TDM.cpp ../../whiteBalance/openmp_TDM/openmp_TDM.cpp

mv ../../whiteBalance/task_parallel/task_parallel.cpp ../../whiteBalance/task_parallel/task_parallel.cpp.old
cp whiteBalance/task_parallel.cpp ../../whiteBalance/task_parallel/task_parallel.cpp
