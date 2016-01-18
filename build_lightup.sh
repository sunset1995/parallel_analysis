#!/bin/bash

cvcc lightup/lightup_sequential.cpp
cvcc lightup/lightup_pthread.cpp -lpthread
cvcc lightup/lightup_pthread_2.cpp -lpthread

cvcc lightup2/lightup2_sequential.cpp
cvcc lightup2/lightup2_pthread.cpp -lpthread
cvcc lightup2/lightup2_pthread_2.cpp -lpthread