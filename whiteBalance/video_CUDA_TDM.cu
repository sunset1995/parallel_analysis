#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <ctime>
#include <thread>
#include <omp.h>

#define SHOW_INFO false
#define OUTPUT_VIDEO true
#define TD_MAX_SIZE 50
#define MAX_ROWS 1920
#define MAX_COLS 1080

using namespace std;
using namespace cv;

int threadNum;
Mat imgs[TD_MAX_SIZE];

VideoWriter setOutput(const VideoCapture &input) {
	// Reference from
	// http://docs.opencv.org/2.4/doc/tutorials/highgui/video-write/video-write.html

	// Acquire input size
	Size S = Size((int)input.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)input.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Get Codec Type- Int form
	int ex = static_cast<int>(input.get(CV_CAP_PROP_FOURCC));

	VideoWriter output;
	output.open("outputVideo.avi", CV_FOURCC('H', 'F', 'Y', 'U'), input.get(CV_CAP_PROP_FPS), S, true);

	return output;
}

__global__ void whiteBa(unsigned char *r, unsigned char *g, unsigned char *b, int tot, int ele) {
	
	int i, frameId = blockIdx.x;
	int from = ele * threadIdx.x;
	int to = from + ele < tot ? from + ele : tot;
	from += frameId * tot;
	to += frameId * tot;
	__shared__ int sum_r, sum_g, sum_b;
	__shared__ double avg_r, avg_g, avg_b;
	sum_r = sum_g = sum_b = 0;
	__syncthreads();
	for (i = from; i < to; i++) {
		atomicAdd(&sum_r, r[i]);
		atomicAdd(&sum_g, g[i]);
		atomicAdd(&sum_b, b[i]);
	}
	__syncthreads();
	avg_r = (double)sum_r / tot;
	avg_g = (double)sum_g / tot;
	avg_b = (double)sum_b / tot;
	double adj_r = avg_g / avg_r;
	double adj_b = avg_g / avg_b;
	for (i = from; i < to; i++) {
		r[i] = r[i] * adj_r < 255.0f ? r[i] * adj_r : 255;
		b[i] = b[i] * adj_b < 255.0f ? b[i] * adj_b : 255;
	}
}

void whiteBalance_CUDA(Mat imgs[], const int &sz) {
	
	int i, j, k, rows = imgs[0].rows, cols = imgs[0].cols;
	int totalElements = rows * cols;
	static unsigned char host_r[TD_MAX_SIZE * MAX_ROWS * MAX_COLS], host_g[TD_MAX_SIZE * MAX_ROWS * MAX_COLS], host_b[TD_MAX_SIZE * MAX_ROWS * MAX_COLS];
	unsigned char *device_r, *device_g, *device_b;
	device_r = device_g = device_b = NULL;

#pragma omp parallel for private(i, j)
	for (k = 0; k < sz; k++)
		for (i = 0; i < rows; i++)
			for (j = 0; j < cols; j++) {
				host_r[totalElements * k + cols * i + j] = imgs[k].at<Vec3b>(i, j)[2];
				host_g[totalElements * k + cols * i + j] = imgs[k].at<Vec3b>(i, j)[1];
				host_b[totalElements * k + cols * i + j] = imgs[k].at<Vec3b>(i, j)[0];
			}
	int size = sz * rows * cols * sizeof(unsigned char);
	cudaMalloc(&device_r, size);
	cudaMalloc(&device_g, size);
	cudaMalloc(&device_b, size);
	
	cudaMemcpy(device_r, host_r, size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_g, host_g, size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

	int blocksPerGrid = sz;
	int threadsPerBlock = 1024;
	int elementsPerThread = (rows * cols + threadsPerBlock - 1) / threadsPerBlock;

	whiteBa << <blocksPerGrid, threadsPerBlock >> >(device_r, device_g, device_b, totalElements, elementsPerThread);
	
	cudaMemcpy(host_r, device_r, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_g, device_g, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_b, device_b, size, cudaMemcpyDeviceToHost);

#pragma omp parallel for private(i, j)
	for (k = 0; k < sz; k++)
		for (i = 0; i < rows; i++)
			for (j = 0; j < cols; j++) {
				imgs[k].at<Vec3b>(i, j)[2] = host_r[totalElements * k + cols * i + j];
				imgs[k].at<Vec3b>(i, j)[0] = host_b[totalElements * k + cols * i + j];
			}

	cudaFree(device_r);
	cudaFree(device_g);
	cudaFree(device_b);
}

void inputVideo(const char *filePath, int rank, int sz, int fid) {
	int numPerThread = sz / threadNum;
	int from = rank * numPerThread;
	int to = (rank == threadNum - 1) ? sz : from + numPerThread;
	VideoCapture cpVideo(filePath);
	cpVideo.set(CV_CAP_PROP_POS_FRAMES, fid + from);

	for (int i = from; i<to; ++i)
		cpVideo >> imgs[i];
}

int main(int argc, const char** argv){
	if (CV_MAJOR_VERSION < 3) {
		puts("Advise you update to OpenCV3");
	}
	if (argc<2) {
		puts("Please specify input image path");
		return 0;
	}
	if (argc<3) {
		puts("Please specify thread num");
		return 0;
	}

	VideoCapture captureVideo;
	captureVideo.open(argv[1]);
	if (!captureVideo.isOpened()) {
		puts("Fail to open video");
		return 0;
	}

	// Setup video output
	VideoWriter outputVideo;
	if (OUTPUT_VIDEO)
		outputVideo = setOutput(captureVideo);
	
	threadNum = atoi(argv[2]);
	if (SHOW_INFO)
		printf("threads: %d\n", threadNum);
	outputVideo.set(CV_CAP_PROP_BUFFERSIZE, 1);

	double Calculate = 0, Input = 0, Output = 0;
	double Total = getTickCount(), Last;

	
	int numFrames = captureVideo.get(CV_CAP_PROP_FRAME_COUNT);
	for (int fid = 0; fid<numFrames; fid += TD_MAX_SIZE) {

		int sz = numFrames - fid;
		if (sz > TD_MAX_SIZE) sz = TD_MAX_SIZE;

		// store all thread
		vector<thread> threads;

		// input enough frames
		Last = getTickCount();
		for (int i = 0; i<threadNum; ++i)
			threads.emplace_back(thread(inputVideo, argv[1], i, sz, fid));
		for (int i = 0; i<threadNum; ++i)
			threads[i].join();
		threads.clear();
		Input += getTickCount() - Last;

		// proc all received frames
		Last = getTickCount();
		whiteBalance_CUDA(imgs, sz);
		Calculate += getTickCount() - Last;

		if (OUTPUT_VIDEO) {
			Last = getTickCount();
			for (int i = 0; i<sz; ++i)
				outputVideo << imgs[i];
			Output += getTickCount() - Last;
		}
	}

	Total = getTickCount() - Total;

	printf("    Total: %.3fs (include time count)\n", Total / getTickFrequency());
	printf("    Input: %.3fs\n", Input / getTickFrequency());
	printf("   Output: %.3fs\n", Output / getTickFrequency());
	printf("Calculate: %.3fs\n", Calculate / getTickFrequency());

	return 0;
}
