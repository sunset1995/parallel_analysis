
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <algorithm>
#include <ctime>

using namespace std;
using namespace cv;



__global__ void blue(unsigned char *mat, int height, int width)
{
	int i = blockIdx.x * 32 + threadIdx.x
		, j = blockIdx.y * 32 + threadIdx.y;
	if (i < height && j < width)
		for (int k = 0; k < 100; k++)
			mat[i * width + j] = min(mat[i * width + j] * 2 + 5, 255);
}


void Video() {
	// Setup video capture device
	// Link it to the first capture device
	cudaError_t err;
	VideoCapture captureVideo;
	captureVideo.open("D:/videoLarge.mp4");

	int i, j;
	Mat frameFromVideo;
	clock_t cnt = 0, cnt_io = 0;

	unsigned char *hmat, *dmat = NULL;

	while (true){
		captureVideo >> frameFromVideo;
		if (frameFromVideo.empty()) break;
		imshow("origin", frameFromVideo);

		clock_t last = clock();
		
		clock_t last_io = clock();
		int rows = frameFromVideo.rows, cols = frameFromVideo.cols;
		int size = rows * cols * sizeof(unsigned char);
		hmat = (unsigned char *)malloc(size);
		for (i = 0; i < rows; i++)
			for (j = 0; j < cols; j++)
				hmat[i * cols + j] = frameFromVideo.at<Vec3b>(i, j)[0];
		dmat = NULL;
		cudaMalloc(&dmat, size);
		cudaMemcpy(dmat, hmat, size, cudaMemcpyHostToDevice);
		cnt_io += clock() - last_io;

		dim3 blk(32, 32);
		dim3 grid(rows / blk.x, cols / blk.y);
		blue << <grid, blk >> >(dmat, rows, cols);
		cnt += clock() - last;

		last_io = clock();
		err = cudaMemcpy(hmat, dmat, size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed while Memcpying back! %s\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		for (i = 0; i < rows; i++)
			for (j = 0; j < cols; j++)
				frameFromVideo.at<Vec3b>(i, j)[0] = hmat[i * cols + j];
		cnt_io += clock() - last_io;

		cnt += clock() - last;
		
		imshow("outputCamera", frameFromVideo);

		if (waitKey(30) >= 0) break;
	}
	printf("Total = %fms\n", 1.0*cnt / (1.0*CLOCKS_PER_SEC / 1000.0));
	printf("I/O = %fms\n", 1.0*cnt_io / (1.0*CLOCKS_PER_SEC / 1000.0));
}

int main()
{
	if (CV_MAJOR_VERSION < 3) {
		puts("Advise you update to OpenCV3");
	}
	Video();
	
	while (1);
	return 0;
}
