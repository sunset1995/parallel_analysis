
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



__global__ void blue(int *mat, int height, int width)
{
	int i = blockIdx.x * 32 + threadIdx.x
		, j = blockIdx.y * 32 + threadIdx.y;
	if (i < height && j < width) {
		int &mi = mat[i * width + j];
		mi = mi * 2 + 5;
	}
}


void Video(const char **argv) {
	// Setup video capture device
	// Link it to the first capture device
	cudaError_t err;
	VideoCapture captureVideo;
	captureVideo.open(argv[1]);

	int i, j;
	Mat frameFromVideo;
	double cnt = 0;

	int *hmat, *dmat = NULL;

	while (true){
		captureVideo >> frameFromVideo;
		if (frameFromVideo.empty()) break;
		// imshow("origin", frameFromVideo);

		// for (int k = 0; k < 3; ++k) {

			int rows = frameFromVideo.rows, cols = frameFromVideo.cols;
			int size = rows * cols * sizeof(int);
			hmat = (int *)malloc(size);
			for (i = 0; i < rows; i++)
				for (j = 0; j < cols; j++)
					hmat[i * cols + j] = frameFromVideo.at<Vec3b>(i, j)[0];

			double last = getTickCount();

			dmat = NULL;
			cudaMalloc(&dmat, size);
			cudaMemcpy(dmat, hmat, size, cudaMemcpyHostToDevice);

			dim3 blk(32, 32);
			dim3 grid(rows / blk.x, cols / blk.y);
			blue << <grid, blk >> >(dmat, rows, cols);
			cnt += getTickCount() - last;

			err = cudaMemcpy(hmat, dmat, size, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess)
			{
				fprintf(stderr, "Failed while Memcpying back! %s\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}

			// getting max value
			int max_val = 0;
			for (i = 0; i < rows; i++)
				for (j = 0; j < cols; j++) {
					if (hmat[i * cols + j] > max_val)
						max_val = hmat[i * cols + j];
				}

			// normalizing
			for (i = 0; i < rows; i++)
				for (j = 0; j < cols; j++)
					frameFromVideo.at<Vec3b>(i, j)[0] = hmat[i * cols + j] * 255 / max_val;

			cudaFree(dmat);
			free(hmat);

			cnt += getTickCount() - last;

		// }

		// imshow("outputCamera", frameFromVideo);

		if (waitKey(30) >= 0) break;
	}
	printf("Total = %fms\n", 1.0*cnt / (getTickFrequency() / 1000.0));
}

int main(int argc, const char** argv){
	if (CV_MAJOR_VERSION < 3) {
		puts("Advise you update to OpenCV3");
	}
	Video(argv);

	return 0;
}