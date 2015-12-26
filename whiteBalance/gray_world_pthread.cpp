#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>
#include <pthread.h>

using namespace std;
using namespace cv;

struct BGR {
	int b, g, r;
	BGR(int b,int g,int r)
	:b(b), g(g), r(r) {}
};

Mat img;
int threadNum;

int avg[3];

vector< vector<int> > local_int_ret;
void* sumBGR(void *rank) {
	int tid = (long)rank;
	int cols = img.cols;
	int nums = img.rows / threadNum;
	int from = tid * nums;
	int to = (tid==threadNum-1)? img.rows : from + nums;
	int localB=0, localG=0, localR=0;
	for(int i=from; i<to; ++i)
		for(int j=0; j<cols; ++j) {
			localB += img.at<Vec3b>(i,j)[0];
			localG += img.at<Vec3b>(i,j)[1];
			localR += img.at<Vec3b>(i,j)[2];
		}
	local_int_ret[tid][0] = localB;
	local_int_ret[tid][1] = localG;
	local_int_ret[tid][2] = localR;
}

void* normal(void *rank) {
	int tid = (long)rank;
	int avgB = avg[0];
	int avgG = avg[1];
	int avgR = avg[2];
	int base = avg[1];

	int cols = img.cols;
	int nums = img.rows / threadNum;
	int from = tid * nums;
	int to = (tid==threadNum-1)? img.rows : from + nums;
	for(int i=from; i<to; ++i)
		for(int j=0; j<cols; ++j) {
			img.at<Vec3b>(i,j)[0] = min(255, 
				(int)(base * img.at<Vec3b>(i,j)[0] / avgB));
			img.at<Vec3b>(i,j)[1] = min(255,
				(int)(base * img.at<Vec3b>(i,j)[1] / avgG));
			img.at<Vec3b>(i,j)[2] = min(255,
				(int)(base * img.at<Vec3b>(i,j)[2] / avgR));
		}
}

int main(int argc, const char** argv){
	if (CV_MAJOR_VERSION < 3) {
		puts("Advise you update to OpenCV3");
	}
	if( argc<2 ) {
		puts("Please specify input image path");
		return 0;
	}
	if( argc<3 ) {
		puts("Please specify thread num");
		return 0;
	}

	// Read an image
	// The 3 channels are BGR (Blue, Green, and Red)
	img = imread(argv[1], 1);
	if( !img.data ) {
		puts("read image failed");
		return 0;
	}
	imshow("origin", img);
	int rows = img.rows;
	int cols = img.cols;
	int picSz = rows * cols;

	threadNum = atoi(argv[2]);
	pthread_t *threads;
	threads = (pthread_t*)malloc(threadNum * sizeof(pthread_t));

	clock_t cnt = clock();

	local_int_ret.resize(threadNum);
	for(int i=0; i<threadNum; ++i)
		local_int_ret[i].resize(3);

	threads = (pthread_t*)malloc(threadNum * sizeof(pthread_t));
	for(long i=0; i<threadNum; ++i)
		pthread_create(&threads[i] , NULL, sumBGR, (void*)i);
	for(int i=0; i<threadNum; ++i)
		pthread_join(threads[i] , NULL);
	free(threads);

	int bSum=0, gSum=0, rSum=0;
	for(int i=0; i<threadNum; ++i) {
		bSum += local_int_ret[i][0];
		gSum += local_int_ret[i][1];
		rSum += local_int_ret[i][2];
	}

	avg[0] = bSum / picSz;
	avg[1] = gSum / picSz;
	avg[2] = rSum / picSz;
	printf("avg(b, g, r): %d %d %d\n",avg[0], avg[1], avg[2]);

	// let gAvg = bAvg = rAvg
	threads = (pthread_t*)malloc(threadNum * sizeof(pthread_t));
	for(long i=0; i<threadNum; ++i)
		pthread_create(&threads[i] , NULL, normal, (void*)i);
	for(int i=0; i<threadNum; ++i)
		pthread_join(threads[i] , NULL);
	free(threads);

	cnt += clock() - cnt;
	printf("%fms\n", 1.0*cnt / (1.0*CLOCKS_PER_SEC / 1000.0));

	//imshow("white balance", img);
	//waitKey(0);
	return 0;
}

