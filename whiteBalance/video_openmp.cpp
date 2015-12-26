#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>
#include <omp.h>

#define SHOW_INFO false
#define SHOW_VIDEO false

using namespace std;
using namespace cv;

struct BGR {
	int b, g, r;
	BGR(int b,int g,int r)
	:b(b), g(g), r(r) {}
};

int threadNum;

void whiteBalance(Mat &img) {

	int rows = img.rows;
	int cols = img.cols;
	int picSz = rows * cols;

	int bSum=0, gSum=0, rSum=0;
	int avg[3], base;

	omp_set_num_threads(threadNum);
	#pragma omp parallel
	{
		#pragma omp for reduction(+:bSum,gSum,rSum)
		for(int i=0; i<rows; ++i)
			for(int j=0; j<cols; ++j) {
				bSum += img.at<Vec3b>(i,j)[0];
				gSum += img.at<Vec3b>(i,j)[1];
				rSum += img.at<Vec3b>(i,j)[2];
			}

		#pragma omp single
		{
		avg[0] = bSum / picSz;
		avg[1] = gSum / picSz;
		avg[2] = rSum / picSz;

		if( SHOW_INFO )
			printf("avg(b, g, r): %d %d %d\n",avg[0], avg[1], avg[2]);

		base = avg[1];
		}

		// let gAvg = bAvg = rAvg
		#pragma omp for nowait
		for(int i=0; i<rows; ++i)
			for(int j=0; j<cols; ++j) {
				img.at<Vec3b>(i,j)[0] = min(255, 
					(int)(base * img.at<Vec3b>(i,j)[0] / avg[0]));
				img.at<Vec3b>(i,j)[1] = min(255, 
					(int)(base * img.at<Vec3b>(i,j)[1] / avg[1]));
				img.at<Vec3b>(i,j)[2] = min(255,
					(int)(base * img.at<Vec3b>(i,j)[2] / avg[2]));
			}
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

	VideoCapture captureVideo(argv[1]);
	if( !captureVideo.isOpened() ) {
		puts("Fail to open video");
		return 0;
	}

	threadNum = atoi(argv[2]);
	if( SHOW_INFO )
		printf("threads: %d\n", threadNum);

	clock_t cnt = clock();

	Mat img;
	while( true ) {
		captureVideo >> img;
		if (img.empty()) break;
		if( SHOW_VIDEO ) imshow("origin", img);
		whiteBalance(img);
		if( SHOW_VIDEO ) imshow("video", img);
		if( waitKey(30)>=0 ) break;
	}

	cnt += clock() - cnt;
	printf("%fms\n", 1.0*cnt / (1.0*CLOCKS_PER_SEC / 1000.0));

	return 0;
}

