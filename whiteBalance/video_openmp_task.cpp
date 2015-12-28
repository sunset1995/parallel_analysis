#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <ctime>
#include <omp.h>

#define SHOW_INFO false
#define OUTPUT_VIDEO false
#define TD_MAX_SIZE 200

using namespace std;
using namespace cv;

VideoWriter setOutput(const VideoCapture &input) {
	// Reference from
	// http://docs.opencv.org/2.4/doc/tutorials/highgui/video-write/video-write.html

	// Acquire input size
	Size S = Size((int) input.get(CV_CAP_PROP_FRAME_WIDTH),
				  (int) input.get(CV_CAP_PROP_FRAME_HEIGHT));

	 // Get Codec Type- Int form
	int ex = static_cast<int>(input.get(CV_CAP_PROP_FOURCC));

	VideoWriter output;
	output.open("outputVideo.avi", ex, input.get(CV_CAP_PROP_FPS), S, true);

    return output;
}

int threadNum;

// Setup video output
void whiteBalance(Mat &img) {

	int rows = img.rows;
	int cols = img.cols;
	int picSz = rows * cols;

	int bSum=0, gSum=0, rSum=0;
	int avg[3], base;

	if( img.isContinuous() ) {
		cols *= rows;
		rows = 1;
	}
	
	for(int i=0; i<rows; ++i) {
		Vec3b *p = img.ptr<Vec3b>(i);
		for(int j=0; j<cols; ++j) {
			bSum += p[j][0];
			gSum += p[j][1];
			rSum += p[j][2];
		}
	}

	avg[0] = bSum / picSz;
	avg[1] = gSum / picSz;
	avg[2] = rSum / picSz;

	if( SHOW_INFO )
		printf("avg(b, g, r): %d %d %d\n",avg[0], avg[1], avg[2]);

	base = avg[1];

	int tableB[256], tableG[256], tableR[256];
	for(int i=0; i<256; ++i) {
		tableB[i] = min(255, base * i / avg[0]);
		tableG[i] = min(255, base * i / avg[1]);
		tableR[i] = min(255, base * i / avg[2]);
	}

	// let gAvg = bAvg = rAvg
	for(int i=0; i<rows; ++i) {
		Vec3b *p = img.ptr<Vec3b>(i);
		for(int j=0; j<cols; ++j) {
			p[j][0] = tableB[ p[j][0] ];
			p[j][1] = tableG[ p[j][1] ];
			p[j][2] = tableR[ p[j][2] ];
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

	int threadNum = atoi(argv[2]);
		if( SHOW_INFO )
			printf("threads: %d\n", threadNum);

	// Setup video output
	VideoWriter outputVideo;
	if( OUTPUT_VIDEO )
		outputVideo = setOutput(captureVideo);

	clock_t Calculate=0, Input=0, Output=0;
	clock_t Total = clock();

	omp_lock_t inputLck[TD_MAX_SIZE];
	omp_lock_t procLck[TD_MAX_SIZE];

	Mat imgs[TD_MAX_SIZE];
	while( true ) {

		for(int i=0; i<TD_MAX_SIZE; ++i) {
			omp_init_lock(&inputLck[i]);
			omp_init_lock(&procLck[i]);
			omp_set_lock(&inputLck[i]);
			omp_set_lock(&procLck[i]);
		}

		int sz = TD_MAX_SIZE;
		omp_set_num_threads(threadNum);
#		pragma omp parallel sections
		{
#			pragma omp section
			{
				int i;
				for(i=0; i<TD_MAX_SIZE; ++i) {
					clock_t Last = clock();
					captureVideo >> imgs[i];
					Input += clock() - Last;
					if (imgs[i].empty()) break;
					omp_unset_lock(&inputLck[i]);
				}
				sz = i;
				for( ; i<TD_MAX_SIZE; ++i)
					omp_unset_lock(&inputLck[i]);
			}

#			pragma omp section
			{
				for(int i=0; i<TD_MAX_SIZE; ++i) {
#					pragma omp task firstprivate(i)
					{
						omp_set_lock(&inputLck[i]);
						if(!imgs[i].empty()) {
							clock_t Last = clock();
							whiteBalance(imgs[i]);
							Calculate += clock() - Last;
						}
						omp_unset_lock(&procLck[i]);
					}
				}
			}

#			pragma omp section
			{
				if( OUTPUT_VIDEO )
					for(int i=0; i<TD_MAX_SIZE; ++i) {
						omp_set_lock(&procLck[i]);
						if(!imgs[i].empty()) {
							clock_t Last = clock();
							outputVideo << imgs[i];
							Output += clock() - Last;
						}
					}
			}
		}

		for(int i=0; i<TD_MAX_SIZE; ++i) {
			omp_destroy_lock(&inputLck[i]);
			omp_destroy_lock(&procLck[i]);
		}

		if( imgs[0].empty() || sz!=TD_MAX_SIZE ) break;

	}

	Total = clock() - Total;

	printf("    Total: %fms (include time count)\n", 1.0*Total / (1.0*CLOCKS_PER_SEC / 1000.0));
	printf("    Input: %fms\n", 1.0*Input / (1.0*CLOCKS_PER_SEC / 1000.0));
	printf("   Output: %fms\n", 1.0*Output / (1.0*CLOCKS_PER_SEC / 1000.0));
	printf("Calculate: %fms\n", 1.0*Calculate / (1.0*CLOCKS_PER_SEC / 1000.0));

	return 0;
}

