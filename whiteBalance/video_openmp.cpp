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
#define OUTPUT_VIDEO true

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
	output.open("outputVideo.avi", CV_FOURCC('P','I','M','1'), input.get(CV_CAP_PROP_FPS), S, true);

    return output;
}

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
	
	if( img.isContinuous() ) {
		cols *= rows;
		rows = 1;
	}
	
	int bSum=0, gSum=0, rSum=0;
	int avg[3], base;

	int tableB[256], tableG[256], tableR[256];

	omp_set_num_threads(threadNum);
	#pragma omp parallel
	{
		#pragma omp for reduction(+:bSum,gSum,rSum)
		for(int i=0; i<rows; ++i) {
			Vec3b *p = img.ptr<Vec3b>(i);
			for(int j=0; j<cols; ++j) {
				bSum += p[j][0];
				gSum += p[j][1];
				rSum += p[j][2];
			}
		}

		#pragma omp single
		{
			avg[0] = bSum / picSz;
			avg[1] = gSum / picSz;
			avg[2] = rSum / picSz;

			if( SHOW_INFO )
				printf("avg(b, g, r): %d %d %d\n",avg[0], avg[1], avg[2]);

			base = avg[1];

			for(int i=0; i<256; ++i) {
				tableB[i] = min(255, base * i / avg[0]);
				tableG[i] = min(255, base * i / avg[1]);
				tableR[i] = min(255, base * i / avg[2]);
			}
		}

		// let gAvg = bAvg = rAvg
		#pragma omp for nowait
		for(int i=0; i<rows; ++i) {
			Vec3b *p = img.ptr<Vec3b>(i);
			for(int j=0; j<cols; ++j) {
				p[j][0] = tableB[ p[j][0] ];
				p[j][1] = tableG[ p[j][1] ];
				p[j][2] = tableR[ p[j][2] ];
			}
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

	// Setup video output
	VideoWriter outputVideo;
	if( OUTPUT_VIDEO )
		outputVideo = setOutput(captureVideo);

	threadNum = atoi(argv[2]);
	if( SHOW_INFO )
		printf("threads: %d\n", threadNum);

	double Calculate=0, Input=0, Output=0;
	double Total = getTickCount(), Last;

	Mat img;
	while( true ) {
		Last = getTickCount();
		captureVideo >> img;
		if (img.empty()) break;
		Input += getTickCount() - Last;

		Last = getTickCount();
		whiteBalance(img);
		Calculate += getTickCount() - Last;

		if( OUTPUT_VIDEO ) {
			Last = getTickCount();
			outputVideo << img;
			Output += getTickCount() - Last;
		}
	}

	Total = getTickCount() - Total;
	
	printf("    Total: %.3fs (include time count)\n", Total / getTickFrequency() );
	printf("    Input: %.3fs\n", Input / getTickFrequency() );
	printf("   Output: %.3fs\n", Output / getTickFrequency() );
	printf("Calculate: %.3fs\n", Calculate / getTickFrequency() );

	return 0;
}

