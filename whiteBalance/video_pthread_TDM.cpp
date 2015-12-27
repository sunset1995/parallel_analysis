#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <ctime>
#include <thread>

#define SHOW_INFO false
#define OUTPUT_VIDEO true
#define TD_MAX_SIZE 200

using namespace std;
using namespace cv;

int threadNum;
int sz;
vector<Mat> imgs;

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

// Setup video output
void whiteBalance(int rank) {

	int rows = imgs[0].rows;
	int cols = imgs[0].cols;
	int picSz = rows * cols;

	for(int id=rank; id<imgs.size()-1; id+=threadNum) {

		int bSum=0, gSum=0, rSum=0;
		int avg[3], base;
		
		for(int i=0; i<rows; ++i)
			for(int j=0; j<cols; ++j) {
				bSum += imgs[id].at<Vec3b>(i,j)[0];
				gSum += imgs[id].at<Vec3b>(i,j)[1];
				rSum += imgs[id].at<Vec3b>(i,j)[2];
			}

		avg[0] = bSum / picSz;
		avg[1] = gSum / picSz;
		avg[2] = rSum / picSz;

		if( SHOW_INFO )
			printf("avg(b, g, r): %d %d %d\n",avg[0], avg[1], avg[2]);

		base = avg[1];

		// let gAvg = bAvg = rAvg
		for(int i=0; i<rows; ++i)
			for(int j=0; j<cols; ++j) {
				imgs[id].at<Vec3b>(i,j)[0] = min(255, 
					(int)(base * imgs[id].at<Vec3b>(i,j)[0] / avg[0]));
				imgs[id].at<Vec3b>(i,j)[1] = min(255, 
					(int)(base * imgs[id].at<Vec3b>(i,j)[1] / avg[1]));
				imgs[id].at<Vec3b>(i,j)[2] = min(255,
					(int)(base * imgs[id].at<Vec3b>(i,j)[2] / avg[2]));
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

	clock_t Calculate=0, Input=0, Output=0;
	clock_t Total = clock(), Last;

	while( true ) {
		// input enough frames
		Last = clock();
		for(int i=0; i<TD_MAX_SIZE; ++i) {
			imgs.push_back(Mat());
			captureVideo >> imgs.back();
			if (imgs.back().empty()) break;
		}
		if( imgs[0].empty() ) break;
		Input += clock() - Last;

		// proc all got frames
		Last = clock();
		vector<thread> threads;
		for(int i=0; i<threadNum; ++i)
			threads.emplace_back(thread(whiteBalance, i));
		for(int i=0; i<threadNum; ++i)
			threads[i].join();
		Calculate += clock() - Last;

		if( OUTPUT_VIDEO ) {
			Last = clock();
			for(int i=0; i<imgs.size()-1; ++i)
				outputVideo << imgs[i];
			Output += clock() - Last;
		}
		imgs.clear();
	}

	Total = clock() - Total;

	printf("    Total: %fms (include time count)\n", 1.0*Total / (1.0*CLOCKS_PER_SEC / 1000.0));
	printf("    Input: %fms\n", 1.0*Input / (1.0*CLOCKS_PER_SEC / 1000.0));
	printf("   Output: %fms\n", 1.0*Output / (1.0*CLOCKS_PER_SEC / 1000.0));
	printf("Calculate: %fms\n", 1.0*Calculate / (1.0*CLOCKS_PER_SEC / 1000.0));

	return 0;
}

