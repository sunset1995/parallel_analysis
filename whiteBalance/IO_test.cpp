#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>

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

// Setup video output
VideoWriter outputVideo;

int main(int argc, const char** argv){
	if (CV_MAJOR_VERSION < 3) {
		puts("Advise you update to OpenCV3");
	}
	if( argc<2 ) {
		puts("Please specify input image path");
		return 0;
	}

	VideoCapture captureVideo(argv[1]);
	if( !captureVideo.isOpened() ) {
		puts("Fail to open video");
		return 0;
	}

	outputVideo = setOutput(captureVideo);

	clock_t cnt = clock();

	Mat imgs[200];
	while( true ) {
		for(int i=0; i<200; ++i) {
			captureVideo >> imgs[i];
			if( imgs[i].empty() ) break;
		}
		if( imgs[0].empty() ) break;
		for(int i=0; i<200; ++i) {
			if( imgs[i].empty() ) break;
			outputVideo << imgs[i];
		}
	}

	cnt += clock() - cnt;
	printf("%fms\n", 1.0*cnt / (1.0*CLOCKS_PER_SEC / 1000.0));

	return 0;
}

