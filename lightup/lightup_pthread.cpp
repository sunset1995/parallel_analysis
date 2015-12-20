#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <algorithm>
#include <ctime>
#include <thread>
#include <vector>

#define threadNum 4

using namespace std;
using namespace cv;

Mat frameFromVideo;

void lightUp(int tid) {
	int threadPerNum = frameFromVideo.rows/threadNum;
	int from = tid * threadPerNum;
	int to = (tid==threadNum-1)? frameFromVideo.rows : from+threadPerNum;
	for(int i=from; i<to; ++i)
		for(int j=0; j<frameFromVideo.cols; ++j)
			frameFromVideo.at<Vec3b>(i,j)[0] = min(
				(frameFromVideo.at<Vec3b>(i,j)[0]<<1) + 5,
				255
			);
}

void Video() {

	// Setup video capture device
	// Link it to the first capture device
	VideoCapture captureVideo;
	captureVideo.open("videoLarge.mp4");

	clock_t cnt = 0;
	while(true) {
		captureVideo >> frameFromVideo;
		if( frameFromVideo.empty() ) break;
		//imshow("origin", frameFromVideo);

		clock_t last = clock();
		vector<thread> threads;
		for(int i=0; i<threadNum; ++i)
			threads.emplace_back(thread(lightUp, i));
		for(int i=0; i<threadNum; ++i)
			threads[i].join();
		cnt += clock() - last;
		//imshow("outputCamera", frameFromVideo);

		if( waitKey(30)>=0 ) break;
	}
	printf("%fms\n", 1.0*cnt / (1.0*CLOCKS_PER_SEC/1000.0));
}

int main(int argc, const char** argv){
	if( CV_MAJOR_VERSION < 3 ) {
		puts("Advise you update to OpenCV3");
	}
	Video();
	return 0;
}

