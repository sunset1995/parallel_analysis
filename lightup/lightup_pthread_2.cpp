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

vector<Mat> frameFromVideos;

void lightUp(int tid) {
	int threadPerNum = frameFromVideos.size()/threadNum;
	int from = tid * threadPerNum;
	int to = (tid==threadNum-1)? frameFromVideos.size() : from+threadPerNum;
	int rows = frameFromVideos[0].rows;
	int cols = frameFromVideos[0].cols;
	for(int id=from; id<to; ++id)
		for(int i=0; i<rows; ++i)
			for(int j=0; j<cols; ++j)
				frameFromVideos[id].at<Vec3b>(i,j)[0] = min(
					(frameFromVideos[id].at<Vec3b>(i,j)[0]<<1) + 5,
					255
				);
}

void Video() {

	// Setup video capture device
	// Link it to the first capture device
	VideoCapture captureVideo;
	captureVideo.open("videoLarge.mp4");

	while(true) {
		frameFromVideos.push_back(Mat());
		captureVideo >> frameFromVideos.back();
		if( frameFromVideos.back().empty() ) {
			frameFromVideos.pop_back();
			break;
		}

		if( waitKey(30)>=0 ) break;
	}

	clock_t cnt = clock();	

	vector<thread> threads;
	for(int i=0; i<threadNum; ++i)
		threads.emplace_back(thread(lightUp, i));
	for(int i=0; i<threadNum; ++i)
		threads[i].join();

	cnt = clock() - cnt;
	printf("%fms\n", 1.0*cnt / (1.0*CLOCKS_PER_SEC/1000.0));
	frameFromVideos.clear();
}

int main(int argc, const char** argv){
	if( CV_MAJOR_VERSION < 3 ) {
		puts("Advise you update to OpenCV3");
	}
	Video();
	return 0;
}

