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

vector<vector<vector<int> > > mats;
vector<Mat> frameFromVideos;

void lightUp(int tid) {
	int threadPerNum = frameFromVideos.size() / threadNum;
	int from = tid * threadPerNum;
	int to = (tid == threadNum - 1) ? frameFromVideos.size() : from + threadPerNum;
	int rows = frameFromVideos[0].rows;
	int cols = frameFromVideos[0].cols;
	for (int id = from; id<to; ++id)
		for (int i = 0; i<rows; ++i)
			for (int j = 0; j<cols; ++j)
				for (int k = 0; k < ((i + j) >> 7); ++k)
					mats[id][i][j] += mats[id][i][j] >> 3;
}

void Video(const char **argv) {

	// Setup video capture device
	// Link it to the first capture device
	VideoCapture captureVideo;
	captureVideo.open(argv[1]);

	while (true) {
		frameFromVideos.push_back(Mat());
		captureVideo >> frameFromVideos.back();
		if (frameFromVideos.back().empty()) {
			frameFromVideos.pop_back();
			break;
		}

		if (waitKey(30) >= 0) break;
	}
	mats.resize(frameFromVideos.size());
	for (int id = 0; id < mats.size(); ++id) {
		mats[id].resize(frameFromVideos[0].rows);
		for (int i = 0; i < frameFromVideos[0].rows; ++i) {
			mats[id][i].resize(frameFromVideos[0].cols);
			for (int j = 0; j < frameFromVideos[0].cols; ++j)
				mats[id][i][j] = frameFromVideos[id].at<Vec3b>(i, j)[0];
		}
	}

	clock_t cnt = clock();

	vector<thread> threads;
	for (int i = 0; i<threadNum; ++i)
		threads.emplace_back(thread(lightUp, i));
	for (int i = 0; i<threadNum; ++i)
		threads[i].join();

	for (int id = 0; id < frameFromVideos.size(); ++id) {
		// getting max value
		int max_val = 0;
		for (int i = 0; i < frameFromVideos[id].rows; i++)
			for (int j = 0; j < frameFromVideos[id].cols; j++) {
				if (mats[id][i][j] > max_val)
					max_val = mats[id][i][j];
			}

		// normalizing
		for (int i = 0; i < frameFromVideos[id].rows; i++)
			for (int j = 0; j < frameFromVideos[id].cols; j++)
				frameFromVideos[id].at<Vec3b>(i, j)[0] = mats[id][i][j] / max_val;
	}

	cnt = clock() - cnt;

	printf("%fms\n", 1.0*cnt / (1.0*CLOCKS_PER_SEC / 1000.0));
	frameFromVideos.clear();
}

int main(int argc, const char** argv){
	if (CV_MAJOR_VERSION < 3) {
		puts("Advise you update to OpenCV3");
	}
	Video(argv);
	return 0;
}
