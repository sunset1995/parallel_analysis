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
#define TD_MAX_SIZE 500

using namespace std;
using namespace cv;

int threadNum;
int sz;
Mat imgs[TD_MAX_SIZE];

VideoWriter setOutput(const VideoCapture &input) {
	// Reference from
	// http://docs.opencv.org/2.4/doc/tutorials/highgui/video-write/video-write.html

	// Acquire input size
	Size S = Size((int)input.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)input.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Get Codec Type- Int form
	int ex = static_cast<int>(input.get(CV_CAP_PROP_FOURCC));

	VideoWriter output;
	output.open("outputVideo.avi", CV_FOURCC('H', 'F', 'Y', 'U'), input.get(CV_CAP_PROP_FPS), S, true);

	return output;
}

// Setup video output
void whiteBalance(int rank, int sz) {

	for (int id = rank; id < sz; id += threadNum) {

		int rows = imgs[id].rows;
		int cols = imgs[id].cols;
		int picSz = rows * cols;

		if (imgs[id].isContinuous()) {
			cols *= rows;
			rows = 1;
		}

		int bSum = 0, gSum = 0, rSum = 0;
		double avg[3], base;

		for (int i = 0; i<rows; ++i) {
			Vec3b *p = imgs[id].ptr<Vec3b>(i);
			for (int j = 0; j<cols; ++j) {
				bSum += p[j][0];
				gSum += p[j][1];
				rSum += p[j][2];
			}
		}

		avg[0] = (double)bSum / picSz;
		avg[1] = (double)gSum / picSz;
		avg[2] = (double)rSum / picSz;

		if (SHOW_INFO)
			printf("avg(b, g, r): %f %f %f\n", avg[0], avg[1], avg[2]);

		base = avg[1];

		int tableB[256], tableG[256], tableR[256];
		for (int i = 0; i<256; ++i) {
			tableB[i] = min(255, (int)(base * i / avg[0]));
			tableG[i] = min(255, (int)(base * i / avg[1]));
			tableR[i] = min(255, (int)(base * i / avg[2]));
		}

		// let gAvg = bAvg = rAvg
		for (int i = 0; i<rows; ++i) {
			Vec3b *p = imgs[id].ptr<Vec3b>(i);
			for (int j = 0; j<cols; ++j) {
				p[j][0] = tableB[p[j][0]];
				p[j][1] = tableG[p[j][1]];
				p[j][2] = tableR[p[j][2]];
			}
		}
	}
}

void inputVideo(const char *filePath, int rank, int sz, int fid) {
	int numPerThread = sz / threadNum;
	int from = rank * numPerThread;
	int to = (rank==threadNum-1)? sz : from + numPerThread;
	VideoCapture cpVideo(filePath);
	cpVideo.set(CV_CAP_PROP_POS_FRAMES, fid+from);

	for (int i=from; i<to; ++i)
		cpVideo >> imgs[i];
}

int main(int argc, const char** argv){
	if (CV_MAJOR_VERSION < 3) {
		puts("Advise you update to OpenCV3");
	}
	if (argc<2) {
		puts("Please specify input image path");
		return 0;
	}
	if (argc<3) {
		puts("Please specify thread num");
		return 0;
	}

	VideoCapture captureVideo(argv[1]);
	if (!captureVideo.isOpened()) {
		puts("Fail to open video");
		return 0;
	}

	// Setup video output
	VideoWriter outputVideo;
	if (OUTPUT_VIDEO)
		outputVideo = setOutput(captureVideo);

	threadNum = atoi(argv[2]);
	if (SHOW_INFO)
		printf("threads: %d\n", threadNum);
	outputVideo.set(CV_CAP_PROP_BUFFERSIZE, 1);

	double Calculate = 0, Input = 0, Output = 0;
	double Total = getTickCount(), Last;


	int numFrames = captureVideo.get(CV_CAP_PROP_FRAME_COUNT);
	for (int fid = 0; fid<numFrames; fid += TD_MAX_SIZE) {

		int sz = numFrames - fid;
		if (sz > TD_MAX_SIZE) sz = TD_MAX_SIZE;

		// store all thread
		vector<thread> threads;

		// input enough frames
		Last = getTickCount();
		for (int i = 0; i<threadNum; ++i)
			threads.emplace_back(thread(inputVideo, argv[1], i, sz, fid));
		for (int i = 0; i<threadNum; ++i)
			threads[i].join();
		threads.clear();
		Input += getTickCount() - Last;

		// proc all got frames
		Last = getTickCount();
		for (int i = 0; i<threadNum; ++i)
			threads.emplace_back(thread(whiteBalance, i, sz));
		for (int i = 0; i<threadNum; ++i)
			threads[i].join();
		threads.clear();
		Calculate += getTickCount() - Last;

		if (OUTPUT_VIDEO) {
			Last = getTickCount();
			for (int i = 0; i<sz; ++i) {
				outputVideo << imgs[i];
				imgs[i].release();
			}
			Output += getTickCount() - Last;
		}
	}

	Total = getTickCount() - Total;

	printf("    Total: %.3fs (include time count)\n", Total / getTickFrequency());
	printf("    Input: %.3fs\n", Input / getTickFrequency());
	printf("   Output: %.3fs\n", Output / getTickFrequency());
	printf("Calculate: %.3fs\n", Calculate / getTickFrequency());

	return 0;
}

