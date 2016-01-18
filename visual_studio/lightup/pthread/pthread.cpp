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

vector<vector<int> > mat;
Mat frameFromVideo;

void lightUp(int tid) {
	int threadPerNum = frameFromVideo.rows / threadNum;
	int from = tid * threadPerNum;
	int to = (tid == threadNum - 1) ? frameFromVideo.rows : from + threadPerNum;
	for (int i = from; i < to; ++i)
		for (int j = 0; j < frameFromVideo.cols; ++j) {
			mat[i][j] = mat[i][j] * 2 + 5;
		}
}

void Video(const char **argv) {

	// Setup video capture device
	// Link it to the first capture device
	VideoCapture captureVideo;
	captureVideo.open(argv[1]);

	double cnt = 0;
	while (true) {
		captureVideo >> frameFromVideo;
		if (frameFromVideo.empty()) break;
		//imshow("origin", frameFromVideo);
		if (mat.empty()) // if it's empty, resize first !
		{
			mat.resize(frameFromVideo.rows);
			for (int i = 0; i < frameFromVideo.rows; ++i)
				mat[i].resize(frameFromVideo.cols);
		}
		for (int k = 0; k < 3; ++k) {

			double last = getTickCount();

			for (int i = 0; i < frameFromVideo.rows; ++i)
				for (int j = 0; j < frameFromVideo.cols; ++j)
					mat[i][j] = frameFromVideo.at<Vec3b>(i, j)[k];

			vector<thread> threads;
			for (int i = 0; i < threadNum; ++i)
				threads.emplace_back(thread(lightUp, i));
			for (int i = 0; i < threadNum; ++i)
				threads[i].join();


			// getting max value
			int max_val = 0;
			for (int i = 0; i < frameFromVideo.rows; i++)
				for (int j = 0; j < frameFromVideo.cols; j++) {
					if (mat[i][j] > max_val)
						max_val = mat[i][j];
				}

			// normalizing
			for (int i = 0; i < frameFromVideo.rows; i++)
				for (int j = 0; j < frameFromVideo.cols; j++)
					frameFromVideo.at<Vec3b>(i, j)[k] = mat[i][j] / max_val;

			cnt += getTickCount() - last;

		}

		// imshow("outputCamera", frameFromVideo);

		if (waitKey(30) >= 0) break;
	}
	printf("%fms\n", cnt / (getTickFrequency()/1000.0));
}

int main(int argc, const char** argv){
	if (CV_MAJOR_VERSION < 3) {
		puts("Advise you update to OpenCV3");
	}
	Video(argv);
	return 0;
}
