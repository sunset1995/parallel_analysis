#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <algorithm>
#include <ctime>

using namespace std;
using namespace cv;

vector<vector<int> > mat;


void Video() {
	// Setup video capture device
	// Link it to the first capture device
	VideoCapture captureVideo;
	captureVideo.open("videoLarge.mp4");

	Mat frameFromVideo;
	double cnt = 0;
	while (true){
		captureVideo >> frameFromVideo;
		if (frameFromVideo.empty()) break;
		//imshow("origin", frameFromVideo);
		if (mat.empty()) // if it's empty, resize first !
		{
			mat.resize(frameFromVideo.rows);
			for (int i = 0; i < frameFromVideo.rows; ++i)
				mat[i].resize(frameFromVideo.cols);
		}
		for (int i = 0; i < frameFromVideo.rows; ++i)
			for (int j = 0; j < frameFromVideo.cols; ++j)
				mat[i][j] = frameFromVideo.at<Vec3b>(i, j)[0];

		double last = getTickCount();
		for (int i = 0; i < frameFromVideo.rows; i++)
			for (int j = 0; j < frameFromVideo.cols; j++) {
				int tmp = mat[i][j];
				for (int k = 0; k < ((i + j) >> 7); ++k)
					tmp += tmp >> 3;
				mat[i][j] = tmp;
			}
				
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
				frameFromVideo.at<Vec3b>(i, j)[0] = mat[i][j] * 255 / max_val;
		
		cnt += getTickCount() - last;
		//imshow("outputCamera", frameFromVideo);

		//if (waitKey(30) >= 0) break;
	}
	printf("%fms\n", cnt / (getTickFrequency()/1000.0));
}

int main(int argc, const char** argv){
	if (CV_MAJOR_VERSION < 3) {
		puts("Advise you update to OpenCV3");
	}
	Video();
	return 0;
}

